"""Compare Marlin's INT4 W4A16 w13 + silu·mul output against an FP32 oracle,
on the EXACT same inputs that the spec describes (and that the kernel sees).

Goal: determine whether the kernel's max relative error ≈ 2.0 / max abs error
≈ 0.023 against the FP32 dequant→BF16 reference is:
  (a) a normal BF16 / K=7168 noise floor that Marlin itself also has, OR
  (b) a bug specific to the agent's v040 kernel.

If (a): Marlin's output will show similar (maxre, maxae) vs the reference
        → the spec's `maxre<5e-3 AND maxae<1.0` is unsatisfiable, v040 is not
          buggy, the validator is mis-specified.
If (b): Marlin's output will be much closer to the reference (e.g. maxre<5e-3)
        → v040 has a real arithmetic bug.

Reference is identical in semantics to v040's `reference()` function:
  - Dequant: bf16( (nibble - 8) * scale )  [nibble is offset-8 unsigned, signed = nibble - 8]
  - Matmul:  fp32 += bf16_to_fp32(W_bf16) * bf16_to_fp32(A_bf16)
  - Epilogue: silu(gate)*up in fp32, cast to bf16
"""
import sys, math
import torch
import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation, apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.scalar_type import scalar_types

device = "cuda"
DTYPE = torch.bfloat16

H, I_R = 7168, 256
N_R = 2 * I_R   # 512
E_LOC = 384
TOP_K = 8
PACK_FACTOR = 8
GROUP_SIZE = 32
BLOCK_SIZE_M = 16
QUANT_TYPE = scalar_types.uint4b8

torch.manual_seed(42)


def fp32_oracle_w13_silu(hidden, raw_q, scales, topk_ids):
    """Compute the dequant→bf16, then matmul (fp32 accum), then silu*mul, then cast to bf16.
    Per spec — matches v040 kernel's reference() function.

    Args:
      hidden  : bf16 [T, H]
      raw_q   : int32 [E, H/8, N]  (8 nibbles per int32, K-major, low-nibble first, offset-8)
      scales  : bf16 [E, H/32, N]  (per-group)
      topk_ids: int32 [T, TOP_K]
    Returns:
      out_bf16: bf16 [P, I_R]  where P = T*TOP_K (one row per routing pair)
    """
    T, _ = hidden.shape
    P = T * TOP_K
    out = torch.zeros(P, I_R, dtype=torch.float32, device=device)

    # Pre-dequant all weights to bf16 once (small enough for E=384 * 512 * 7168 = 1.4 GB bf16)
    # Save memory by dequanting per-expert as needed.
    # We dequant only ACTIVE experts.
    active_eids = torch.unique(topk_ids).tolist()
    dequanted = {}  # e -> bf16 [N_R, H]
    for e in active_eids:
        # raw_q[e] is [H/8, N_R] int32
        # we want W[e] of shape [N_R, H] in bf16
        raw = raw_q[e].to(torch.int32)             # [H/8, N_R]
        # Unpack: 8 nibbles per int32, low nibble first
        # Build [H, N_R] uint values then transpose to [N_R, H]
        H_packs = raw.shape[0]
        # nibbles: [H_packs * 8, N_R]
        u = raw.unsqueeze(0).expand(8, -1, -1)     # [8, H/8, N_R]
        shifts = torch.arange(8, device=device, dtype=torch.int32) * 4  # [8]
        nibbles = (u >> shifts.view(8, 1, 1)) & 0xF                       # [8, H/8, N_R]
        # Interleave so that nibble[s] of pack[k] is at row k*8 + s
        # We want flat K-axis: row order is pack0:nib0, pack0:nib1, ..., pack0:nib7, pack1:nib0, ...
        nibbles = nibbles.permute(1, 0, 2).contiguous()  # [H/8, 8, N_R]
        flat = nibbles.view(H, N_R)                       # [H, N_R]   K is fastest-varying within a pack
        signed = (flat.to(torch.float32) - 8.0)            # [H, N_R]
        # Apply per-group scale: scales[e] is [H/32, N_R]
        sc = scales[e].to(torch.float32)                   # [H/32, N_R]
        sc_expanded = sc.repeat_interleave(GROUP_SIZE, dim=0)  # [H, N_R]
        w_bf16 = ((signed * sc_expanded).to(torch.bfloat16)).to(torch.float32)
        # [H, N_R] → [N_R, H]
        dequanted[e] = w_bf16.t().contiguous()             # [N_R, H] fp32

    a_fp32 = hidden.to(torch.float32)                       # [T, H]
    for t in range(T):
        for k in range(TOP_K):
            e = int(topk_ids[t, k].item())
            W = dequanted[e]                                # [N_R, H] fp32 (already bf16-rounded values)
            gate_up = a_fp32[t] @ W.t()                     # [N_R] fp32
            gate = gate_up[:I_R]
            up = gate_up[I_R:]
            silu = gate / (1.0 + torch.exp(-gate))
            out[t * TOP_K + k] = silu * up

    # Cast accumulator to bf16 to compare against Marlin's bf16 output
    return out.to(torch.bfloat16)


def main():
    print(f"H={H}, I_R={I_R}, N_R={N_R}, E_LOC={E_LOC}, TOP_K={TOP_K}", file=sys.stderr)

    for T in (1, 8):
        P = T * TOP_K
        print(f"\n=== T={T}, P={P} ===", file=sys.stderr)

        # ── inputs (same shape as baseline_marlin.py) ────────────────────
        g = torch.Generator(device=device).manual_seed(42 + T)

        # Raw INT4 weights (compressed-tensors layout)
        raw_q = torch.randint(0, 2**31 - 1, (E_LOC, H // PACK_FACTOR, N_R),
                              dtype=torch.int32, device=device, generator=g)
        # Marlin layout
        perm_empty = torch.empty((E_LOC, 0), dtype=torch.int32, device=device)
        marlin_q = ops.gptq_marlin_moe_repack(raw_q, perm_empty,
                                               size_k=H, size_n=N_R, num_bits=4,
                                               is_a_8bit=False)
        # Scales
        scales = (torch.randn(E_LOC, H // GROUP_SIZE, N_R, generator=g, device=device) * 0.01).to(DTYPE)

        # Routing
        topk_ids = torch.empty((T, TOP_K), dtype=torch.int32, device=device)
        for t in range(T):
            topk_ids[t] = torch.randperm(E_LOC, generator=g, device=device)[:TOP_K].to(torch.int32)
        topk_weights = torch.full((T, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)

        # Hidden
        hidden = (torch.randn(T, H, device=device, generator=g) * 0.01).to(DTYPE)

        # ── Marlin output ────────────────────────────────────────────────
        sorted_ids, expert_ids, num_post = moe_align_block_size(
            topk_ids, BLOCK_SIZE_M, E_LOC, expert_map=None, ignore_invalid_experts=True,
        )
        inter_cache = torch.empty(P, N_R, dtype=DTYPE, device=device)
        final_marlin = torch.empty(P, I_R, dtype=DTYPE, device=device)
        ws = torch.zeros(1024, dtype=torch.int32, device=device)

        ops.moe_wna16_marlin_gemm(
            hidden, inter_cache, marlin_q, None, scales, None, None, None, None, None,
            ws, sorted_ids, expert_ids, num_post, topk_weights,
            moe_block_size=BLOCK_SIZE_M, top_k=TOP_K, mul_topk_weights=False,
            b_q_type=QUANT_TYPE, size_m=T, size_n=N_R, size_k=H,
            is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
        )
        apply_moe_activation(MoEActivation.SILU, final_marlin, inter_cache)
        torch.cuda.synchronize()

        # ── FP32 oracle (matches v040 reference) ─────────────────────────
        oracle = fp32_oracle_w13_silu(hidden, raw_q, scales, topk_ids)
        torch.cuda.synchronize()

        # ── error stats ──────────────────────────────────────────────────
        m_f = final_marlin.to(torch.float32)
        o_f = oracle.to(torch.float32)
        ae = (m_f - o_f).abs()
        maxae = ae.max().item()
        # Same denominator rule the kernel uses: max(|ref|, |kernel|, 1e-6)
        denom = torch.maximum(torch.maximum(o_f.abs(), m_f.abs()),
                              torch.tensor(1e-6, device=device))
        re = (ae / denom)
        maxre = re.max().item()

        # Distribution at multiple epsilons
        for eps in (1e-6, 1e-3, 1e-2, 1e-1):
            d2 = torch.maximum(torch.maximum(o_f.abs(), m_f.abs()),
                               torch.tensor(eps, device=device))
            r2 = (ae / d2).max().item()
            print(f"  eps={eps:.0e}  maxre = {r2:.4f}", file=sys.stderr)

        print(f"  Marlin vs FP32 oracle:  maxae = {maxae:.4f}   maxre = {maxre:.4f}   "
              f"(eps=1e-6, same metric kernel uses)", file=sys.stderr)
        # Spec strict-AND test
        passes_AND = (maxre < 5e-3) and (maxae < 1.0)
        passes_OR  = (maxre < 5e-3) or  (maxae < 1.0)
        print(f"  passes strict AND (re<5e-3 AND ae<1.0): {passes_AND}", file=sys.stderr)
        print(f"  passes loose OR  (re<5e-3 OR  ae<1.0):  {passes_OR}", file=sys.stderr)


if __name__ == "__main__":
    main()
