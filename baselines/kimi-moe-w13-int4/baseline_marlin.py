"""Marlin baseline for the Kimi K2.6/K2.7 MoE w13 (gate+up) INT4 grouped GEMM
+ silu_and_mul fused operation.

Setup (TP=8 EP=1):
  H=7168, I_r=256, 2I_r=512, E_loc=384, top_k=8, T ∈ {1, 8}.

Reference invokes:
  intermediate = ops.moe_wna16_marlin_gemm(input, ..., w_q, ..., w_scale, ...)
  output       = apply_moe_activation(MoEActivation.SILU, output, intermediate)

→ output is [P, I_r=256] bf16, the same operation the agent's kernel must
produce. Times both calls as a single fused-equivalent region.

Methodology mirrors the ferret task spec:
  - 128 MB L2 flush per timed iteration
  - 2-sec continuous warmup
  - 300-iter median over cudaEvent timings

Run:
  /home/xinhaoc/vllm-venv/bin/python3 baseline_marlin.py
"""
import argparse
import sys
import time

import torch
import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.scalar_type import scalar_types

device = "cuda"
DTYPE = torch.bfloat16

# Kimi K2.6/K2.7 MoE config (TP=8 EP=1)
H = 7168
I_R = 256
N_R = 2 * I_R  # 512 — gate ‖ up
E_LOC = 384
TOP_K = 8
PACK_FACTOR = 8  # 8 int4 nibbles per int32
GROUP_SIZE = 32  # K-axis group size for scales
BLOCK_SIZE_M = 16  # Marlin's tile-M; per vllm fused_marlin_moe default

QUANT_TYPE = scalar_types.uint4b8  # compressed-tensors INT4 sym, offset-8 unsigned

FLUSH_BYTES = 128 * 1024 * 1024
WARMUP_SEC = 2.0
NI = 300

CONFIGS = [
    ("T1", 1),
    ("T8", 8),
]


def build_marlin_weights(E, K, N, *, seed=0):
    """Create random INT4 weight (compressed-tensors layout) + bf16 scales,
    then run gptq_marlin_moe_repack to produce Marlin's permuted format."""
    g = torch.Generator(device=device).manual_seed(seed)

    # Raw INT4 weight (compressed-tensors): [E, K/PACK_FACTOR, N] int32, K-major
    # Each int32 holds 8 nibbles of K-dim values; nibble range is offset-8
    # unsigned (0..15) → signed (-8..7) after subtraction.
    raw_q = torch.randint(
        0, 2**31 - 1, (E, K // PACK_FACTOR, N),
        dtype=torch.int32, device=device, generator=g,
    )

    # Identity perm (sym-no-zp doesn't need group re-permutation).
    perm = torch.empty((E, 0), dtype=torch.int32, device=device)

    # Repack into Marlin's internal layout.
    # Output shape: (E, K/16, N * (num_bits/2)) = (E, K/16, 2N) int32.
    marlin_q = ops.gptq_marlin_moe_repack(
        raw_q, perm, size_k=K, size_n=N, num_bits=4, is_a_8bit=False,
    )

    # bf16 scales, group-32 along K: [E, K/GROUP_SIZE, N]
    scales = (
        torch.randn(E, K // GROUP_SIZE, N, generator=g, device=device) * 0.01
    ).to(DTYPE)

    return marlin_q, scales


def build_routing(T, E, top_k, *, seed=0):
    """Balanced random routing: each of T tokens picks top_k distinct experts."""
    g = torch.Generator(device=device).manual_seed(seed)
    topk_ids = torch.empty((T, top_k), dtype=torch.int32, device=device)
    for t in range(T):
        # distinct experts via randperm; take first top_k
        perm = torch.randperm(E, generator=g, device=device)[:top_k]
        topk_ids[t] = perm.to(torch.int32)
    # uniform weights (router gating values) — values don't affect kernel cost
    topk_weights = torch.full(
        (T, top_k), 1.0 / top_k, dtype=torch.float32, device=device
    )
    return topk_ids, topk_weights


def make_workspace(M, top_k, N, num_experts):
    # vllm uses a small workspace tensor for marlin's scratch state.
    # Reference: vllm/model_executor/layers/fused_moe/marlin_utils.py
    # workspace is int32, shape (max_par_par * sm_count,) — small (~1KB).
    ws = torch.zeros(1024, dtype=torch.int32, device=device)
    return ws


def bench_one(T):
    P = T * TOP_K  # routed pairs

    # ── weights (one-time setup, NOT timed) ────────────────────────────
    w13_q, w13_scale = build_marlin_weights(E_LOC, H, N_R)

    # ── routing (one-time, NOT timed) ──────────────────────────────────
    topk_ids, topk_weights = build_routing(T, E_LOC, TOP_K)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, BLOCK_SIZE_M, E_LOC, expert_map=None,
        ignore_invalid_experts=True,
    )

    # Hidden states
    hidden = (torch.randn(T, H, device=device) * 0.01).to(DTYPE)

    # Output buffers
    inter_cache = torch.empty(P, N_R, dtype=DTYPE, device=device)   # post-GEMM, before silu
    final_out = torch.empty(P, I_R, dtype=DTYPE, device=device)     # post-silu·mul

    # Workspace
    ws = make_workspace(T, TOP_K, N_R, E_LOC)

    # ── single fused-equivalent call (GEMM + silu·mul) ────────────────
    def run():
        ops.moe_wna16_marlin_gemm(
            hidden,                  # input [T, H] bf16
            inter_cache,             # output [P, N_R] bf16
            w13_q,                   # b_qweight [E, K/16, 2N] int32 (Marlin)
            None,                    # b_bias
            w13_scale,               # b_scales [E, K/32, N_R] bf16
            None,                    # a_scales (BF16 input, no quant)
            None,                    # global_scale
            None,                    # b_qzeros (sym, none)
            None,                    # g_idx (no act-order)
            None,                    # perm
            ws,                      # workspace
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=BLOCK_SIZE_M,
            top_k=TOP_K,
            mul_topk_weights=False,
            b_q_type=QUANT_TYPE,
            size_m=T,
            size_n=N_R,
            size_k=H,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )
        apply_moe_activation(MoEActivation.SILU, final_out, inter_cache)

    # First call sanity-check
    run()
    torch.cuda.synchronize()
    if not torch.isfinite(final_out).all():
        raise RuntimeError(f"T={T}: non-finite values in output — Marlin broken?")

    # ── warmup ────────────────────────────────────────────────────────
    t0 = time.time()
    while time.time() - t0 < WARMUP_SEC:
        run()
    torch.cuda.synchronize()

    # ── timed iters with L2 flush ─────────────────────────────────────
    flush = torch.zeros(FLUSH_BYTES // 4, dtype=torch.int32, device=device)
    times_us = []
    for _ in range(NI):
        flush.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        run()
        e.record()
        torch.cuda.synchronize()
        times_us.append(s.elapsed_time(e) * 1000.0)
    times_us.sort()
    median_us = times_us[NI // 2]

    # FLOPS for the GEMM part only (per spec)
    flops = 2 * P * N_R * H
    tflops = flops / (median_us * 1e-6) / 1e12

    # Effective INT4 weight bandwidth
    # Distinct active experts at most P=T*TOP_K (could be fewer if routing
    # has collisions, but balanced random has 0 collision for T<=64).
    active_experts = min(P, E_LOC)
    w_bytes = active_experts * N_R * H * 0.5  # int4 = 0.5 bytes
    bw_tbs = w_bytes / (median_us * 1e-6) / 1e12

    return median_us, tflops, bw_tbs, active_experts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Run only the named config (T1 or T8)")
    args = parser.parse_args()

    print(f"=== Marlin baseline — Kimi K2.6/K2.7 MoE w13 INT4 + silu·mul fused ===", file=sys.stderr)
    print(f"GPU:        {torch.cuda.get_device_name(0)}", file=sys.stderr)
    print(f"Config:     TP=8 EP=1, H={H}, 2I_r={N_R}, I_r={I_R}, E_loc={E_LOC}, top_k={TOP_K}",
          file=sys.stderr)
    print(f"Methodology: {NI}-iter median, {FLUSH_BYTES//1024//1024}MB L2 flush, "
          f"{WARMUP_SEC:.1f}s warmup", file=sys.stderr)
    print(f"Note: SM100 has NO INT4 MMA; Marlin dequants to bf16 internally.",
          file=sys.stderr)
    print(file=sys.stderr)

    results = {}
    for name, T in CONFIGS:
        if args.config and args.config != name:
            continue
        us, tflops, bw, active = bench_one(T)
        results[name] = {"us": us, "tflops": tflops, "bw_tbs": bw,
                         "T": T, "active_experts": active}
        print(f"{name:>4}: {us:7.2f} us  {tflops:6.3f} TFLOPS  "
              f"weight_bw={bw:5.2f} TB/s  ({active} active experts)",
              file=sys.stderr)

    import json
    print(json.dumps({k: {"tflops": v["tflops"]} for k, v in results.items()}))


if __name__ == "__main__":
    main()
