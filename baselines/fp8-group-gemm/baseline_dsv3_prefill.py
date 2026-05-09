"""DeepGEMM grouped FP8 GEMM baseline — DeepSeek V3 MoE prefill (contiguous layout).

This is what vLLM ships in deep_gemm_moe.py for DSv3 MoE. Calls
deep_gemm.m_grouped_fp8_gemm_nt_contiguous: same kernel as the dense
FP8 GEMM but with kGemmType::GroupedContiguous scheduler.

Per-rank shapes (EP=8, 32 experts/rank):
  gate_up: K=7168, N=4096  (gate ∥ up fused)
  down:    K=2048, N=7168

M-per-expert ∈ {64, 128, 256, 512} (prefill chunk T = 32 * M_per_expert tokens
on this rank, after topk=8 routing).

Usage:
    python3 baselines/fp8-group-gemm/baseline_dsv3_prefill.py
"""
import torch
import deep_gemm

device = "cuda"
NUM_GROUPS = 32  # experts per rank for DSv3 EP=8

configs = [
    ("gate_up_M64",  64,  NUM_GROUPS, 7168, 4096),
    ("gate_up_M128", 128, NUM_GROUPS, 7168, 4096),
    ("gate_up_M256", 256, NUM_GROUPS, 7168, 4096),
    ("gate_up_M512", 512, NUM_GROUPS, 7168, 4096),
    ("down_M64",     64,  NUM_GROUPS, 2048, 7168),
    ("down_M128",    128, NUM_GROUPS, 2048, 7168),
    ("down_M256",    256, NUM_GROUPS, 2048, 7168),
    ("down_M512",    512, NUM_GROUPS, 2048, 7168),
]

print("=== DeepGEMM Grouped FP8 GEMM Baseline (DSv3 MoE prefill, EP=8) ===\n")

for name, M_per_e, E, K, N in configs:
    M_total = M_per_e * E

    A_bf = torch.randn(M_total, K, device=device, dtype=torch.bfloat16)
    W_bf = torch.randn(E, N, K, device=device, dtype=torch.bfloat16)

    A_fp8, A_scale = deep_gemm.per_token_cast_to_fp8(A_bf, use_ue8m0=False)

    # Per-expert weight cast (DeepGEMM helper handles 3D)
    W_fp8 = torch.empty(E, N, K, device=device, dtype=torch.float8_e4m3fn)
    W_scale = torch.empty(E, N // 128, K // 128, device=device, dtype=torch.float32)
    for e in range(E):
        we_fp8, we_scale = deep_gemm.per_block_cast_to_fp8(W_bf[e], use_ue8m0=False)
        W_fp8[e] = we_fp8
        W_scale[e] = we_scale

    # m_indices[M_total]: expert id per row (rows are pre-permuted by expert)
    m_indices = torch.arange(E, device=device, dtype=torch.int32).repeat_interleave(M_per_e)

    out = torch.empty(M_total, N, device=device, dtype=torch.bfloat16)

    for _ in range(10):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (A_fp8, A_scale), (W_fp8, W_scale), out, m_indices,
        )
    torch.cuda.synchronize()

    NI = 100
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(NI):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (A_fp8, A_scale), (W_fp8, W_scale), out, m_indices,
        )
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / NI
    us = ms * 1000
    flops = 2 * M_total * N * K
    tflops = flops / (ms / 1000) / 1e12
    print(f"{name}: {tflops:.2f} TFLOPS, {us:.1f} us  "
          f"(M_total={M_total} K={K} N={N} E={E})")
