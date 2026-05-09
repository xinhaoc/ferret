"""DeepGEMM grouped FP8 GEMM baseline — DSv3 MoE DECODE shapes.

DeepGEMM contiguous pads M_per_expert to BLOCK_M=128 internally — at decode
shapes (M_per_expert=1..16) most compute is wasted on padding. Wall time is
~constant regardless of real M_per_expert ≤ 128. This is the bar to beat.

TFLOPS reported over REAL FLOPs (2 * M_real * N * K), NOT padded —
that's the production-relevant throughput.

Usage:
    python3 baselines/fp8-group-gemm/baseline_dsv3_decode.py
"""
import torch
import deep_gemm

device = "cuda"
NUM_GROUPS = 32  # DSv3 EP=8: 32 experts/rank
DEEPGEMM_BLOCK_M = 128

configs = [
    ("gate_up_M1",  1,  NUM_GROUPS, 7168, 4096),
    ("gate_up_M4",  4,  NUM_GROUPS, 7168, 4096),
    ("gate_up_M8",  8,  NUM_GROUPS, 7168, 4096),
    ("gate_up_M16", 16, NUM_GROUPS, 7168, 4096),
    ("down_M1",     1,  NUM_GROUPS, 2048, 7168),
    ("down_M4",     4,  NUM_GROUPS, 2048, 7168),
    ("down_M8",     8,  NUM_GROUPS, 2048, 7168),
    ("down_M16",    16, NUM_GROUPS, 2048, 7168),
]

print("=== DeepGEMM Grouped FP8 GEMM Baseline (DSv3 MoE DECODE) ===\n")

for name, M_per_e, E, K, N in configs:
    M_real_total = M_per_e * E
    # DeepGEMM pads each expert to BLOCK_M
    M_per_e_pad = ((M_per_e + DEEPGEMM_BLOCK_M - 1) // DEEPGEMM_BLOCK_M) * DEEPGEMM_BLOCK_M
    M_pad_total = M_per_e_pad * E

    A_bf = torch.randn(M_pad_total, K, device=device, dtype=torch.bfloat16)
    W_bf = torch.randn(E, N, K, device=device, dtype=torch.bfloat16)
    A_fp8, A_scale = deep_gemm.per_token_cast_to_fp8(A_bf, use_ue8m0=False)
    W_fp8 = torch.empty(E, N, K, device=device, dtype=torch.float8_e4m3fn)
    W_scale = torch.empty(E, N // 128, K // 128, device=device, dtype=torch.float32)
    for e in range(E):
        we_fp8, we_scale = deep_gemm.per_block_cast_to_fp8(W_bf[e], use_ue8m0=False)
        W_fp8[e] = we_fp8
        W_scale[e] = we_scale
    m_indices = torch.arange(E, device=device, dtype=torch.int32).repeat_interleave(M_per_e_pad)
    out = torch.empty(M_pad_total, N, device=device, dtype=torch.bfloat16)

    for _ in range(10):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (A_fp8, A_scale), (W_fp8, W_scale), out, m_indices, disable_ue8m0_cast=True,
        )
    torch.cuda.synchronize()

    NI = 100
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(NI):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (A_fp8, A_scale), (W_fp8, W_scale), out, m_indices, disable_ue8m0_cast=True,
        )
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / NI
    us = ms * 1000
    flops_real = 2 * M_real_total * N * K  # production-relevant
    tflops_real = flops_real / (ms / 1000) / 1e12
    print(f"{name}: {tflops_real:.2f} TFLOPS, {us:.1f} us  "
          f"(M_real={M_per_e} M_pad={M_per_e_pad} K={K} N={N} E={E})")
