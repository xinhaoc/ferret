"""DeepGEMM grouped FP8 GEMM baseline — DSv3 MoE DECODE shapes.

Calls deep_gemm.m_grouped_fp8_gemm_nt_contiguous with the REAL M_total
(no fake padding). DG's scheduler launches only ceil(M_total/BM) m-tiles,
so giving it real M_total is the production-equivalent measurement.

L2 flush between iterations: in real inference, attention + layernorm +
other ops evict L2 between MoE GEMM calls, so weights are NOT cached.
The 128 MB flush before each timed iteration mirrors that and gives the
production-relevant "L2 cold" measurement. Without flush, DG's per-call
time is artificially low (~3x faster) because the 939 MB weight matrix
partially stays L2-resident across consecutive calls.

Usage:
    python3 baselines/fp8-group-gemm/baseline_dsv3_decode.py
"""
import torch
import deep_gemm

device = "cuda"
NUM_GROUPS = 32  # DSv3 EP=8: 32 experts/rank
FLUSH_BYTES = 128 * 1024 * 1024  # 128 MB > 96 MB B200 L2

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
    M_total = M_per_e * E

    A_bf = torch.randn(M_total, K, device=device, dtype=torch.bfloat16)
    W_bf = torch.randn(E, N, K, device=device, dtype=torch.bfloat16)
    A_fp8, A_scale = deep_gemm.per_token_cast_to_fp8(A_bf, use_ue8m0=False)
    W_fp8 = torch.empty(E, N, K, device=device, dtype=torch.float8_e4m3fn)
    W_scale = torch.empty(E, N // 128, K // 128, device=device, dtype=torch.float32)
    for e in range(E):
        we_fp8, we_scale = deep_gemm.per_block_cast_to_fp8(W_bf[e], use_ue8m0=False)
        W_fp8[e] = we_fp8
        W_scale[e] = we_scale
    # m_indices[i] = expert id for row i. Rows are contiguous per expert.
    m_indices = torch.arange(M_total, device=device, dtype=torch.int32) // M_per_e
    out = torch.empty(M_total, N, device=device, dtype=torch.bfloat16)
    flush = torch.zeros(FLUSH_BYTES // 4, dtype=torch.int32, device=device)

    for _ in range(10):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (A_fp8, A_scale), (W_fp8, W_scale), out, m_indices, disable_ue8m0_cast=True,
        )
    torch.cuda.synchronize()

    NI = 100
    # Per-iteration timing with L2 flush between iters → "L2 cold" production-relevant
    iter_times_us = []
    for _ in range(NI):
        flush.zero_()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (A_fp8, A_scale), (W_fp8, W_scale), out, m_indices, disable_ue8m0_cast=True,
        )
        en.record()
        torch.cuda.synchronize()
        iter_times_us.append(st.elapsed_time(en) * 1000.0)
    iter_times_us.sort()
    ms = iter_times_us[len(iter_times_us) // 2] / 1000.0  # median
    us = ms * 1000
    flops = 2 * M_total * N * K
    tflops = flops / (ms / 1000) / 1e12
    print(f"{name}: {tflops:.2f} TFLOPS, {us:.1f} us  "
          f"(M_total={M_total} K={K} N={N} E={E})")
