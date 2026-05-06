"""cuBLAS FP8 GEMM Baseline (B200) via torch._scaled_mm
DeepSeek V3 attention projections at TP=8.
This is what vLLM uses (cutlass_scaled_mm → cuBLAS FP8).

Usage:
    python3 baselines/fp8-gemm/baseline_dense.py
"""
import torch

device = "cuda"

configs = [
    ("q_b_proj_M1",   1,    1536, 3072),
    ("kv_b_proj_M1",  1,    512,  4096),
    ("o_proj_M1",     1,    2048, 7168),
    ("q_b_proj_M16",  16,   1536, 3072),
    ("kv_b_proj_M16", 16,   512,  4096),
    ("o_proj_M16",    16,   2048, 7168),
    ("q_b_proj_M512", 512,  1536, 3072),
    ("o_proj_M512",   512,  2048, 7168),
]

print("=== cuBLAS FP8 GEMM Baseline (TP=8) ===\n")

for name, M, K, N in configs:
    A = torch.randn(M, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scale_a = torch.ones(M, 1, device=device, dtype=torch.float32)
    scale_b = torch.ones(1, N, device=device, dtype=torch.float32)

    for _ in range(10):
        torch._scaled_mm(A, B.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()

    NI = 100
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(NI):
        torch._scaled_mm(A, B.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / NI
    us = ms * 1000
    tflops = 2 * M * N * K / (ms / 1000) / 1e12
    print(f"{name}: {tflops:.2f} TFLOPS, {us:.1f} us  (M={M} K={K} N={N})")
