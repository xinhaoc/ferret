"""DeepGEMM Dense FP8 GEMM Baseline (B200)
DeepSeek V3 attention projections at TP=8.
Block-scaled: activation 1x128, weight 128x128.

Usage:
    python3 baselines/fp8-gemm/baseline_dense.py
"""
import torch
import deep_gemm

device = "cuda"

configs = [
    {"name": "q_b_proj_M1",   "M": 1,    "K": 1536, "N": 3072},
    {"name": "kv_b_proj_M1",  "M": 1,    "K": 512,  "N": 4096},
    {"name": "o_proj_M1",     "M": 1,    "K": 2048, "N": 7168},
    {"name": "q_b_proj_M16",  "M": 16,   "K": 1536, "N": 3072},
    {"name": "kv_b_proj_M16", "M": 16,   "K": 512,  "N": 4096},
    {"name": "o_proj_M16",    "M": 16,   "K": 2048, "N": 7168},
    {"name": "q_b_proj_M512", "M": 512,  "K": 1536, "N": 3072},
    {"name": "o_proj_M512",   "M": 512,  "K": 2048, "N": 7168},
]

print("=== DeepGEMM Dense FP8 GEMM Baseline (TP=8) ===\n")

for cfg in configs:
    M, K, N = cfg["M"], cfg["K"], cfg["N"]

    A_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    B_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)
    A_fp8, scale_a = deep_gemm.per_token_cast_to_fp8(A_bf16, use_ue8m0=False)
    B_fp8, scale_b = deep_gemm.per_block_cast_to_fp8(B_bf16, use_ue8m0=False)
    out = torch.empty(M, N, device=device, dtype=torch.bfloat16)

    for _ in range(10):
        deep_gemm.fp8_gemm_nt((A_fp8, scale_a), (B_fp8, scale_b), out)
    torch.cuda.synchronize()

    NI = 100
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(NI):
        deep_gemm.fp8_gemm_nt((A_fp8, scale_a), (B_fp8, scale_b), out)
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / NI
    us = ms * 1000
    flops = 2 * M * N * K
    tflops = flops / (ms / 1000) / 1e12
    print(f"{cfg['name']}: {tflops:.2f} TFLOPS, {us:.1f} us  (M={M} K={K} N={N})")
