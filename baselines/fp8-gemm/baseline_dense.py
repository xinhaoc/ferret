"""DeepGEMM Dense FP8 GEMM Baseline (B200)
DeepSeek V3 attention projections at TP=8.
Block-scaled: activation 1x128, weight 128x128.

Usage:
    python3 baselines/fp8-gemm/baseline_dense.py
"""
import torch
import deep_gemm

device = "cuda"

# DeepSeek V3 attention projections at TP=8
# M = batch tokens (decode=1-16, prefill=512-4096)
configs = [
    # Decode (M=1-16)
    {"name": "q_a_proj_M1",   "M": 1,    "K": 7168, "N": 1536,  "desc": "q_a_proj (replicated)"},
    {"name": "q_b_proj_M1",   "M": 1,    "K": 1536, "N": 3072,  "desc": "q_b_proj (TP=8, 16 heads)"},
    {"name": "kv_a_proj_M1",  "M": 1,    "K": 7168, "N": 576,   "desc": "kv_a_proj (replicated)"},
    {"name": "kv_b_proj_M1",  "M": 1,    "K": 512,  "N": 4096,  "desc": "kv_b_proj (TP=8, 16 heads)"},
    {"name": "o_proj_M1",     "M": 1,    "K": 2048, "N": 7168,  "desc": "o_proj (TP=8)"},
    # Decode batch=16
    {"name": "q_a_proj_M16",  "M": 16,   "K": 7168, "N": 1536,  "desc": "q_a_proj"},
    {"name": "q_b_proj_M16",  "M": 16,   "K": 1536, "N": 3072,  "desc": "q_b_proj"},
    {"name": "kv_a_proj_M16", "M": 16,   "K": 7168, "N": 576,   "desc": "kv_a_proj"},
    {"name": "kv_b_proj_M16", "M": 16,   "K": 512,  "N": 4096,  "desc": "kv_b_proj"},
    {"name": "o_proj_M16",    "M": 16,   "K": 2048, "N": 7168,  "desc": "o_proj"},
    # Prefill chunk=512
    {"name": "q_a_proj_M512", "M": 512,  "K": 7168, "N": 1536,  "desc": "q_a_proj"},
    {"name": "q_b_proj_M512", "M": 512,  "K": 1536, "N": 3072,  "desc": "q_b_proj"},
    {"name": "kv_a_proj_M512","M": 512,  "K": 7168, "N": 576,   "desc": "kv_a_proj"},
    {"name": "kv_b_proj_M512","M": 512,  "K": 512,  "N": 4096,  "desc": "kv_b_proj"},
    {"name": "o_proj_M512",   "M": 512,  "K": 2048, "N": 7168,  "desc": "o_proj"},
]

print("=== DeepGEMM Dense FP8 GEMM Baseline (TP=8) ===")
print()

for cfg in configs:
    M, K, N = cfg["M"], cfg["K"], cfg["N"]

    # FP8 e4m3 inputs
    A = torch.randn(M, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)

    # Block scales: activation 1x128, weight 128x128
    scale_a = torch.ones(M, (K + 127) // 128, device=device, dtype=torch.float32)
    scale_b = torch.ones((N + 127) // 128, (K + 127) // 128, device=device, dtype=torch.float32)

    out = torch.empty(M, N, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(10):
        deep_gemm.fp8_gemm_nt(A, scale_a, B, scale_b, out)
    torch.cuda.synchronize()

    # Benchmark
    NI = 100
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(NI):
        deep_gemm.fp8_gemm_nt(A, scale_a, B, scale_b, out)
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / NI
    us = ms * 1000
    flops = 2 * M * N * K
    tflops = flops / (ms / 1000) / 1e12
    print(f"{cfg['name']}: {tflops:.2f} TFLOPS, {us:.1f} us  ({cfg['desc']})")
