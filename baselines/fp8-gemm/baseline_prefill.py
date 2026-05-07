"""DeepGEMM FP8 GEMM Prefill Baseline (B200) — what vLLM and SGLang ship.

DeepGEMM is the open-source FP8 block-scaled GEMM that vLLM/SGLang use by
default for DeepSeek V3 attention projections on Blackwell. Selected via
`should_use_deepgemm_for_fp8_linear` (vllm) / `flashinfer.deep_gemm` (sglang).
At M >= 4096 it matches or beats cuBLAS on these shapes.

Block-scaled FP8: activation scale per 1×128, weight scale per 128×128.

Usage:
    python3 baselines/fp8-gemm/baseline_prefill.py
"""
import torch  # MUST be first to load libc10
import deep_gemm

device = "cuda"

configs = [
    ("q_b_proj_M512",   512,  1536, 3072),
    ("kv_b_proj_M512",  512,  512,  4096),
    ("o_proj_M512",     512,  2048, 7168),
    ("q_b_proj_M1024",  1024, 1536, 3072),
    ("kv_b_proj_M1024", 1024, 512,  4096),
    ("o_proj_M1024",    1024, 2048, 7168),
    ("q_b_proj_M2048",  2048, 1536, 3072),
    ("kv_b_proj_M2048", 2048, 512,  4096),
    ("o_proj_M2048",    2048, 2048, 7168),
    ("q_b_proj_M4096",  4096, 1536, 3072),
    ("kv_b_proj_M4096", 4096, 512,  4096),
    ("o_proj_M4096",    4096, 2048, 7168),
    ("q_b_proj_M8192",  8192, 1536, 3072),
    ("kv_b_proj_M8192", 8192, 512,  4096),
    ("o_proj_M8192",    8192, 2048, 7168),
]

print("=== DeepGEMM FP8 GEMM Prefill Baseline (TP=8) ===\n")

for name, M, K, N in configs:
    A_bf = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    B_bf = torch.randn(N, K, device=device, dtype=torch.bfloat16)
    A_fp8, scale_a = deep_gemm.per_token_cast_to_fp8(A_bf, use_ue8m0=False)
    B_fp8, scale_b = deep_gemm.per_block_cast_to_fp8(B_bf, use_ue8m0=False)
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
    tflops = 2 * M * N * K / (ms / 1000) / 1e12
    print(f"{name}: {tflops:.2f} TFLOPS, {us:.1f} us  (M={M} K={K} N={N})")
