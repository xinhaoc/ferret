"""FA2 Batch MLA Chunked Prefill Baseline (B200) — Batched
Uses BatchPrefillWithRaggedKVCacheWrapper (vLLM/SGLang production path).

Usage:
    python3 baselines/mla-prefill/baseline_chunked_batched.py --num-heads 16
"""
import argparse
import torch
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--num-heads", type=int, default=16)
args = parser.parse_args()

H = args.num_heads
UNABS_QK = 192
UNABS_V = 128
ABS_QK, ABS_V = 576, 512
device, dtype = "cuda", torch.bfloat16
sm_scale = 1.0 / (UNABS_QK ** 0.5)

configs = [
    {"name": "BS1_C512_KV4096",  "batch": 1,  "chunk": 512, "kv_len": 4096},
    {"name": "BS4_C512_KV4096",  "batch": 4,  "chunk": 512, "kv_len": 4096},
    {"name": "BS8_C512_KV4096",  "batch": 8,  "chunk": 512, "kv_len": 4096},
    {"name": "BS16_C512_KV4096", "batch": 16, "chunk": 512, "kv_len": 4096},
]

print(f"=== FA2 Batch Chunked Prefill Baseline ===")
print(f"H={H}, D_QK={UNABS_QK}, D_V={UNABS_V}")
print()

workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)

for cfg in configs:
    B, chunk, kv_len = cfg["batch"], cfg["chunk"], cfg["kv_len"]

    # Ragged format: B sequences, each with chunk Q tokens and kv_len KV tokens
    total_q = B * chunk
    total_kv = B * kv_len
    q = torch.randn(total_q, H, UNABS_QK, device=device, dtype=dtype)
    k = torch.randn(total_kv, 1, UNABS_QK, device=device, dtype=dtype)
    v = torch.randn(total_kv, 1, UNABS_V, device=device, dtype=dtype)

    qo_indptr = torch.arange(B + 1, dtype=torch.int32, device=device) * chunk
    kv_indptr = torch.arange(B + 1, dtype=torch.int32, device=device) * kv_len

    w = BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
    w.plan(qo_indptr, kv_indptr, H, 1, UNABS_QK, UNABS_V,
           q_data_type=dtype, kv_data_type=dtype,
           causal=True, sm_scale=sm_scale)

    for _ in range(10):
        w.run(q, k, v)
    torch.cuda.synchronize()

    N = 50
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(N):
        w.run(q, k, v)
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / N
    us = ms * 1000
    flops = 2 * B * H * chunk * kv_len * (ABS_QK + ABS_V)
    tflops = flops / (ms / 1000) / 1e12
    print(f"{cfg['name']}: {tflops:.2f} TFLOPS, {us:.1f} us")
