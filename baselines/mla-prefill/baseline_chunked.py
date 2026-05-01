"""MLA Chunked Prefill Baselines (B200)
Two FA2 implementations:
  1. BatchPrefillWithRaggedKVCacheWrapper (FA2 JIT — SGLang production path)
  2. single_prefill_with_kv_cache (CUTLASS SM100a — faster on B200)

Usage:
    python3 baselines/mla-prefill/baseline_chunked.py --num-heads 16
"""
import argparse
import torch
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper, single_prefill_with_kv_cache

parser = argparse.ArgumentParser()
parser.add_argument("--num-heads", type=int, default=16)
args = parser.parse_args()

B = 1
H = args.num_heads
UNABS_QK = 192
UNABS_V = 128
ABS_QK, ABS_V = 576, 512
device, dtype = "cuda", torch.bfloat16
sm_scale = 1.0 / (UNABS_QK ** 0.5)

configs = [
    {"name": "C256_KV2048",  "chunk": 256,  "kv_len": 2048},
    {"name": "C256_KV4096",  "chunk": 256,  "kv_len": 4096},
    {"name": "C512_KV4096",  "chunk": 512,  "kv_len": 4096},
    {"name": "C512_KV8192",  "chunk": 512,  "kv_len": 8192},
    {"name": "C1024_KV4096", "chunk": 1024, "kv_len": 4096},
    {"name": "C1024_KV8192", "chunk": 1024, "kv_len": 8192},
    {"name": "C2048_KV8192", "chunk": 2048, "kv_len": 8192},
]

print(f"=== MLA Chunked Prefill Baselines ===")
print(f"H={H}, D_QK={UNABS_QK}, D_V={UNABS_V}")

# ── 1. CUTLASS SM100a (stronger baseline) ──
print("\n--- CUTLASS SM100a single_prefill ---")
for cfg in configs:
    chunk, kv_len = cfg["chunk"], cfg["kv_len"]
    q = torch.randn(chunk, H, UNABS_QK, device=device, dtype=dtype)
    k = torch.randn(kv_len, 1, UNABS_QK, device=device, dtype=dtype)
    v = torch.randn(kv_len, 1, UNABS_V, device=device, dtype=dtype)

    for _ in range(10):
        single_prefill_with_kv_cache(q, k, v, causal=True, sm_scale=sm_scale, kv_layout="NHD")
    torch.cuda.synchronize()

    N = 50
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(N):
        single_prefill_with_kv_cache(q, k, v, causal=True, sm_scale=sm_scale, kv_layout="NHD")
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / N
    us = ms * 1000
    flops = 2 * B * H * chunk * kv_len * (ABS_QK + ABS_V)
    tflops = flops / (ms / 1000) / 1e12
    print(f"{cfg['name']}: {tflops:.2f} TFLOPS, {us:.1f} us")

# ── 2. FA2 batch (SGLang production) ──
print("\n--- FA2 batch (SGLang production) ---")
workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
for cfg in configs:
    chunk, kv_len = cfg["chunk"], cfg["kv_len"]
    q = torch.randn(B * chunk, H, UNABS_QK, device=device, dtype=dtype)
    k = torch.randn(B * kv_len, 1, UNABS_QK, device=device, dtype=dtype)
    v = torch.randn(B * kv_len, 1, UNABS_V, device=device, dtype=dtype)
    qo_indptr = torch.tensor([0, chunk], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)

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
