"""CUTLASS SM100a MLA Chunked Prefill Baseline (B200)
Uses flashinfer single_prefill_with_kv_cache which dispatches to
CUTLASS SM100a tcgen05 kernel on B200.

Usage:
    python3 baselines/mla-prefill/baseline_chunked.py --num-heads 16
"""
import argparse
import torch
from flashinfer.prefill import single_prefill_with_kv_cache

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

print(f"=== CUTLASS SM100a Chunked Prefill Baseline ===")
print(f"H={H}, D_QK={UNABS_QK}, D_V={UNABS_V}")
print()

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
