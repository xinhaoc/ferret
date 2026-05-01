"""FA2 Unabsorbed MLA Chunked Prefill Baseline (B200)
Q covers a chunk [q_start, q_start+chunk_size), KV covers [0, kv_len).
Causal: position q attends to kv positions ≤ q.

Usage:
    python3 baselines/mla-prefill/baseline_chunked.py --num-heads 16
"""
import argparse
import torch
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--num-heads", type=int, default=16)
args = parser.parse_args()

B = 1
H = args.num_heads
UNABS_QK = 192
UNABS_V = 128
ABS_QK, ABS_V = 576, 512  # for absorbed-equivalent TFLOPS
device, dtype = "cuda", torch.bfloat16
sm_scale = 1.0 / (UNABS_QK ** 0.5)

configs = [
    {"name": "C512_KV2048",  "chunk": 512,  "kv_len": 2048},
    {"name": "C512_KV4096",  "chunk": 512,  "kv_len": 4096},
    {"name": "C1024_KV4096", "chunk": 1024, "kv_len": 4096},
    {"name": "C1024_KV8192", "chunk": 1024, "kv_len": 8192},
]

print(f"=== FA2 Chunked Prefill Baseline ===")
print(f"H={H}, D_QK={UNABS_QK}, D_V={UNABS_V}")
print()

workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)

for cfg in configs:
    chunk = cfg["chunk"]
    kv_len = cfg["kv_len"]
    q_start = kv_len - chunk  # last chunk of the sequence

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
