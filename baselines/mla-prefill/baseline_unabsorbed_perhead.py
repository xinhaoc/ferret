"""FA2 Unabsorbed MLA Prefill Baseline — PER-HEAD KV (correct DeepSeek V3)
After kv_b_proj decompression: K[S, H, 192], V[S, H, 128] — H KV heads.
Standard MHA attention, NOT MQA.

Usage:
    python3 baselines/mla-prefill/baseline_unabsorbed_perhead.py --num-heads 16
"""
import argparse
import torch
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--num-heads", type=int, default=16)
args = parser.parse_args()

H = args.num_heads
H_KV = H  # per-head KV, NOT 1 shared head
D_QK = 192  # qk_nope(128) + qk_rope(64)
D_V = 128
device, dtype = "cuda", torch.bfloat16
sm_scale = 1.0 / (D_QK ** 0.5)

configs = [
    {"name": "C512_KV2048",  "chunk": 512,  "kv_len": 2048},
    {"name": "C512_KV4096",  "chunk": 512,  "kv_len": 4096},
    {"name": "C1024_KV4096", "chunk": 1024, "kv_len": 4096},
    {"name": "C1024_KV8192", "chunk": 1024, "kv_len": 8192},
]

print(f"=== FA2 Unabsorbed MLA Prefill (per-head KV) ===")
print(f"H_Q={H}, H_KV={H_KV}, D_QK={D_QK}, D_V={D_V}")
print()

workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)

for cfg in configs:
    chunk, kv_len = cfg["chunk"], cfg["kv_len"]
    q = torch.randn(chunk, H, D_QK, device=device, dtype=dtype)
    k = torch.randn(kv_len, H_KV, D_QK, device=device, dtype=dtype)
    v = torch.randn(kv_len, H_KV, D_V, device=device, dtype=dtype)

    qo_indptr = torch.tensor([0, chunk], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)

    w = BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
    w.plan(qo_indptr, kv_indptr, H, H_KV, D_QK, D_V,
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
    flops = 4 * chunk * kv_len * H * D_QK  # 2x QK + 2x PV (using D_QK as proxy)
    tflops = flops / (ms / 1000) / 1e12
    print(f"{cfg['name']}: {tflops:.2f} TFLOPS, {us:.1f} us")
