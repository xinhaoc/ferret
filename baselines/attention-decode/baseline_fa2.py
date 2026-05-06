"""FlashInfer fa2 baseline at mirage's TP=8 per-rank shape.

GQA multi-token decode: 4 qo_heads, 1 kv_head, head_dim=128, BF16, ragged KV.
This is what FlashInfer falls back to on B200 with backend='auto' (because
fa3 was compiled SM90-only and fails on Blackwell).

Usage:
    python3 baselines/attention-decode/baseline_fa2.py
"""
import math
import torch
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

device = "cuda"
dtype = torch.bfloat16
qo_heads, kv_heads, head_dim = 4, 1, 128

configs = [
    ("mt1_kv128",  1,  128),
    ("mt1_kv512",  1,  512),
    ("mt4_kv128",  4,  128),
    ("mt4_kv512",  4,  512),
    ("mt8_kv128",  8,  128),
    ("mt8_kv512",  8,  512),
]

print("=== FlashInfer fa2 GQA decode (qo=4 kv=1 head_dim=128) ===\n")

ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)

for name, mt, kv_len in configs:
    q = torch.randn(mt,     qo_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(kv_len, kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(kv_len, kv_heads, head_dim, device=device, dtype=dtype)

    w = BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
    w.plan(
        qo_indptr=torch.tensor([0, mt],    device=device, dtype=torch.int32),
        kv_indptr=torch.tensor([0, kv_len], device=device, dtype=torch.int32),
        num_qo_heads=qo_heads, num_kv_heads=kv_heads,
        head_dim_qk=head_dim, head_dim_vo=head_dim,
        q_data_type=dtype, kv_data_type=dtype, causal=True,
    )

    for _ in range(20):
        w.run(q, k, v)
    torch.cuda.synchronize()

    NI = 500
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(NI):
        w.run(q, k, v)
    en.record()
    torch.cuda.synchronize()
    us = st.elapsed_time(en) * 1000.0 / NI
    flops = 4 * mt * kv_len * qo_heads * head_dim  # 2x QK + 2x PV
    tflops = flops / (us * 1e-6) / 1e12
    print(f"{name}: {tflops:6.2f} TFLOPS, {us:7.3f} us  (mt={mt} kv_len={kv_len})")
