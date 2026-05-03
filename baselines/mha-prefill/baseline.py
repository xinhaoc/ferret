"""CUTLASS SM100a MHA Prefill Baseline (B200)
Standard multi-head attention, causal, BF16.
Uses flashinfer single_prefill_with_kv_cache → CUTLASS SM100a tcgen05 kernel.

Usage:
    python3 baselines/mha-prefill/baseline.py
    python3 baselines/mha-prefill/baseline.py --num-heads 32 --num-kv-heads 8 --head-dim 128
"""
import argparse
import torch
from flashinfer.prefill import single_prefill_with_kv_cache

parser = argparse.ArgumentParser()
parser.add_argument("--num-heads", type=int, default=32)
parser.add_argument("--num-kv-heads", type=int, default=32)
parser.add_argument("--head-dim", type=int, default=128)
parser.add_argument("--seq-len", type=int, nargs="+", default=[1024, 2048, 4096])
parser.add_argument("--batch", type=int, default=1)
args = parser.parse_args()

B = args.batch
H_Q = args.num_heads
H_KV = args.num_kv_heads
D = args.head_dim
device, dtype = "cuda", torch.bfloat16
sm_scale = 1.0 / (D ** 0.5)

print(f"=== CUTLASS SM100a MHA Prefill Baseline ===")
print(f"H_Q={H_Q}, H_KV={H_KV}, D={D}, B={B}")
print()

for S in args.seq_len:
    q = torch.randn(S, H_Q, D, device=device, dtype=dtype)
    k = torch.randn(S, H_KV, D, device=device, dtype=dtype)
    v = torch.randn(S, H_KV, D, device=device, dtype=dtype)

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
    flops = 2 * B * H_Q * S * S * D * 2  # QK + PV
    tflops = flops / (ms / 1000) / 1e12
    print(f"S{S}: {tflops:.2f} TFLOPS, {us:.1f} us")
