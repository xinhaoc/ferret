"""FlashAttention-4 SM100 CuTe DSL Baseline (B200)
Standard MHA, causal, BF16. Uses FA4's native Blackwell kernel.

IMPORTANT: must use flash_attn.cute.interface (SM100 CuTe DSL),
NOT flash_attn.flash_attn_interface (SM80/SM90 C++ fallback).
Also requires CUTLASS_HOME pointing to CUTLASS installation.

Usage:
    python3 baselines/mha-prefill/baseline.py
"""
import argparse
import os
import torch

# Must set before importing FA4 cute
os.environ["CUTLASS_HOME"] = os.path.join(os.path.dirname(__file__), "../../resources/cutlass-4.4.2")

from flash_attn.cute.interface import flash_attn_func

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

print(f"=== FlashAttention-4 SM100 CuTe DSL Baseline ===")
print(f"B={B}, H_Q={H_Q}, H_KV={H_KV}, D={D}")
print()

for S in args.seq_len:
    q = torch.randn(B, S, H_Q, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H_KV, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H_KV, D, device=device, dtype=dtype)

    # JIT warmup (first call compiles)
    for _ in range(20):
        flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()

    N = 100
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(N):
        flash_attn_func(q, k, v, causal=True)
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / N
    us = ms * 1000
    flops = 4 * B * H_Q * S * S * D
    tflops = flops / (ms / 1000) / 1e12
    print(f"S{S}: {tflops:.2f} TFLOPS, {us:.1f} us")
