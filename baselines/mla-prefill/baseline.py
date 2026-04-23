"""FA2 Unabsorbed MLA Prefill Baseline (B200)
Production path: BatchPrefillWithRaggedKVCacheWrapper, D_QK=192, D_V=128

Usage:
    python3 baselines/mla-prefill/baseline.py --num-heads 16 --seq-len 512 1024 2048
"""
import argparse
import torch
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--num-heads", type=int, default=128)
parser.add_argument("--seq-len", type=int, nargs="+", default=[512, 1024, 2048])
args = parser.parse_args()

B = 1
H = args.num_heads
UNABS_QK = 192   # qk_nope_head_dim(128) + qk_rope_head_dim(64)
UNABS_V = 128
device, dtype = "cuda", torch.bfloat16
sm_scale = 1.0 / (UNABS_QK ** 0.5)

print(f"=== FA2 Unabsorbed MLA Prefill Baseline ===")
print(f"H={H}, D_QK={UNABS_QK}, D_V={UNABS_V}")
print()

workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)

for S in args.seq_len:
    q = torch.randn(B * S, H, UNABS_QK, device=device, dtype=dtype)
    k = torch.randn(B * S, 1, UNABS_QK, device=device, dtype=dtype)
    v = torch.randn(B * S, 1, UNABS_V, device=device, dtype=dtype)
    qo_indptr = torch.tensor([0, S], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, S], dtype=torch.int32, device=device)

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
    # Must match kernel's formula: 2.0 * B * H * S * S * (D_CKV + D_KPE + D_V)
    # = 2 * B * H * S * S * (512 + 64 + 512) = 2 * B * H * S * S * 1088
    ABS_QK, ABS_V = 576, 512
    flops = 2 * B * H * S * S * (ABS_QK + ABS_V)
    tflops = flops / (ms / 1000) / 1e12
    print(f"S{S}: {tflops:.2f} TFLOPS (absorbed-equivalent), {us:.1f} us latency")
