"""FlashInfer FA2 MLA Multi-Token Decode Baseline (B200)
DeepSeek V3: D_CKV=512, D_KPE=64
Multi-token prediction: Q_LEN=1,2,3,4 with KV_LEN=4096

Usage:
    python3 baselines/mla-mtp-decode/baseline.py                  # 128 heads (default)
    python3 baselines/mla-mtp-decode/baseline.py --num-heads 64   # TP=2
    python3 baselines/mla-mtp-decode/baseline.py --num-heads 32   # TP=4
    python3 baselines/mla-mtp-decode/baseline.py --num-heads 16   # TP=8
"""
import argparse
import torch
import flashinfer

parser = argparse.ArgumentParser()
parser.add_argument("--num-heads", type=int, default=128)
parser.add_argument("--kv-len", type=int, default=4096)
args = parser.parse_args()

B, H, D_CKV, D_KPE, PAGE_SIZE = 1, args.num_heads, 512, 64, 128
KV_LEN = args.kv_len
device, dtype = "cuda", torch.bfloat16

num_pages = (KV_LEN + PAGE_SIZE - 1) // PAGE_SIZE
ckv_cache = torch.randn(num_pages, PAGE_SIZE, D_CKV, device=device, dtype=dtype)
kpe_cache = torch.randn(num_pages, PAGE_SIZE, D_KPE, device=device, dtype=dtype)

print(f"=== FlashInfer FA2 MLA Baseline (B200) ===")
print(f"H={H}, D_CKV={D_CKV}, D_KPE={D_KPE}, KV_LEN={KV_LEN}")
print()

for Q_LEN in [1, 2, 3, 4]:
    q_nope = torch.randn(Q_LEN, H, D_CKV, device=device, dtype=dtype)
    q_pe = torch.randn(Q_LEN, H, D_KPE, device=device, dtype=dtype)

    w = flashinfer.BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device), backend="fa2")
    qo_indptr = torch.tensor([0, Q_LEN], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    kv_len_arr = torch.tensor([KV_LEN], dtype=torch.int32, device=device)

    w.plan(qo_indptr=qo_indptr, kv_indptr=kv_indptr, kv_indices=kv_indices,
           kv_len_arr=kv_len_arr, num_heads=H, head_dim_ckv=D_CKV, head_dim_kpe=D_KPE,
           page_size=PAGE_SIZE, causal=(Q_LEN > 1), sm_scale=1.0 / (576 ** 0.5),
           q_data_type=dtype, kv_data_type=dtype)

    for _ in range(20):
        out = w.run(q_nope, q_pe, ckv_cache, kpe_cache)
    torch.cuda.synchronize()

    N = 100
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(N):
        out = w.run(q_nope, q_pe, ckv_cache, kpe_cache)
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / N
    flops = B * H * Q_LEN * KV_LEN * (D_CKV + D_KPE + D_CKV)
    tflops = flops / (ms / 1000) / 1e12
    print(f"Q{Q_LEN}: {tflops:.2f} TFLOPS, {ms * 1000:.1f} us")
