"""MLA Prefill Baselines (B200)
Two references:
  1. trtllm-gen absorbed (D_QK=576, D_V=512) — same algorithm as agent's kernel
  2. FA2 unabsorbed (D_QK=192, D_V=128) — what vLLM/SGLang deploy for prefill

Usage:
    python3 baselines/mla-prefill/baseline.py --num-heads 16 --seq-len 512 1024 2048
"""
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--num-heads", type=int, default=128)
parser.add_argument("--seq-len", type=int, nargs="+", default=[512, 1024, 2048])
parser.add_argument("--page-size", type=int, default=64)
args = parser.parse_args()

B = 1
H = args.num_heads
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
HEAD_DIM_QK = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
HEAD_DIM_V = KV_LORA_RANK                        # 512
# Unabsorbed dims
UNABS_QK = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM   # 192
UNABS_V = 128
PAGE_SIZE = args.page_size
device, dtype = "cuda", torch.bfloat16

sm_scale_abs = 1.0 / (HEAD_DIM_QK ** 0.5)
sm_scale_unabs = 1.0 / (UNABS_QK ** 0.5)

print(f"=== MLA Prefill Baselines (B200) ===")
print(f"H={H}, PAGE_SIZE={PAGE_SIZE}")
print(f"Absorbed: D_QK={HEAD_DIM_QK}, D_V={HEAD_DIM_V}")
print(f"Unabsorbed: D_QK={UNABS_QK}, D_V={UNABS_V}")
print()

# ── 1. trtllm-gen absorbed ──
from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

print("--- trtllm-gen absorbed ---")
for S in args.seq_len:
    num_pages = (S + PAGE_SIZE - 1) // PAGE_SIZE
    kv_cache = torch.randn(num_pages, 1, PAGE_SIZE, KV_LORA_RANK + QK_ROPE_HEAD_DIM,
                            device=device, dtype=dtype)
    block_tables = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    query = torch.randn(B, S, H, HEAD_DIM_QK, device=device, dtype=dtype)
    seq_lens = torch.tensor([S], dtype=torch.int32, device=device)

    for _ in range(10):
        trtllm_batch_decode_with_kv_cache_mla(
            query, kv_cache, workspace,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM, kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM, block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=S,
            bmm1_scale=sm_scale_abs, bmm2_scale=1.0, backend="trtllm-gen")
    torch.cuda.synchronize()

    N = 50
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(N):
        trtllm_batch_decode_with_kv_cache_mla(
            query, kv_cache, workspace,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM, kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM, block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=S,
            bmm1_scale=sm_scale_abs, bmm2_scale=1.0, backend="trtllm-gen")
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / N
    flops = B * H * S * S * (HEAD_DIM_QK + HEAD_DIM_V)
    tflops = flops / (ms / 1000) / 1e12
    print(f"S{S}: {tflops:.2f} TFLOPS, {ms*1000:.1f} us")

# ── 2. FA2 unabsorbed ──
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

print()
print("--- FA2 unabsorbed ---")
workspace2 = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)

for S in args.seq_len:
    q = torch.randn(B * S, H, UNABS_QK, device=device, dtype=dtype)
    k = torch.randn(B * S, 1, UNABS_QK, device=device, dtype=dtype)
    v = torch.randn(B * S, 1, UNABS_V, device=device, dtype=dtype)
    qo_indptr = torch.tensor([0, S], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, S], dtype=torch.int32, device=device)

    w = BatchPrefillWithRaggedKVCacheWrapper(workspace2, "NHD")
    w.plan(qo_indptr, kv_indptr, H, 1, UNABS_QK, UNABS_V,
           q_data_type=dtype, kv_data_type=dtype,
           causal=True, sm_scale=sm_scale_unabs)

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
    flops = B * H * S * S * (UNABS_QK + UNABS_V)
    tflops = flops / (ms / 1000) / 1e12
    print(f"S{S}: {tflops:.2f} TFLOPS, {ms*1000:.1f} us")
