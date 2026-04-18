"""trtllm-gen MLA Multi-Token Decode Baseline (B200)
DeepSeek V3: D_CKV=512, D_KPE=64
Uses flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla with backend="trtllm-gen"

Usage:
    python3 baselines/mla-mtp-decode/baseline.py                  # 128 heads (default)
    python3 baselines/mla-mtp-decode/baseline.py --num-heads 64   # TP=2
    python3 baselines/mla-mtp-decode/baseline.py --num-heads 32   # TP=4
    python3 baselines/mla-mtp-decode/baseline.py --num-heads 16   # TP=8
"""
import argparse
import torch
from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

parser = argparse.ArgumentParser()
parser.add_argument("--num-heads", type=int, default=128)
parser.add_argument("--kv-len", type=int, default=4096)
parser.add_argument("--page-size", type=int, default=64)
args = parser.parse_args()

B = 1
H = args.num_heads
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128  # must be 128 or 64 per API
QK_ROPE_HEAD_DIM = 64
HEAD_DIM_QK = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
KV_LEN = args.kv_len
PAGE_SIZE = args.page_size
device, dtype = "cuda", torch.bfloat16

num_pages = (KV_LEN + PAGE_SIZE - 1) // PAGE_SIZE

# kv_cache: [num_pages, 1, page_size, head_dim_ckv + head_dim_kpe] (4D format)
kv_cache = torch.randn(num_pages, 1, PAGE_SIZE, KV_LORA_RANK + QK_ROPE_HEAD_DIM,
                        device=device, dtype=dtype)
# block_tables: [batch_size, max_num_pages_per_seq]
block_tables = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
# workspace for multi-block mode
workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

sm_scale = 1.0 / (HEAD_DIM_QK ** 0.5)

print(f"=== trtllm-gen MLA Decode Baseline (B200) ===")
print(f"H={H}, KV_LORA_RANK={KV_LORA_RANK}, QK_ROPE={QK_ROPE_HEAD_DIM}, KV_LEN={KV_LEN}, PAGE_SIZE={PAGE_SIZE}")
print()

for Q_LEN in [1, 2, 3, 4]:
    # query: [batch, q_len, num_heads, head_dim_qk]
    query = torch.randn(B, Q_LEN, H, HEAD_DIM_QK, device=device, dtype=dtype)
    seq_lens = torch.tensor([KV_LEN], dtype=torch.int32, device=device)

    # warmup
    for _ in range(20):
        trtllm_batch_decode_with_kv_cache_mla(
            query, kv_cache, workspace,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=KV_LEN,
            bmm1_scale=sm_scale,
            bmm2_scale=1.0,
            backend="trtllm-gen",
        )
    torch.cuda.synchronize()

    N = 100
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(N):
        trtllm_batch_decode_with_kv_cache_mla(
            query, kv_cache, workspace,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=KV_LEN,
            bmm1_scale=sm_scale,
            bmm2_scale=1.0,
            backend="trtllm-gen",
        )
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / N
    flops = B * H * Q_LEN * KV_LEN * (KV_LORA_RANK + QK_ROPE_HEAD_DIM + KV_LORA_RANK)
    tflops = flops / (ms / 1000) / 1e12
    print(f"Q{Q_LEN}: {tflops:.2f} TFLOPS, {ms * 1000:.1f} us")
