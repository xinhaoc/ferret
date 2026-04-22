"""trtllm-gen MLA Prefill Baseline (B200)
DeepSeek V3: D_CKV=512, D_KPE=64
Prefill: Q_LEN = SEQ_LEN (self-attention, causal)

Usage:
    python3 baselines/mla-prefill/baseline.py                                    # H=128, S=1024
    python3 baselines/mla-prefill/baseline.py --num-heads 64 --seq-len 1024      # TP=2
    python3 baselines/mla-prefill/baseline.py --num-heads 32 --seq-len 1024      # TP=4
    python3 baselines/mla-prefill/baseline.py --num-heads 16 --seq-len 1024      # TP=8
"""
import argparse
import torch
from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

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
PAGE_SIZE = args.page_size
device, dtype = "cuda", torch.bfloat16

sm_scale = 1.0 / (HEAD_DIM_QK ** 0.5)

print(f"=== trtllm-gen MLA Prefill Baseline (B200) ===")
print(f"H={H}, KV_LORA_RANK={KV_LORA_RANK}, QK_ROPE={QK_ROPE_HEAD_DIM}, PAGE_SIZE={PAGE_SIZE}")
print()

for S in args.seq_len:
    num_pages = (S + PAGE_SIZE - 1) // PAGE_SIZE
    kv_cache = torch.randn(num_pages, 1, PAGE_SIZE, KV_LORA_RANK + QK_ROPE_HEAD_DIM,
                            device=device, dtype=dtype)
    block_tables = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    # Prefill: q_len = S (full sequence, causal self-attention)
    query = torch.randn(B, S, H, HEAD_DIM_QK, device=device, dtype=dtype)
    seq_lens = torch.tensor([S], dtype=torch.int32, device=device)

    for _ in range(10):
        trtllm_batch_decode_with_kv_cache_mla(
            query, kv_cache, workspace,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=S,
            bmm1_scale=sm_scale,
            bmm2_scale=1.0,
            backend="trtllm-gen",
        )
    torch.cuda.synchronize()

    N = 50
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
            max_seq_len=S,
            bmm1_scale=sm_scale,
            bmm2_scale=1.0,
            backend="trtllm-gen",
        )
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / N
    flops = B * H * S * S * (KV_LORA_RANK + QK_ROPE_HEAD_DIM + KV_LORA_RANK)
    tflops = flops / (ms / 1000) / 1e12
    print(f"S{S}: {tflops:.2f} TFLOPS, {ms * 1000:.1f} us")
