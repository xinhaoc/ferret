"""trtllm-gen MLA Prefill Baseline (B200) — Context kernel
DeepSeek V3 MLA: D_CKV=512, D_KPE=64, num_kv_heads=1 (shared KV)

Uses flashinfer.prefill.trtllm_batch_context_with_kv_cache which
dispatches to trtllm-gen's FmhaKernelType::Context (persistent scheduler).

Usage:
    python3 baselines/mla-prefill/baseline.py                                    # H=128, S=1024
    python3 baselines/mla-prefill/baseline.py --num-heads 64 --seq-len 1024      # TP=2
    python3 baselines/mla-prefill/baseline.py --num-heads 16 --seq-len 512 1024 2048  # TP=8
"""
import argparse
import torch
from flashinfer.prefill import trtllm_batch_context_with_kv_cache

parser = argparse.ArgumentParser()
parser.add_argument("--num-heads", type=int, default=128)
parser.add_argument("--seq-len", type=int, nargs="+", default=[512, 1024, 2048])
parser.add_argument("--page-size", type=int, default=64)
args = parser.parse_args()

B = 1
H = args.num_heads
NUM_KV_HEADS = 1   # MLA: single shared KV head
HEAD_DIM = 576      # D_CKV + D_KPE = 512 + 64
HEAD_DIM_V = 512    # D_CKV
PAGE_SIZE = args.page_size
device, dtype = "cuda", torch.bfloat16

sm_scale = 1.0 / (HEAD_DIM ** 0.5)

print(f"=== trtllm-gen MLA Prefill Baseline (Context kernel) ===")
print(f"H={H}, NUM_KV_HEADS={NUM_KV_HEADS}, HEAD_DIM={HEAD_DIM}, PAGE_SIZE={PAGE_SIZE}")
print()

for S in args.seq_len:
    num_pages = (S + PAGE_SIZE - 1) // PAGE_SIZE

    # Query: [num_tokens, num_heads, head_dim] (packed ragged format)
    num_tokens = B * S
    query = torch.randn(num_tokens, H, HEAD_DIM, device=device, dtype=dtype)

    # KV cache: [num_pages, 2, num_kv_heads, page_size, head_dim] (HND layout)
    # For MLA: num_kv_heads=1, head_dim=576 (D_CKV+D_KPE for K, D_CKV for V)
    # K and V share same compressed representation in MLA
    kv_cache = torch.randn(num_pages, 2, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM,
                           device=device, dtype=dtype)

    # Page table: [batch, max_num_pages_per_seq]
    block_tables = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)

    # Cumulative sequence lengths (ragged format)
    cum_seq_lens_q = torch.tensor([0, S], dtype=torch.int32, device=device)
    cum_seq_lens_kv = torch.tensor([0, S], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([S], dtype=torch.int32, device=device)

    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    kwargs = dict(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_q_len=S,
        max_kv_len=S,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        batch_size=B,
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv,
        kv_layout="HND",
    )

    # Warmup
    for _ in range(10):
        trtllm_batch_context_with_kv_cache(**kwargs)
    torch.cuda.synchronize()

    N = 50
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(N):
        trtllm_batch_context_with_kv_cache(**kwargs)
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / N
    # FLOPs: QK (2*B*H*S*S*D_K) + PV (2*B*H*S*S*D_V) ≈ 2*B*H*S^2*(D_K+D_V)
    flops = 2 * B * H * S * S * (HEAD_DIM + HEAD_DIM_V)
    tflops = flops / (ms / 1000) / 1e12
    print(f"S{S}: {tflops:.2f} TFLOPS, {ms * 1000:.1f} us")
