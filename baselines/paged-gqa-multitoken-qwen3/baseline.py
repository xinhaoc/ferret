"""FlashInfer paged GQA multi-token (Q>1) decode baseline — Qwen3-30B-A3B.

Q_LEN in {2, 3, 4} (speculative decode / MTP regime), causal-among-Q.
NUM_QO_HEADS=8, NUM_KV_HEADS=1 (GQA=8), HEAD_DIM=128, bf16.

Causal mask among query tokens: query token i attends to positions
[0, seq_len - Q_LEN + 1 + i). Earlier queries see fewer positions.

For Q>1 we use BatchPrefillWithPagedKVCacheWrapper (decode wrapper assumes
Q=1). This is the multi-token decode aka speculative-decode-style path.

FLOPs (per batch call, sum over query tokens):
    sum_{i=0..Q_LEN-1} 4 * NUM_QO_HEADS * (seq_len - Q_LEN + 1 + i) * HEAD_DIM
  = 4 * NUM_QO_HEADS * HEAD_DIM * sum_{i=0..Q-1} (seq_len - Q + 1 + i)
  = 4 * NUM_QO_HEADS * HEAD_DIM * Q * (seq_len - (Q-1)/2)

L2 flush 128 MB before each timed iteration, 300 iter median, 2-sec
continuous warmup to ramp clocks to steady state.

Usage:
    python3 baselines/paged-gqa-multitoken-qwen3/baseline.py
"""
import argparse
import time

import torch
torch.manual_seed(0)
import flashinfer

device = "cuda"
DTYPE = torch.bfloat16
NUM_QO_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 128
PAGE_SIZE = 4096
FLUSH_BYTES = 128 * 1024 * 1024
NI = 300
WARMUP_SEC = 2.0

# Q in {2, 3, 4} crossed with seq_len in {128, 512, 4096, 32768}.
# Tests the multi-token kernel across both the overhead-dominated regime
# (small seq) and the bandwidth-bound regime (long seq).
CONFIGS = [
    # (name,        Q_LEN, SEQ_LEN)
    ("q2_seq128",   2,     128),
    ("q2_seq512",   2,     512),
    ("q2_seq4k",    2,     4096),
    ("q2_seq32k",   2,     32768),
    ("q3_seq128",   3,     128),
    ("q3_seq512",   3,     512),
    ("q3_seq4k",    3,     4096),
    ("q3_seq32k",   3,     32768),
    ("q4_seq128",   4,     128),
    ("q4_seq512",   4,     512),
    ("q4_seq4k",    4,     4096),
    ("q4_seq32k",   4,     32768),
]


def bench(q_len, seq_len, ni=NI):
    b = 1
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    last_page_len = (seq_len - 1) % PAGE_SIZE + 1 if seq_len > 0 else 0

    # Paged KV cache (same layout as decode case): NHD =
    # (num_pages, 2, page_size, num_kv_heads, head_dim)
    kv_cache = torch.randn(
        num_pages, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
        dtype=DTYPE, device=device,
    )
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)
    qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device=device)

    q = torch.randn(q_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE,
        causal=True,
        q_data_type=DTYPE, kv_data_type=DTYPE,
    )

    # Warmup continuously for WARMUP_SEC to ramp GPU clocks to steady state
    t0 = time.time()
    while time.time() - t0 < WARMUP_SEC:
        out = wrapper.run(q, kv_cache)
    torch.cuda.synchronize()

    flush = torch.zeros(FLUSH_BYTES // 4, dtype=torch.int32, device=device)
    times_us = []
    for _ in range(ni):
        flush.zero_()
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        se.record()
        wrapper.run(q, kv_cache)
        ee.record()
        torch.cuda.synchronize()
        times_us.append(se.elapsed_time(ee) * 1000.0)
    times_us.sort()
    median_us = times_us[len(times_us) // 2]

    # Causal among Q tokens: query i attends to (seq_len - q_len + 1 + i) positions.
    # Total attended = sum over i of (seq_len - q_len + 1 + i)
    #                = q_len * seq_len - q_len*(q_len - 1)/2
    total_attended = q_len * seq_len - q_len * (q_len - 1) // 2
    # Per-token FLOPs: 4 * N_h * attended * HEAD_DIM (QK^T + softmax + PV)
    flops = 4 * NUM_QO_HEADS * total_attended * HEAD_DIM
    mem_bytes = 2 * seq_len * NUM_KV_HEADS * HEAD_DIM * 2  # K+V read
    tflops = flops / (median_us * 1e-6) / 1e12
    gbps = mem_bytes / (median_us * 1e-6) / 1e9
    return tflops, median_us, flops, mem_bytes, gbps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Run only the named config (e.g. q2)")
    args = parser.parse_args()

    print("=== FlashInfer Paged GQA Multi-Token (Q=2/3/4) Baseline (Qwen3-30B-A3B) ===")
    print(f"Device:        {torch.cuda.get_device_name(0)}")
    print(f"FlashInfer:    {flashinfer.__version__}")
    print(f"NUM_QO_HEADS={NUM_QO_HEADS}  NUM_KV_HEADS={NUM_KV_HEADS}  HEAD_DIM={HEAD_DIM}  "
          f"PAGE_SIZE={PAGE_SIZE}  dtype=bf16  causal=True")
    print(f"Iterations:    {NI} timed median, L2 flush per iter, "
          f"{FLUSH_BYTES // (1024 * 1024)} MB, {WARMUP_SEC:.0f}-sec warmup")
    print()

    for name, q_len, seq_len in CONFIGS:
        if args.config and args.config != name:
            continue
        tflops, us, flops, mem, gbps = bench(q_len, seq_len)
        print(f"{name:>6}: {tflops:>7.2f} TFLOPS  {us:>7.1f} us  "
              f"{gbps:>7.1f} GB/s  (Q={q_len} seq={seq_len} "
              f"flops={flops/1e9:.3f}G mem={mem/1e6:.2f}MB)")


if __name__ == "__main__":
    main()
