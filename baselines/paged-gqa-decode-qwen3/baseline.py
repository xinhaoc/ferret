"""FlashInfer paged GQA decode baseline — Qwen3-30B-A3B shapes.

Q_len=1 decode, varying seq_len. GQA=8, head_dim=128, bf16.

Configuration: NUM_QO_HEADS=8, NUM_KV_HEADS=1 (the TP=4 per-rank shape for
Qwen3-30B-A3B's 32-head / 4-kv-head config). HEAD_DIM=128, PAGE_SIZE=4096.

Calls flashinfer.BatchDecodeWithPagedKVCacheWrapper with one request per
launch. Reports TFLOPS using the formula:
    flops = 2 * NUM_QO_HEADS * SEQ_LEN * 2 * HEAD_DIM
          = QK^T term + softmax · V term, per query token
          = 4 * NUM_QO_HEADS * SEQ_LEN * HEAD_DIM

L2 flush 128 MB before each timed iteration, 100 iter median.

Usage:
    python3 baselines/paged-gqa-decode-qwen3/baseline.py
"""
import argparse

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
NI = 300              # was 100 — more samples for tighter median
WARMUP_SEC = 2.0      # ramp clocks to steady state then measure-while-hot

# Q=1 decode shapes across seq_len axis
SEQ_LENS = [128, 512, 4096, 32768]   # seq128k removed per user instruction


def bench(seq_len, ni=NI):
    b = 1
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    last_page_len = (seq_len - 1) % PAGE_SIZE + 1 if seq_len > 0 else 0

    # KV cache layout NHD: (num_pages, 2, page_size, num_kv_heads, head_dim)
    kv_cache = torch.randn(
        num_pages, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
        dtype=DTYPE, device=device,
    )
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)

    q = torch.randn(b, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    # use_tensor_cores=True is required on B200 — the CUDA-core path is
    # ~4x slower (verified seq=131072: 247us vs 58us)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_tensor_cores=True,
    )
    wrapper.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE,
        q_data_type=DTYPE, kv_data_type=DTYPE,
    )

    # Warmup — run continuously for WARMUP_SEC to ramp GPU clocks to steady
    # state. Without this, the first iterations measure rising clocks. With
    # cooldown, the clocks decay before measurement. So: warm-and-measure-hot.
    import time
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

    # Real FLOPs (per query): QK^T = N_h × S × 2D + softmax·V = N_h × 2D × S
    # Total = 4 × N_h × S × D
    flops = 4 * NUM_QO_HEADS * seq_len * HEAD_DIM
    # Memory volume: KV (FP16/BF16) bytes read = 2 × S × N_kv × D × 2 bytes (K+V)
    mem_bytes = 2 * seq_len * NUM_KV_HEADS * HEAD_DIM * 2
    tflops = flops / (median_us * 1e-6) / 1e12
    gbps = mem_bytes / (median_us * 1e-6) / 1e9
    return tflops, median_us, flops, mem_bytes, gbps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Run only the named config (e.g. seq4k)")
    args = parser.parse_args()

    print("=== FlashInfer Paged GQA Decode Baseline (Qwen3-30B-A3B shape) ===")
    print(f"Device:        {torch.cuda.get_device_name(0)}")
    print(f"FlashInfer:    {flashinfer.__version__}")
    print(f"NUM_QO_HEADS={NUM_QO_HEADS}  NUM_KV_HEADS={NUM_KV_HEADS}  HEAD_DIM={HEAD_DIM}  "
          f"PAGE_SIZE={PAGE_SIZE}  dtype=bf16")
    print(f"Iterations:    {NI} timed (median), L2 flush per iter, "
          f"{FLUSH_BYTES // (1024 * 1024)} MB")
    print()

    name_map = {128: "seq128", 512: "seq512", 4096: "seq4k",
                32768: "seq32k", 131072: "seq128k"}
    for s in SEQ_LENS:
        name = name_map[s]
        if args.config and args.config != name:
            continue
        tflops, us, flops, mem, gbps = bench(s)
        print(f"{name:>8}: {tflops:>7.2f} TFLOPS  {us:>7.1f} us  "
              f"{gbps:>7.1f} GB/s  (Q=1 seq={s} "
              f"flops={flops/1e9:.3f}G mem={mem/1e6:.2f}MB)")


if __name__ == "__main__":
    main()
