"""FlashInfer paged GQA baseline — UNIFIED Q=1/2/3/4 x seq=128/512/4k/32k.

For Q=1 uses BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True).
For Q>1 uses BatchPrefillWithPagedKVCacheWrapper(causal=True).
This matches what vLLM/SGLang actually dispatch.

16 configs total covering the fusion target — all of:
  examples/paged-gqa-decode-qwen3/v012_q1_seq4k_2x.cu      (Q=1 paged decode)
  examples/paged-gqa-multitoken-qwen3/v007_q234_seq4k_2x.cu (Q=2/3/4 with separate combine)
  workspace v009 (Q=2/3/4 with fused atomic combine)

Causal mask among query tokens (Q>1 only): query token i attends to positions
[0, seq_len - Q_LEN + 1 + i). Earlier queries see fewer positions.

L2 flush 128 MB per iter, 300 iter median, 2-sec hot warmup.
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

# All 16 configs: Q in {1,2,3,4} x seq in {128, 512, 4096, 32768}
CONFIGS = [
    (f"q{q}_seq{s_name}", q, s_int)
    for q in [1, 2, 3, 4]
    for s_name, s_int in [("128", 128), ("512", 512), ("4k", 4096), ("32k", 32768)]
]


def bench(q_len, seq_len, ni=NI):
    b = 1
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    last_page_len = (seq_len - 1) % PAGE_SIZE + 1 if seq_len > 0 else 0

    # KV cache: NHD layout (num_pages, 2, page_size, num_kv_heads, head_dim)
    kv_cache = torch.randn(num_pages, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                           dtype=DTYPE, device=device)
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    if q_len == 1:
        # Decode wrapper for Q=1 (matches the Q=1 baseline)
        q = torch.randn(b, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "NHD", use_tensor_cores=True,
        )
        wrapper.plan(
            kv_indptr, kv_indices, kv_last_page_len,
            NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE,
            q_data_type=DTYPE, kv_data_type=DTYPE,
        )
    else:
        # Prefill wrapper for Q>1 (causal among query tokens)
        q = torch.randn(q_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
        qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device=device)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
        wrapper.plan(
            qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
            NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE,
            causal=True,
            q_data_type=DTYPE, kv_data_type=DTYPE,
        )

    # Hot warmup
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

    # FLOPs: per query i, attended = (seq_len - q_len + 1 + i) tokens; sum over Q queries
    if q_len == 1:
        total_attended = seq_len
    else:
        total_attended = q_len * seq_len - q_len * (q_len - 1) // 2
    flops = 4 * NUM_QO_HEADS * total_attended * HEAD_DIM
    mem_bytes = 2 * seq_len * NUM_KV_HEADS * HEAD_DIM * 2
    tflops = flops / (median_us * 1e-6) / 1e12
    gbps = mem_bytes / (median_us * 1e-6) / 1e9
    return tflops, median_us, flops, mem_bytes, gbps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    print("=== FlashInfer Paged GQA UNIFIED Baseline (Q=1..4 x seq=128..32k) ===")
    print(f"Device:        {torch.cuda.get_device_name(0)}")
    print(f"FlashInfer:    {flashinfer.__version__}")
    print(f"NUM_QO_HEADS={NUM_QO_HEADS}  NUM_KV_HEADS={NUM_KV_HEADS}  HEAD_DIM={HEAD_DIM}  "
          f"PAGE_SIZE={PAGE_SIZE}  dtype=bf16")
    print(f"Q=1 uses Decode wrapper; Q>1 uses Prefill wrapper (causal=True)")
    print(f"Iterations: {NI} median, L2 flush per iter, {WARMUP_SEC:.0f}-sec warmup")
    print()

    for name, q_len, seq_len in CONFIGS:
        if args.config and args.config != name:
            continue
        tflops, us, flops, mem, gbps = bench(q_len, seq_len)
        print(f"{name:>10}: {tflops:>7.3f} TFLOPS  {us:>7.1f} us  "
              f"{gbps:>7.1f} GB/s  (Q={q_len} seq={seq_len})")


if __name__ == "__main__":
    main()
