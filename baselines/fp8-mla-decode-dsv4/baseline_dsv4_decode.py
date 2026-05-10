"""FlashMLA sparse FP8 MLA decode baseline — DeepSeek V3.2 / V4 (V32) shapes.

Calls flash_mla.flash_mla_with_kvcache with is_fp8_kvcache=True and sparse
indices, exactly as DSv4 inference does in vLLM (see
vllm/v1/attention/backends/mla/flashmla_sparse.py:965).

KV cache layout (V32 / DSv4), 656 bytes/token:
    512 fp8_e4m3 NoPE + 16 bytes (4 fp32 scales, one per 128-elem NoPE chunk)
    + 128 bytes (64 bf16 RoPE values, not quantized).

FLOPs counted by FlashMLA's own formula (tests/lib.py:count_flop_and_mem_vol_for_decode):
    flop = 2 * h_q * (b * s_q * topk) * (d_qk + d_v)
i.e. QK^T accumulates d_qk-dim per head per attended token, PV accumulates d_v-dim.

L2 flush between iterations: in real inference, attention + layernorm + other
ops evict L2 between MoE/attention calls, so the paged FP8 KV is NOT cached.
The 128 MB flush before each timed iter mirrors that. Without flush, FlashMLA's
per-call time is artificially low because partial weight reuse happens across
calls. This matches the methodology of baselines/fp8-group-gemm/.

Reuses FlashMLA's own test harness (tests/lib.py + tests/quant.py) for data
generation and FLOPs accounting, so this baseline cannot drift from FlashMLA's
internal reference.

Usage:
    python3 baselines/fp8-mla-decode-dsv4/baseline_dsv4_decode.py
"""
import argparse
import sys

# FlashMLA installation paths on catalyst-fleet1. The built .so is for cpython-312.
FLASHMLA_DIR = "/home/xinhaoc/mirage-cuda-agent/resources/flashmla-main"
sys.path.insert(0, FLASHMLA_DIR)
sys.path.insert(0, FLASHMLA_DIR + "/tests")

import torch
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")

import flash_mla
import lib
from lib import (
    RawTestParamForDecode,
    generate_testcase_for_decode,
    run_flash_mla_decode,
    count_flop_and_mem_vol_for_decode,
)

device = "cuda"
FLUSH_BYTES = 128 * 1024 * 1024
NI = 100  # number of timed iterations (median)

# DSv4 (V32) production decode shapes.
# Source: FlashMLA tests/test_flash_mla_sparse_decoding.py:103-105
#   (RawTestParam(0, 128, 2, 1, 32768, True, topk=2048, d_qk=576), [2, 64, 74, 128])
# Two head configs:
#   h_q=128 — TP=1 deployment (full 128 heads on one rank)
#   h_q=64  — TP>=2 deployment, or vLLM padding of small num_heads up to 64
configs = [
    # (name,                b,    h_q, s_q, h_kv, s_kv,  topk, d_qk, d_v)
    ("b2_h64",              2,    64,  2,   1,    32768, 2048, 576,  512),
    ("b64_h64",             64,   64,  2,   1,    32768, 2048, 576,  512),
    ("b128_h64",            128,  64,  2,   1,    32768, 2048, 576,  512),
    ("b2_h128",             2,    128, 2,   1,    32768, 2048, 576,  512),
    ("b64_h128",            64,   128, 2,   1,    32768, 2048, 576,  512),
    ("b128_h128",           128,  128, 2,   1,    32768, 2048, 576,  512),
]


def bench_one(name, b, h_q, s_q, h_kv, s_kv, topk, d_qk, d_v, seed=42):
    p = RawTestParamForDecode(
        b=b, h_q=h_q, s_q=s_q, h_kv=h_kv, s_kv=s_kv,
        is_varlen=False, topk=topk, d_qk=d_qk, d_v=d_v,
        is_all_indices_invalid=False,
        have_zero_seqlen_k=False,
        have_topk_length=False,
        enable_attn_sink=False,
        check_correctness=False,
        num_runs=0,
        seed=seed,
    ).to_test_param()

    t = generate_testcase_for_decode(p)
    sched_meta, _ = flash_mla.get_mla_metadata()

    for _ in range(10):
        run_flash_mla_decode(p, t, sched_meta, None)
    torch.cuda.synchronize()

    flush = torch.zeros(FLUSH_BYTES // 4, dtype=torch.int32, device=device)
    times_us = []
    for _ in range(NI):
        flush.zero_()
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        se.record()
        run_flash_mla_decode(p, t, sched_meta, None)
        ee.record()
        torch.cuda.synchronize()
        times_us.append(se.elapsed_time(ee) * 1000.0)
    times_us.sort()
    median_us = times_us[len(times_us) // 2]

    stats = count_flop_and_mem_vol_for_decode(p, t)
    tflops = stats.flop / (median_us * 1e-6) / 1e12
    return tflops, median_us, stats.flop, stats.mem_vol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Run only the named config (default: all)")
    args = parser.parse_args()

    print("=== FlashMLA Sparse FP8 MLA Decode Baseline (DSv4 / V32) ===")
    print(f"FlashMLA path: {FLASHMLA_DIR}")
    print(f"Device:        {torch.cuda.get_device_name(0)}")
    print(f"Iterations:    {NI} timed (median), L2 flush per iter, {FLUSH_BYTES//(1024*1024)} MB")
    print()

    for cfg in configs:
        name = cfg[0]
        if args.config and name != args.config:
            continue
        tflops, us, flop, mem = bench_one(*cfg)
        b, h_q, s_q, h_kv, s_kv, topk, d_qk, d_v = cfg[1:]
        print(f"{name}: {tflops:.2f} TFLOPS, {us:.1f} us  "
              f"(b={b} h_q={h_q} s_q={s_q} s_kv={s_kv} topk={topk} d_qk={d_qk} d_v={d_v}  "
              f"flops={flop/1e9:.2f}G mem={mem/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
