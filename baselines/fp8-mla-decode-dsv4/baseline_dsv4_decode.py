"""FlashMLA sparse FP8 MLA decode baseline — DeepSeek V4 (MODEL1 layout).

Targets DSv4's CSA core attention kernel: FlashMLA
`csrc/sm100/decode/head64/kernel.cuh` instantiated with
`ModelType::MODEL1` (NOT V32 — V32 is V3.2). The MODEL1 enum is the
generic name FlashMLA gave to the V4 layout when open-sourcing before
V4's public release. Confirmed by vLLM's deepseek_v4_attention.py:221
hardcoding the per-token bytes layout that exactly matches MODEL1:
    head_bytes = 448 (fp8 NoPE) + 64*2 (bf16 RoPE) + 448//64 (scales) + 1 = 584

MODEL1 / V4 KV cache layout (per-token, 584 bytes):
    448 fp8_e4m3 NoPE
  +  64 bf16 RoPE (last 64 dims, unquantized)
  +   7 e8m0 FP8 scale factors (one per 64-element NoPE chunk)
  +   1 byte pad
This differs from V32 (V3.2): 512 NoPE / 4 scales / 656 bytes, V==NoPE only.
In MODEL1 the value uses the FULL 512 dims (V_HAVE_ROPE=true).

V4 production attention call structure (per vLLM's
deepseek_v4_attention.py:898): flash_mla_with_kvcache receives BOTH a main
KV cache (the SWA window) and an `extra_k_cache` (the CSA-compressed pool),
with corresponding `indices` (SWA window) and `extra_indices_in_kvcache`
(CSA top-k from lightning indexer). FlashMLA test MODEL1 CONFIG1 (h_q=64,
extra_topk=512) matches V4-Flash; CONFIG2 (h_q=128, extra_topk=1024)
matches V4-Pro — see paper §4.2.1.

FLOPs counted by FlashMLA's own
tests/lib.py:count_flop_and_mem_vol_for_decode:
    flop = 2 * h_q * (b * s_q * (topk + extra_topk)) * (d_qk + d_v)

L2 flush + per-iter event pairs + median: same methodology as the FP8
grouped GEMM baseline.

This baseline reuses FlashMLA's own test harness (tests/lib.py +
tests/quant.py) for data generation, KV quantization, and FLOPs accounting
— so it cannot drift from FlashMLA's internal reference.

Usage:
    python3 baselines/fp8-mla-decode-dsv4/baseline_dsv4_decode.py
"""
import argparse
import os
import sys

# FlashMLA installation path (built extension)
FLASHMLA_DIR = os.environ.get("FLASHMLA_DIR", os.path.join(os.path.dirname(__file__), "../../resources/flashmla-main"))
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

# V4 (MODEL1) production decode shapes.
#
# Source: FlashMLA tests/test_flash_mla_sparse_decoding.py:107-109
#   MODEL1 CONFIG1: RawTestParam(0, 64,  2, 1, 16384, True, topk=128,
#                  d_qk=512, extra_s_k=16384, extra_topk=512,  block_size=256,
#                  extra_block_size=64)  → V4-Flash
#   MODEL1 CONFIG2: RawTestParam(0, 128, 2, 1, 16384, True, topk=128,
#                  d_qk=512, extra_s_k=16384, extra_topk=1024, block_size=256,
#                  extra_block_size=64)  → V4-Pro
# Batches per FlashMLA test: [2, 64, 74, 128]. We use {2, 64, 128}.
#
# Per V4 paper §4.2.1:
#   V4-Flash: 64 query heads, CSA top-k = 512
#   V4-Pro:  128 query heads, CSA top-k = 1024
#   Both:    s_q=2 (MTP), SWA window n_win=128
configs = [
    # (name,           b,    h_q,  extra_topk)
    ("b2_v4flash",     2,    64,   512),
    ("b64_v4flash",    64,   64,   512),
    ("b128_v4flash",   128,  64,   512),
    ("b2_v4pro",       2,    128,  1024),
    ("b64_v4pro",      64,   128,  1024),
    ("b128_v4pro",     128,  128,  1024),
]

# Shared parameters across all configs (matching FlashMLA MODEL1 test configs)
S_Q          = 2       # MTP / speculative depth = 2 (V4 trains with MTP=1
                       # per paper §4.2.1, but the inference kernel still
                       # supports s_q=2 for spec decoding)
H_KV         = 1       # MQA: single shared K/V head (paper §2.3.1 Eq 18-19)
S_KV         = 16384   # main (SWA) KV cache size
TOPK         = 128     # main topk = SWA window indices (per query)
EXTRA_S_K    = 16384   # extra (CSA compressed) KV cache size
D_QK         = 512     # MODEL1 head dim (vs V32's 576)
D_V          = 512     # value head dim
BLOCK_SIZE       = 256 # main KV cache page block size
EXTRA_BLOCK_SIZE = 64  # extra KV cache page block size


def bench_one(name, b, h_q, extra_topk, seed=42):
    p = RawTestParamForDecode(
        b=b, h_q=h_q, s_q=S_Q, h_kv=H_KV, s_kv=S_KV,
        is_varlen=False,
        topk=TOPK,
        d_qk=D_QK, d_v=D_V,
        extra_s_k=EXTRA_S_K, extra_topk=extra_topk,
        block_size=BLOCK_SIZE, extra_block_size=EXTRA_BLOCK_SIZE,
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

    print("=== FlashMLA Sparse FP8 MLA Decode Baseline (DSv4 / MODEL1) ===")
    print(f"FlashMLA path: {FLASHMLA_DIR}")
    print(f"Device:        {torch.cuda.get_device_name(0)}")
    print(f"Iterations:    {NI} timed (median), L2 flush per iter, "
          f"{FLUSH_BYTES // (1024 * 1024)} MB")
    print(f"KV layout:     MODEL1 (V4)  d_qk={D_QK} d_v={D_V}  "
          f"per-tok bytes = 448 NoPE + 128 RoPE + 7 scales + 1 pad = 584")
    print(f"Shared:        s_q={S_Q} h_kv={H_KV} s_kv={S_KV} topk={TOPK} "
          f"block_size={BLOCK_SIZE} extra_s_k={EXTRA_S_K} "
          f"extra_block_size={EXTRA_BLOCK_SIZE}")
    print()

    for cfg in configs:
        name, b, h_q, extra_topk = cfg
        if args.config and name != args.config:
            continue
        tflops, us, flop, mem = bench_one(*cfg)
        print(f"{name}: {tflops:.2f} TFLOPS, {us:.1f} us  "
              f"(b={b} h_q={h_q} extra_topk={extra_topk}  "
              f"flops={flop / 1e9:.2f}G mem={mem / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
