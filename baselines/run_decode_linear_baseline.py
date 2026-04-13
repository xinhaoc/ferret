#!/usr/bin/env python3
"""BF16 GEMV baseline for Qwen3-8B decode linear projections.

Measures median latency + TFLOPS for the 5 decode-phase linear ops at M=1.
Starts with torch.matmul (cuBLAS underneath) as a pragmatic reference;
agent is expected to extend this to call trtllm-gen directly for a tighter
baseline comparison (trtllm-gen is typically 5–10% faster than vanilla cuBLAS
on these shapes via tile/split tuning).

Usage:
    python3 baselines/run_decode_linear_baseline.py                        # all shapes
    python3 baselines/run_decode_linear_baseline.py --config QKV           # single
    python3 baselines/run_decode_linear_baseline.py --backend cublas       # default
    python3 baselines/run_decode_linear_baseline.py --iters 200            # median over 200

Emits (stdout, parsed by ferret's KERNEL_RESULT parser):
    KERNEL_RESULT {"QKV": 4.52, "O": 3.91, "GateUp": 5.07, "Down": 4.82, "LMHead": 5.34}

Per-shape human-readable output also shown on separate lines.
"""

import argparse
import json
import sys
from dataclasses import dataclass


SHAPES = {
    "QKV":    {"N": 6144,    "K": 4096,  "residual": False},
    "O":      {"N": 4096,    "K": 4096,  "residual": True},
    "GateUp": {"N": 24576,   "K": 4096,  "residual": False},
    "Down":   {"N": 4096,    "K": 12288, "residual": True},
    "LMHead": {"N": 153600,  "K": 4096,  "residual": False},
}


@dataclass
class Result:
    name: str
    N: int
    K: int
    median_us: float
    tflops: float
    bytes_moved_gb: float
    hbm_gbs: float


def flops_count(M: int, N: int, K: int) -> int:
    return 2 * M * N * K  # multiply + add per MAC


def bytes_moved(M: int, N: int, K: int, residual: bool) -> int:
    # bf16 everywhere (2 bytes)
    dtype_bytes = 2
    weight = N * K * dtype_bytes
    activation = M * K * dtype_bytes
    output = M * N * dtype_bytes
    residual_bytes = M * N * dtype_bytes if residual else 0
    return weight + activation + output + residual_bytes


def measure_torch_cublas(N: int, K: int, residual: bool, warmup: int, iters: int) -> float:
    """Returns median microseconds for one call. Backend: torch (calls cuBLAS for bf16 matmul)."""
    import torch
    assert torch.cuda.is_available(), "CUDA not available — script requires a GPU."
    device = torch.device("cuda")
    dtype = torch.bfloat16

    x = torch.randn(1, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)
    res = torch.randn(1, N, dtype=dtype, device=device) if residual else None

    def call():
        y = torch.matmul(x, w.t())
        if res is not None:
            y = y + res
        return y

    # Warmup
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_us = []
    for _ in range(iters):
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) * 1000.0)  # ms → us

    times_us.sort()
    return times_us[len(times_us) // 2]


def measure_trtllm_gen(N: int, K: int, residual: bool, warmup: int, iters: int) -> float:
    """Placeholder for trtllm-gen BF16 GEMV path. Agent should implement this.

    Options for implementation:
      1. Compile + link trtllm-gen's BF16 GEMM kernel from
         resources/tensorrt-llm-1.2.0/cpp/tensorrt_llm/kernels/trtllmGenKernels/gemm/
      2. Use flashinfer's tgv_gemm Python binding if available
      3. Call into TensorRT-LLM's Python API if installed

    For now raises NotImplementedError. Agent extends this function
    as part of its first REPRODUCE-stage task (see task.yaml hint).
    """
    raise NotImplementedError(
        "trtllm-gen baseline not wired up yet. "
        "Agent: read resources/tensorrt-llm-1.2.0/cpp/tensorrt_llm/kernels/trtllmGenKernels/gemm/ "
        "or resources/flashinfer-0.6.7.post3/include/flashinfer/gemm/tgv_gemm.cuh "
        "and implement this function. Until then, --backend cublas runs."
    )


def run_one(name: str, shape: dict, backend: str, warmup: int, iters: int) -> Result:
    N, K, residual = shape["N"], shape["K"], shape["residual"]
    if backend == "cublas":
        median_us = measure_torch_cublas(N, K, residual, warmup, iters)
    elif backend == "trtllm_gen":
        median_us = measure_trtllm_gen(N, K, residual, warmup, iters)
    else:
        raise ValueError(f"unknown backend: {backend}")

    tflops = flops_count(1, N, K) / (median_us * 1e-6) / 1e12
    moved_gb = bytes_moved(1, N, K, residual) / 1e9
    hbm_gbs = moved_gb / (median_us * 1e-6)
    return Result(name, N, K, median_us, tflops, moved_gb, hbm_gbs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="all", choices=list(SHAPES) + ["all"])
    parser.add_argument(
        "--backend", default="cublas",
        choices=["cublas", "trtllm_gen"],
        help="cublas (default) uses torch.matmul. trtllm_gen requires agent to wire it up.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    configs = list(SHAPES) if args.config == "all" else [args.config]

    print(f"Backend: {args.backend}")
    print(f"Warmup: {args.warmup} | Iters: {args.iters}")
    print()
    print(f"{'name':8} {'N':>7} {'K':>6}  {'time_us':>9}  {'TFLOPS':>8}  {'HBM GB/s':>10}  {'bytes':>6}")
    print("-" * 70)

    results: dict[str, float] = {}
    for name in configs:
        try:
            r = run_one(name, SHAPES[name], args.backend, args.warmup, args.iters)
        except NotImplementedError as e:
            print(f"{name}: {e}", file=sys.stderr)
            return 2
        print(
            f"{r.name:8} {r.N:>7} {r.K:>6}  {r.median_us:>9.2f}  "
            f"{r.tflops:>8.2f}  {r.hbm_gbs:>10.1f}  {r.bytes_moved_gb:>5.2f}GB"
        )
        results[name] = round(r.tflops, 2)

    # Machine-parseable line for ferret's result parser
    print()
    print(f"KERNEL_RESULT {json.dumps(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
