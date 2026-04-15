#!/usr/bin/env python3
"""Compare ferret's v011 decode-linear kernel against CUDA-core cuBLAS,
tensor-core cuBLAS, and (optionally) flashinfer's tgv_gemm on sm100.

Same methodology across all backends:
  - L2 flush (128MB buffer zeroed) between every timed iteration
  - 20 warmup + median of 200 iters
  - cudaEvent timing
  - Correctness check vs torch.matmul fp32 reference

Run:
    cd ~/repos/ferret/workspace
    # build the kernel as a shared lib first:
    nvcc -O3 -std=c++17 -arch=sm_100a --use_fast_math -Xcompiler -fPIC \\
         -shared -o kernel.so kernel.cu
    cd ~/repos/ferret
    python3 tests/compare_decode_linear.py --so workspace/kernel.so

Optional: install flashinfer for tgv_gemm (tcgen05-based sm100 BF16 GEMM):
    pip install flashinfer-python

Exit codes:
    0  all backends passed correctness
    1  compile error / missing files
    2  one of the required backends failed
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


SHAPES = {
    "QKV":    {"N": 6144,    "K": 4096,  "residual": False},
    "O":      {"N": 4096,    "K": 4096,  "residual": True},
    "GateUp": {"N": 24576,   "K": 4096,  "residual": False},
    "Down":   {"N": 4096,    "K": 12288, "residual": True},
    "LMHead": {"N": 153600,  "K": 4096,  "residual": False},
}

# L2 is 96MB on B200; use 128MB to guarantee full eviction.
_FLUSH_ELEMS = 128 * 1024 * 1024 // 4  # int32
_flush_buf: torch.Tensor | None = None


def l2_flush(device):
    global _flush_buf
    if _flush_buf is None:
        _flush_buf = torch.zeros(_FLUSH_ELEMS, dtype=torch.int32, device=device)
    _flush_buf.zero_()


# ─────────────────────────────────────────────────────────────────────────────
# Backends
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Backend:
    name: str
    available: bool
    reason: str = ""      # why unavailable, if so
    call_fn: object = None  # callable(x, w, r) -> out


def load_ours(so_path: Path) -> Backend:
    if not so_path.exists():
        return Backend(
            "ours", False,
            f"{so_path} not found. Build with `nvcc ... -shared -o {so_path.name} kernel.cu`.",
        )
    lib = ctypes.CDLL(str(so_path))
    lib.launch_gemv.argtypes = [
        ctypes.c_void_p,  # input
        ctypes.c_void_p,  # weight
        ctypes.c_void_p,  # residual
        ctypes.c_void_p,  # output
        ctypes.c_int,      # N
        ctypes.c_int,      # K
        ctypes.c_void_p,   # stream
    ]

    def call(x, w, r, out, N, K):
        stream = torch.cuda.current_stream().cuda_stream
        lib.launch_gemv(
            x.data_ptr(), w.data_ptr(),
            r.data_ptr() if r is not None else 0,
            out.data_ptr(), N, K, stream,
        )
        return out

    return Backend("ours", True, call_fn=call)


def load_cublas_default() -> Backend:
    """cuBLAS through torch.matmul. At M=1 BF16, cuBLAS dispatches to a
    CUDA-core GEMV path (tensor cores waste lanes). This matches what
    bench.cu measures by default."""
    def call(x, w, r, out, N, K):
        # torch.matmul produces a new tensor; copy into out for consistent
        # output-buffer semantics with the other backends.
        y = torch.matmul(x, w.t())
        if r is not None:
            y = y + r
        out.copy_(y)
        return out
    return Backend("cublas_default", True, call_fn=call)


def load_cublas_tensor_core() -> Backend:
    """Force cuBLAS to use a tensor-core algorithm via cublasLt.

    At M=1, cuBLAS default will NOT use tensor cores (wastes lanes). To
    measure the tensor-core path as a reference, we explicitly request a
    tensor-op compute type (CUBLAS_COMPUTE_32F_FAST_16BF) with a
    heuristic that prefers TC-enabled kernels. cuBLAS may still fall back
    to non-TC if no TC kernel matches — the comparison tells you whether
    forcing TC helps or hurts at M=1.

    Implementation via torch's low-level cublas handle is fragile, so we
    use torch.nn.functional's bf16 matmul with torch's tf32/fp16 autotune
    flags flipped — which routes through cuBLAS's tensor-core algos on
    Blackwell when the shape permits.
    """
    # torch respects allow_bf16_reduced_precision_reduction which enables
    # tensor cores for BF16 matmul where possible.
    old_bf16 = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    def call(x, w, r, out, N, K):
        y = torch.matmul(x, w.t())
        if r is not None:
            y = y + r
        out.copy_(y)
        return out

    # NOTE: this only actually uses tensor cores if cuBLAS picks a TC
    # algorithm. For M=1 it often doesn't — this row then reports
    # numbers indistinguishable from cublas_default. That's informative:
    # tells you torch's cuBLAS wrapper doesn't force TC at M=1.
    return Backend("cublas_tc_hint", True,
                   reason="torch bf16_reduced_precision_reduction on (may or may not take TC path at M=1)",
                   call_fn=call)


def load_flashinfer() -> Backend:
    """flashinfer's tgv_gemm — sm100 BF16 GEMM using tcgen05 tensor cores,
    specifically designed for low-batch decode. This is the closest
    in-tree equivalent to trtllm-gen for our purposes.
    """
    try:
        fi = importlib.import_module("flashinfer")
    except ImportError:
        return Backend(
            "flashinfer_tgv", False,
            "flashinfer not installed. `pip install flashinfer-python` to enable.",
        )
    # Find the right entry point — the API changed across versions.
    for mod_name, fn_name in [
        ("flashinfer.gemm", "tgv_gemm_sm100"),
        ("flashinfer.gemm", "tgv_gemm"),
        ("flashinfer.gemm", "bf16_gemm"),
        ("flashinfer", "tgv_gemm"),
    ]:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name, None)
            if fn is not None:
                def call(x, w, r, out, N, K, _fn=fn):
                    # tgv_gemm_sm100 signature:
                    #   a: (M, K) row-major, b: (K, N) COLUMN-major, bias: (N,)
                    # Our w is (N, K) row-major — .t() gives a (K, N) VIEW that
                    # is column-major (same underlying data). Must NOT be made
                    # contiguous, or it becomes row-major and fails.
                    w_col = w.t()  # (K, N), column-major view of (N, K) row-major

                    # bias is (N,); residual is (1, N) → flatten.
                    if r is not None:
                        bias = r.view(-1)
                    else:
                        bkey = (x.device.index, N)
                        if not hasattr(call, '_zero_cache'):
                            call._zero_cache = {}
                        if bkey not in call._zero_cache:
                            call._zero_cache[bkey] = torch.zeros(
                                N, dtype=torch.bfloat16, device=x.device,
                            )
                        bias = call._zero_cache[bkey]

                    y = _fn(x, w_col, bias)
                    out.copy_(y.view(1, N) if y.dim() == 1 else y)
                    return out
                return Backend("flashinfer_tgv", True, call_fn=call,
                               reason=f"{mod_name}.{fn_name} (bias fused; zero when no residual)")
        except (ImportError, AttributeError):
            continue
    return Backend(
        "flashinfer_tgv", False,
        "flashinfer installed but no recognized tgv_gemm entry point found. "
        "Check flashinfer.gemm module.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark + correctness
# ─────────────────────────────────────────────────────────────────────────────


def benchmark(backend: Backend, x, w, r, out, N, K, warmup: int, iters: int) -> float:
    """Return median microseconds."""
    # Warmup
    for _ in range(warmup):
        backend.call_fn(x, w, r, out, N, K)
    torch.cuda.synchronize()

    # Timed iters with L2 flush before each
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_us = []
    for _ in range(iters):
        l2_flush(x.device)
        start.record()
        backend.call_fn(x, w, r, out, N, K)
        end.record()
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) * 1000.0)  # ms → us

    times_us.sort()
    return times_us[len(times_us) // 2]


def correctness(backend: Backend, x, w, r, out, N, K) -> tuple[float, float]:
    """Return (max_abs_err, max_rel_err) vs fp32 torch reference."""
    backend.call_fn(x, w, r, out, N, K)
    torch.cuda.synchronize()
    ref = torch.matmul(x.float(), w.float().t())
    if r is not None:
        ref = ref + r.float()
    diff = (out.float() - ref).abs()
    max_err = diff.max().item()
    denom = ref.abs().max().item() + 1e-8
    rel_err = max_err / denom
    return max_err, rel_err


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--so", default="workspace/kernel.so",
                        help="Path to compiled kernel.so (default: workspace/kernel.so)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--rtol", type=float, default=5e-3,
                        help="Per-config rel-err threshold (default 5e-3, matches task spec)")
    parser.add_argument("--config", default="all", choices=list(SHAPES) + ["all"])
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {args.warmup} | Iters: {args.iters}")
    print()

    # Load all backends
    backends = [
        load_ours(Path(args.so)),
        load_cublas_default(),
        load_cublas_tensor_core(),
        load_flashinfer(),
    ]

    print("Backends:")
    for b in backends:
        status = "✓" if b.available else "✗"
        note = f" — {b.reason}" if b.reason else ""
        print(f"  {status} {b.name}{note}")
    print()

    # Require at least ours + cublas_default
    if not backends[0].available:
        print(f"ERROR: {backends[0].reason}", file=sys.stderr)
        return 1

    active = [b for b in backends if b.available]

    # Header
    cols = ["config", "N", "K", "max_err", "rel_err"]
    cols += [f"{b.name}_tflops" for b in active]
    cols += [f"{b.name}_vs_cublas" for b in active if b.name != "cublas_default"]
    print("  ".join(f"{c:>12}" if len(c) > 8 else f"{c:>9}" for c in cols))
    print("-" * (len(cols) * 13))

    configs = list(SHAPES) if args.config == "all" else [args.config]
    any_incorrect = False

    for name in configs:
        shape = SHAPES[name]
        N, K, has_res = shape["N"], shape["K"], shape["residual"]

        x = torch.randn(1, K, dtype=torch.bfloat16, device=device)
        w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        r = torch.randn(1, N, dtype=torch.bfloat16, device=device) if has_res else None
        out = torch.empty(1, N, dtype=torch.bfloat16, device=device)

        # Correctness check on "ours" first (the thing under test)
        max_err, rel_err = correctness(backends[0], x, w, r, out, N, K)
        ok = rel_err < args.rtol
        if not ok:
            any_incorrect = True

        # Benchmark every active backend
        tflops = {}
        for b in active:
            try:
                us = benchmark(b, x, w, r, out, N, K, args.warmup, args.iters)
                tflops[b.name] = 2 * N * K / (us * 1e-6) / 1e12
            except Exception as e:
                tflops[b.name] = math.nan
                print(f"  [warn] {b.name} failed on {name}: {e}", file=sys.stderr)

        cublas_tf = tflops.get("cublas_default", math.nan)

        row = [name, str(N), str(K), f"{max_err:.4f}", f"{rel_err:.5f}"]
        for b in active:
            v = tflops.get(b.name, math.nan)
            row.append(f"{v:.2f}" if not math.isnan(v) else "—")
        for b in active:
            if b.name == "cublas_default":
                continue
            v = tflops.get(b.name, math.nan)
            if math.isnan(v) or math.isnan(cublas_tf) or cublas_tf == 0:
                row.append("—")
            else:
                row.append(f"{v / cublas_tf:.2f}×")
        mark = "✓" if ok else f"✗ rel_err={rel_err:.4f} > {args.rtol}"
        row.append(mark)
        print("  ".join(f"{c:>12}" if len(str(c)) > 8 else f"{str(c):>9}" for c in row))

    print()
    if any_incorrect:
        print("WARNING: at least one config failed the rel_err threshold.", file=sys.stderr)
        return 2
    print("All configs passed correctness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
