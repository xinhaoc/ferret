"""ferret profile CLI — sync wrapper around tools.profiler for subagent use.

The original ``tools/profiler.py`` exposes an async ``Profiler.quick_profile``
that requires a shell-fn / event-loop wired into motus's agent runtime. The
Claude-Code subagents talk to ferret through ordinary ``Bash``, so this module
provides a fully synchronous CLI that wraps the same ncu invocation + CSV
parser and prints the structured summary directly.

Usage::

    python3 -m ferret.profile <workspace> [--kernel NAME] [--binary ./kernel]
                              [--no-pickgpu] [--save-baseline]

Behaviour:
  1. ``cd <workspace>``.
  2. Run ``eval $(<ferret-root>/pick_gpu.sh)`` to pick a quiet GPU (unless
     ``--no-pickgpu``).
  3. If ``<workspace>/kernel.cu`` exists but ``<workspace>/<binary>`` does
     not, refuse to run — the kernel must be compiled and runnable before
     profiling. (We don't auto-compile because each task has its own nvcc
     flags; the mainthread knows them, this wrapper does not.)
  4. Run ``ncu --csv --metrics ... -c 1 --launch-skip 1 -k <name> <binary>``
     with ``TMPDIR=/tmp/$USER`` set (see docs/dev-memory/machine.md for the
     reason).
  5. Parse with ``tools.profiler.parse_ncu_csv``, print
     ``ProfileMetrics.summary()``.
  6. Save the parsed metrics to ``<workspace>/.profile_last.json`` (atomic
     write). If a previous snapshot exists, also print a one-line delta
     versus the previous run.

Exit code 0 on success, non-zero on ncu failure / missing inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .tools.profiler import QUICK_METRICS, extract_kernel_names, parse_ncu_csv


def _ferret_root() -> Path:
    """Return the directory containing pick_gpu.sh and this module."""
    return Path(__file__).resolve().parent


def _resolve_gpu_export(pickgpu_path: Path) -> dict[str, str]:
    """Run pick_gpu.sh and parse its `export CUDA_VISIBLE_DEVICES=...` line.

    pick_gpu.sh is intended to be `eval`'d, so we mimic that by reading its
    stdout and lifting `export FOO=bar` assignments into a dict that we can
    merge into the env passed to ncu.
    """
    if not pickgpu_path.exists():
        return {}
    try:
        out = subprocess.check_output(
            ["bash", str(pickgpu_path)], text=True, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, OSError):
        return {}
    env: dict[str, str] = {}
    for line in out.splitlines():
        m = re.match(r"\s*export\s+(\w+)=(.*)$", line)
        if m:
            k = m.group(1)
            v = m.group(2).strip().strip('"').strip("'")
            env[k] = v
    return env


def _read_first_kernel_name(kernel_cu: Path) -> str:
    if not kernel_cu.exists():
        return ""
    try:
        src = kernel_cu.read_text()
    except OSError:
        return ""
    names = extract_kernel_names(src)
    return names[0] if names else ""


def _delta_line(prev: dict, curr: dict) -> str:
    """One-line summary: keys that changed >5% relative."""
    interesting = (
        "duration_us",
        "sm_throughput_pct",
        "memory_throughput_pct",
        "warp_occupancy_pct",
        "tensor_active_pct",
    )
    parts = []
    for k in interesting:
        p = float(prev.get(k, 0.0))
        c = float(curr.get(k, 0.0))
        if p == 0 and c == 0:
            continue
        if p == 0:
            parts.append(f"{k}: +new {c:.2f}")
            continue
        rel = (c - p) / p
        if abs(rel) >= 0.05:
            parts.append(f"{k}: {p:.2f} -> {c:.2f} ({rel*100:+.1f}%)")
    return "  ".join(parts) if parts else "no notable change (<5% rel) vs last profile"


def _main() -> int:
    ap = argparse.ArgumentParser(prog="ferret.profile", description=__doc__)
    ap.add_argument("workspace", help="Path to workspace dir (must contain kernel binary).")
    ap.add_argument("--kernel", default="", help="__global__ name to filter (-k). Default: first one in kernel.cu.")
    ap.add_argument("--binary", default="./kernel", help="Binary to profile, relative to workspace.")
    ap.add_argument("--no-pickgpu", action="store_true", help="Don't run pick_gpu.sh.")
    ap.add_argument(
        "--save-baseline", action="store_true",
        help="After running, copy .profile_last.json to .profile_baseline.json so future runs diff against this one.",
    )
    args = ap.parse_args()

    ws = Path(args.workspace).resolve()
    if not ws.exists():
        print(f"ERROR: workspace not found: {ws}", file=sys.stderr)
        return 2

    binary = ws / args.binary
    if not binary.exists():
        print(
            f"ERROR: binary not found: {binary}\n"
            "       Compile your kernel first (the profile CLI does not "
            "auto-compile because nvcc flags are task-specific).",
            file=sys.stderr,
        )
        return 2

    root = _ferret_root()
    pickgpu = root / "pick_gpu.sh"

    kname = args.kernel or _read_first_kernel_name(ws / "kernel.cu")
    if not kname:
        print(
            "WARN: no __global__ name supplied / found in kernel.cu — "
            "ncu will profile all launches.", file=sys.stderr,
        )

    env = os.environ.copy()
    env.setdefault("TMPDIR", f"/tmp/{os.environ.get('USER', 'user')}")
    if not args.no_pickgpu:
        env.update(_resolve_gpu_export(pickgpu))

    k_flag = ["-k", kname] if kname else []
    cmd = [
        "ncu", "--csv", "--metrics", QUICK_METRICS,
        "-c", "1", "--launch-skip", "1",
        *k_flag,
        args.binary,
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=ws, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180,
        )
    except FileNotFoundError:
        print("ERROR: ncu not found on PATH.", file=sys.stderr)
        return 127
    except subprocess.TimeoutExpired:
        print("ERROR: ncu timed out (>180s).", file=sys.stderr)
        return 124

    metrics = parse_ncu_csv(proc.stdout)
    if proc.returncode != 0 and metrics.duration_us == 0.0:
        print(proc.stdout, file=sys.stderr)
        print(f"ERROR: ncu exit {proc.returncode}", file=sys.stderr)
        return proc.returncode

    print(f"=== Profile of {binary.name} (kernel={kname or 'all'}) ===")
    print(metrics.summary())

    # persist JSON for delta + downstream consumers
    snapshot = {k: v for k, v in asdict(metrics).items() if k != "raw_csv"}
    snapshot["kernel"] = kname
    snapshot_path = ws / ".profile_last.json"
    prev_snapshot = None
    if snapshot_path.exists():
        try:
            prev_snapshot = json.loads(snapshot_path.read_text())
        except (OSError, json.JSONDecodeError):
            prev_snapshot = None

    tmp = snapshot_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snapshot, indent=2))
    tmp.replace(snapshot_path)

    if args.save_baseline:
        (ws / ".profile_baseline.json").write_text(json.dumps(snapshot, indent=2))

    if prev_snapshot:
        print()
        print("vs previous profile:")
        print(" ", _delta_line(prev_snapshot, snapshot))
    elif (ws / ".profile_baseline.json").exists():
        try:
            baseline = json.loads((ws / ".profile_baseline.json").read_text())
            print()
            print("vs saved baseline:")
            print(" ", _delta_line(baseline, snapshot))
        except (OSError, json.JSONDecodeError):
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
