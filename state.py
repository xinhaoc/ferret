"""Run state for ferret — what's the current best, what stage are we in.

This module is the bridge between git history (where the agent saves its
work) and the orchestrator's decisions (which prompt to send next).

Design:
- Pure functions where possible. Easy to test in isolation.
- Sync subprocess. Git ops on a local repo are <100ms; running them through
  the async shell layer (which is built for the agent's tool calls) adds
  complexity for no real benefit.
- No imports from orchestrator or agents — this module is leaf-level. It only
  depends on task_spec for parsing and scoring.

Module growth plan:
  step 1 (here): get_best_results()           — latest tagged kernel's per-config TFLOPS
  step 2:        RunState dataclass            — stage + score + ratios + worst config
  step 2:        compute_state()               — full RunState from workspace + spec
  later:         tag_history()                 — all tagged versions for trend analysis
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .task_spec import parse_kernel_output


def get_best_results(workspace_path: str | Path) -> dict[str, float]:
    """Return per-config TFLOPS dict from the latest tagged kernel version.

    Convention from the agent's git workflow: every tag is by definition an
    improvement (failed and no-gain attempts get a commit body but no tag), so
    the most recent reachable tag is the current best. This matches what the
    agent's own revert command (`git checkout $(git describe --tags --abbrev=0)`)
    already loads.

    Args:
        workspace_path: directory containing the agent's kernel.cu and .git

    Returns:
        Dict like {"Q1": 31.0, "Q2": 54.7, "Q4": 87.7}, or {} if there are no
        tags, or the tag exists but its commit body has no parseable result line.

    The parser (task_spec.parse_kernel_output) handles three formats in this
    priority order: KERNEL_RESULT JSON line, v004 printf "Q_LEN=N: X TFLOPS",
    and commit-body "Q<N>=<float>". The first format that matches wins.
    """
    ws = str(Path(workspace_path).resolve())

    try:
        tag = subprocess.check_output(
            ["git", "-C", ws, "describe", "--tags", "--abbrev=0"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}

    if not tag:
        return {}

    try:
        body = subprocess.check_output(
            ["git", "-C", ws, "log", "-1", "--format=%B", tag],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return {}

    return parse_kernel_output(body)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI for testing — `python3 -m ferret.state /path/to/workspace`
# ─────────────────────────────────────────────────────────────────────────────


def _main() -> int:
    import sys
    if len(sys.argv) < 2:
        print("usage: python3 -m ferret.state /path/to/workspace", file=sys.stderr)
        return 2
    ws = sys.argv[1]
    if not Path(ws).exists():
        print(f"ERROR: workspace not found: {ws}", file=sys.stderr)
        return 1
    if not (Path(ws) / ".git").exists():
        print(f"ERROR: no .git in workspace: {ws}", file=sys.stderr)
        return 1

    results = get_best_results(ws)
    if not results:
        print(f"no tagged improvements in {ws}")
        return 0

    print(f"latest tag's per-config TFLOPS:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
