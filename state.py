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
from dataclasses import dataclass, field
from pathlib import Path

from .task_spec import (
    TaskSpec,
    compute_score,
    parse_kernel_output,
    parse_reference_output,
    should_advance_stage,
)


def _read_latest_tag_body(workspace_path: str | Path) -> str:
    """Return the latest tag's full commit message body, or '' if no tags."""
    ws = str(Path(workspace_path).resolve())
    try:
        tag = subprocess.check_output(
            ["git", "-C", ws, "describe", "--tags", "--abbrev=0"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    if not tag:
        return ""
    try:
        return subprocess.check_output(
            ["git", "-C", ws, "log", "-1", "--format=%B", tag],
            text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return ""


def get_best_results(workspace_path: str | Path) -> dict[str, float]:
    """Return per-config TFLOPS dict from the latest tagged kernel version.

    Convention from the agent's git workflow: every tag is by definition an
    improvement (failed and no-gain attempts get a commit body but no tag), so
    the most recent reachable tag is the current best.

    Returns own-kernel measurements, parsed via parse_kernel_output (handles
    KERNEL_RESULT JSON, v004 printf, and TFLOPS:Q<N>= commit format).
    """
    return parse_kernel_output(_read_latest_tag_body(workspace_path))


def get_best_reference(workspace_path: str | Path) -> dict[str, float]:
    """Return per-config REFERENCE TFLOPS from the latest tagged kernel version.

    The agent's benchmark must emit BOTH KERNEL_RESULT and KERNEL_RESULT_REFERENCE
    lines and commit them into the tag's commit body so we can score live.
    Returns {} if no reference line in the latest tag.
    """
    return parse_reference_output(_read_latest_tag_body(workspace_path))


# ─────────────────────────────────────────────────────────────────────────────
# RunState — a snapshot of "where are we right now"
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RunState:
    """Snapshot of the agent's current progress against a spec.

    Every field is computed together from the same (workspace, spec) pair, so a
    caller that holds a RunState can trust all fields are consistent with each
    other (unlike reading them one-by-one with separate calls where the best
    kernel could change between calls).
    """
    stage: str                                       # "REPRODUCE" or "OPTIMIZE"
    score: float                                     # aggregate per spec.scoring
    results: dict[str, float] = field(default_factory=dict)  # own-kernel TFLOPS per config
    reference: dict[str, float] = field(default_factory=dict)  # baseline TFLOPS per config
    ratios: dict[str, float] = field(default_factory=dict)   # results / reference per config
    worst_config: str = ""                           # config name with lowest ratio
    has_any_kernel: bool = False                     # False on a fresh workspace
    has_reference: bool = False                      # False if agent hasn't measured baseline yet


def compute_state(workspace_path: str | Path, spec: TaskSpec) -> RunState:
    """Build a RunState by combining git-tagged results with a task spec.

    Flow:
      1. Read the latest tag's per-config TFLOPS (get_best_results)
      2. Compute per-config ratios + aggregate score per spec.scoring
      3. Decide REPRODUCE/OPTIMIZE via spec.stage_gate
      4. Identify the bottleneck config (lowest ratio)

    Fresh workspace with no tags → stage=REPRODUCE, score=0, everything empty.
    This is the signal to _first_turn that the agent is starting from scratch.
    """
    results = get_best_results(workspace_path)
    reference = get_best_reference(workspace_path)

    if not results:
        return RunState(
            stage="REPRODUCE",
            score=0.0,
            has_any_kernel=False,
            has_reference=bool(reference),
            reference=reference,
        )

    score, ratios = compute_score(results, reference, spec)
    advance = should_advance_stage(score, ratios, spec)
    worst = min(ratios, key=ratios.get) if ratios else ""

    return RunState(
        stage="OPTIMIZE" if advance else "REPRODUCE",
        score=score,
        results=results,
        reference=reference,
        ratios=ratios,
        worst_config=worst,
        has_reference=bool(reference),
        has_any_kernel=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI for testing — `python3 -m ferret.state /path/to/workspace [task.yaml]`
# ─────────────────────────────────────────────────────────────────────────────


def _main() -> int:
    import sys
    if len(sys.argv) < 2:
        print(
            "usage: python3 -m ferret.state /path/to/workspace [task.yaml]",
            file=sys.stderr,
        )
        return 2
    ws = sys.argv[1]
    if not Path(ws).exists():
        print(f"ERROR: workspace not found: {ws}", file=sys.stderr)
        return 1
    if not (Path(ws) / ".git").exists():
        print(f"ERROR: no .git in workspace: {ws}", file=sys.stderr)
        return 1

    # Always show the raw results first
    results = get_best_results(ws)
    reference = get_best_reference(ws)
    if not results:
        print(f"no tagged improvements in {ws}")
        return 0
    print(f"latest tag's KERNEL_RESULT:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    if reference:
        print(f"latest tag's KERNEL_RESULT_REFERENCE:")
        for k, v in reference.items():
            print(f"  {k}: {v}")
    else:
        print(f"latest tag's KERNEL_RESULT_REFERENCE: (not present — agent must measure)")

    # If a task.yaml was provided, also show the full RunState
    if len(sys.argv) >= 3:
        from .task_spec import load_task_spec

        spec_path = sys.argv[2]
        if not Path(spec_path).exists():
            print(f"ERROR: task spec not found: {spec_path}", file=sys.stderr)
            return 1
        spec = load_task_spec(spec_path)
        state = compute_state(ws, spec)

        print()
        print(f"RunState vs {spec.name}:")
        print(f"  stage           : {state.stage}")
        print(f"  score           : {state.score:.3f} (via {spec.scoring})")
        print(f"  worst_config    : {state.worst_config or '(none)'}")
        print(f"  has_any_kernel  : {state.has_any_kernel}")
        print(f"  has_reference   : {state.has_reference}")
        print(f"  per-config ratios (kernel / reference):")
        for cfg in spec.configs:
            tflops = state.results.get(cfg.name, 0.0)
            ref = state.reference.get(cfg.name, 0.0)
            ratio = state.ratios.get(cfg.name, 0.0)
            marker = " ✓" if ratio >= cfg.target_ratio else (
                "  ← WORST" if cfg.name == state.worst_config else ""
            )
            print(
                f"    {cfg.name}: {tflops:6.1f} / {ref:6.1f} "
                f"= {ratio*100:5.1f}% (target {cfg.target_ratio*100:.0f}%){marker}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
