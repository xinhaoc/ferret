"""ferret.cc_goal — render a concrete `/goal` string from a workspace's task.yaml.

This module is consumed by ``scripts/cc-run.sh`` (and may be called by the
mirage dispatcher subagent) to produce a single-line, numeric goal statement
that ``--append-system-prompt`` and ``/goal`` can both consume. The goal is
intentionally specific so the mainthread's session-scoped Stop hook has a
crisp success condition rather than an abstract "iterate until done".

Usage:

    python3 -m ferret.cc_goal <workspace>            # reads <workspace>/task.yaml
    python3 -m ferret.cc_goal --task-yaml <file>     # use any task.yaml path
    python3 -m ferret.cc_goal <workspace> --json     # structured form

The rendered text looks like::

    Beat SOTA `trtllm-gen MLA decode` in workspace1/kernel.cu. Required per
    workspace1/task.yaml: Q1 reaches ≥100% of trtllm-gen TFLOPS, Q2 ≥100%,
    Q4 ≥100% (min_ratio scoring). Iterate write → compile → benchmark →
    commit + tag → reviewer continuously; do not stop until python3 -m
    ferret.state reports advance? True AND every config row shows ✓. See
    CLAUDE.md §6.5 for the exhaustive stop conditions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .task_spec import load_task_spec, TaskSpec


def _format_ratio(r: float) -> str:
    """Render `target_ratio` as a human percentage relative to baseline.

    1.00 → "100%"
    1.10 → "110% (i.e. beat by 10%)"
    0.95 → "95%"
    """
    pct = r * 100.0
    if r > 1.0:
        return f"≥{pct:.0f}% (i.e. beat by {(r - 1.0) * 100:.0f}%)"
    if r < 1.0:
        return f"≥{pct:.0f}% (match within {(1.0 - r) * 100:.0f}%)"
    return "≥100% (match exactly)"


def render_goal(spec: TaskSpec, workspace: str) -> str:
    """Return the one-line goal text for ``scripts/cc-run.sh``."""
    ws = workspace
    sota = spec.baseline.source
    cfg_parts = ", ".join(
        f"{c.name} {_format_ratio(c.target_ratio)}" for c in spec.configs
    )
    extras = []
    if spec.stage_gate.strict:
        extras.append("strict stage gate — every config must hit its target before the run can declare done")
    extras.append(f"scoring: {spec.scoring}")

    return (
        f"Beat SOTA `{sota}` in {ws}/kernel.cu. Required per {ws}/task.yaml: "
        f"{cfg_parts}. {'; '.join(extras)}. Iterate write → compile → "
        f"benchmark → commit + tag → reviewer continuously; do not stop until "
        f"`python3 -m ferret.state {ws} {ws}/task.yaml` reports advance? True "
        f"AND every per-config row shows the ✓ marker. See CLAUDE.md §6.5 for "
        f"exhaustive stop conditions and forbidden stop reasons."
    )


def render_goal_json(spec: TaskSpec, workspace: str) -> dict:
    """Structured form — useful when the caller (e.g. the mirage dispatcher)
    needs the parts separately rather than baked into one string."""
    return {
        "workspace": workspace,
        "sota": spec.baseline.source,
        "scoring": spec.scoring,
        "stage_gate": {
            "ratio": spec.stage_gate.ratio,
            "strict": spec.stage_gate.strict,
        },
        "configs": [
            {
                "name": c.name,
                "target_ratio": c.target_ratio,
                "human": _format_ratio(c.target_ratio),
                "weight": c.weight,
            }
            for c in spec.configs
        ],
        "stop_condition": (
            f"python3 -m ferret.state {workspace} {workspace}/task.yaml "
            "reports advance? True AND every config ✓"
        ),
        "goal_line": render_goal(spec, workspace),
    }


def _main() -> int:
    ap = argparse.ArgumentParser(prog="ferret.cc_goal")
    ap.add_argument(
        "workspace", nargs="?",
        help="Path to workspace dir (must contain task.yaml). "
             "Used as the workspace label in the rendered goal too.",
    )
    ap.add_argument(
        "--task-yaml", help="Explicit task.yaml path. Wins over <workspace>/task.yaml.",
    )
    ap.add_argument(
        "--label", help="Override the workspace label used inside the goal text. "
                        "Defaults to the basename of <workspace>.",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of one-line text.")
    args = ap.parse_args()

    if not args.workspace and not args.task_yaml:
        ap.error("provide <workspace> or --task-yaml")

    if args.task_yaml:
        task_path = Path(args.task_yaml)
    else:
        task_path = Path(args.workspace) / "task.yaml"

    if not task_path.exists():
        print(f"ERROR: task.yaml not found: {task_path}", file=sys.stderr)
        return 2

    try:
        spec = load_task_spec(task_path)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.workspace:
        label = args.label or Path(args.workspace).name
    else:
        label = args.label or "<workspace>"

    if args.json:
        out = render_goal_json(spec, label)
        print(json.dumps(out, indent=2))
    else:
        print(render_goal(spec, label))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
