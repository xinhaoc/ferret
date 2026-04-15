"""Task spec loader for ferret.

Loads workspace/task.yaml into a structured TaskSpec object with validation.
Standalone — imports nothing from orchestrator. Safe to test in isolation.

Usage:
    from .task_spec import load_task_spec, compute_score, parse_kernel_output

    spec = load_task_spec("workspace/task.yaml")
    results = parse_kernel_output(kernel_stdout)
    score, ratios = compute_score(results, spec)

Standalone test:
    python3 task_spec.py tasks/template.yaml
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ConfigEntry:
    """One (shape, baseline) pair the kernel must satisfy."""
    name: str                       # e.g. "Q1"
    args: dict[str, Any]            # e.g. {"Q_LEN": 1}
    baseline_tflops: float          # the number to compare against
    target_ratio: float = 0.95      # 1.0 = match, 0.95 = within 5%, 1.20 = beat by 20%
    weight: float = 1.0             # only used by scoring=weighted_avg


@dataclass
class BaselineSpec:
    """Where the reference comes from + how to (re)measure it."""
    source: str                     # path to reference impl (must exist)
    command: str = ""               # optional: command to run to measure baseline
    pre_measured: bool = True       # if True, baselines come from configs[].baseline_tflops


@dataclass
class StageGate:
    """When to advance from REPRODUCE to OPTIMIZE."""
    ratio: float = 0.9              # aggregate score must reach this
    strict: bool = False            # if True, every config must hit its own target_ratio


@dataclass
class Budget:
    max_iterations: int = 100
    max_wall_minutes: int = 120
    # max_tokens removed: it conflated cumulative billing with per-call context.
    # Per-call context is now handled in orchestrator via
    # agent.context_window_usage at a 500K reset threshold.


@dataclass
class TaskSpec:
    name: str                       # used for tags, log paths, example dir name
    gpu: str                        # B200 | H100 | A100
    arch: str                       # nvcc -gencode arch (e.g. sm_100a)
    precision: str                  # BF16 | FP16 | FP8 | FP32
    description: str                # free-text problem description (agent reads this)
    shapes: dict[str, Any]          # machine-checkable shape facts
    baseline: BaselineSpec
    configs: list[ConfigEntry]
    scoring: str = "min_ratio"      # min_ratio | weighted_avg | focus
    focus_config: str = ""          # only used when scoring == "focus"
    stage_gate: StageGate = field(default_factory=StageGate)
    constraints: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    result_format: str = "kernel_result_json"
    result_keys: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Loader + validation
# ─────────────────────────────────────────────────────────────────────────────


_VALID_SCORING = ("min_ratio", "weighted_avg", "focus")


def load_task_spec(path: str | Path) -> TaskSpec:
    """Load + validate task.yaml. Raises ValueError with a clear message on bad input.

    The validation is intentionally strict — the agent burns expensive API tokens,
    so we want to catch bad specs at startup, not at iteration 30.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"task spec not found: {path}")

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"task spec is not valid YAML: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"task spec must be a YAML mapping, got {type(data).__name__}")

    # Required top-level fields
    for key in ("name", "gpu", "arch", "precision", "problem", "baseline", "configs"):
        if key not in data:
            raise ValueError(f"task spec missing required field: {key}")

    # problem block
    problem = data["problem"]
    if not isinstance(problem, dict):
        raise ValueError("task spec.problem must be a mapping")
    if "description" not in problem:
        raise ValueError("task spec.problem missing required field: description")
    if "shapes" not in problem:
        raise ValueError("task spec.problem missing required field: shapes")
    if not isinstance(problem["shapes"], dict):
        raise ValueError("task spec.problem.shapes must be a mapping")

    # baseline block
    baseline_data = data["baseline"]
    if not isinstance(baseline_data, dict):
        raise ValueError("task spec.baseline must be a mapping")
    if "source" not in baseline_data:
        raise ValueError("task spec.baseline missing required field: source")
    baseline = BaselineSpec(
        source=str(baseline_data["source"]),
        command=str(baseline_data.get("command", "")),
        pre_measured=bool(baseline_data.get("pre_measured", True)),
    )

    # configs
    configs_data = data["configs"]
    if not isinstance(configs_data, list) or not configs_data:
        raise ValueError("task spec.configs must be a non-empty list")

    configs: list[ConfigEntry] = []
    seen_names: set[str] = set()
    for i, c in enumerate(configs_data):
        if not isinstance(c, dict):
            raise ValueError(f"config[{i}] must be a mapping")
        for req in ("name", "args"):
            if req not in c:
                raise ValueError(f"config[{i}] missing required field: {req}")
        name = str(c["name"])
        if name in seen_names:
            raise ValueError(f"duplicate config name: {name}")
        seen_names.add(name)
        if not isinstance(c["args"], dict):
            raise ValueError(f"config[{i}] ({name}).args must be a mapping")
        if baseline.pre_measured and "baseline_tflops" not in c:
            raise ValueError(
                f"config[{i}] ({name}) missing baseline_tflops "
                f"(required when baseline.pre_measured is true)"
            )
        try:
            baseline_tflops = float(c.get("baseline_tflops", 0.0))
        except (TypeError, ValueError) as e:
            raise ValueError(f"config[{i}] ({name}).baseline_tflops must be a number") from e
        try:
            target_ratio = float(c.get("target_ratio", 0.95))
        except (TypeError, ValueError) as e:
            raise ValueError(f"config[{i}] ({name}).target_ratio must be a number") from e
        try:
            weight = float(c.get("weight", 1.0))
        except (TypeError, ValueError) as e:
            raise ValueError(f"config[{i}] ({name}).weight must be a number") from e
        if baseline_tflops < 0:
            raise ValueError(f"config[{i}] ({name}).baseline_tflops must be >= 0")
        if target_ratio <= 0:
            raise ValueError(f"config[{i}] ({name}).target_ratio must be > 0")
        if weight < 0:
            raise ValueError(f"config[{i}] ({name}).weight must be >= 0")
        configs.append(ConfigEntry(
            name=name,
            args=c["args"],
            baseline_tflops=baseline_tflops,
            target_ratio=target_ratio,
            weight=weight,
        ))

    # scoring policy
    scoring = str(data.get("scoring", "min_ratio"))
    if scoring not in _VALID_SCORING:
        raise ValueError(
            f"invalid scoring policy: {scoring!r}. Must be one of {_VALID_SCORING}"
        )
    focus_config = str(data.get("focus_config", ""))
    if scoring == "focus":
        if not focus_config:
            raise ValueError("scoring=focus requires focus_config to be set")
        if not any(c.name == focus_config for c in configs):
            raise ValueError(
                f"focus_config {focus_config!r} not in configs list "
                f"(available: {[c.name for c in configs]})"
            )
    if scoring == "weighted_avg" and sum(c.weight for c in configs) <= 0:
        raise ValueError("scoring=weighted_avg requires at least one config with weight > 0")

    # stage gate
    sg_data = data.get("stage_gate", {}) or {}
    if not isinstance(sg_data, dict):
        raise ValueError("task spec.stage_gate must be a mapping")
    try:
        sg_ratio = float(sg_data.get("ratio", 0.9))
    except (TypeError, ValueError) as e:
        raise ValueError("task spec.stage_gate.ratio must be a number") from e
    if sg_ratio <= 0:
        raise ValueError("task spec.stage_gate.ratio must be > 0")
    stage_gate = StageGate(ratio=sg_ratio, strict=bool(sg_data.get("strict", False)))

    # budget
    bg_data = data.get("budget", {}) or {}
    if not isinstance(bg_data, dict):
        raise ValueError("task spec.budget must be a mapping")
    try:
        budget = Budget(
            max_iterations=int(bg_data.get("max_iterations", 100)),
            max_wall_minutes=int(bg_data.get("max_wall_minutes", 120)),
        )
        # max_tokens silently ignored if present in old yaml — see Budget docstring
    except (TypeError, ValueError) as e:
        raise ValueError(f"task spec.budget contains invalid number: {e}") from e

    # output / result format
    output_data = data.get("output", {}) or {}
    if not isinstance(output_data, dict):
        raise ValueError("task spec.output must be a mapping")
    result_format = str(output_data.get("result_format", "kernel_result_json"))
    result_keys = list(output_data.get("result_keys", [c.name for c in configs]))

    # constraints / hints
    constraints = list(data.get("constraints", []) or [])
    hints = list(data.get("hints", []) or [])
    if not all(isinstance(s, str) for s in constraints):
        raise ValueError("task spec.constraints must be a list of strings")
    if not all(isinstance(s, str) for s in hints):
        raise ValueError("task spec.hints must be a list of strings")

    return TaskSpec(
        name=str(data["name"]),
        gpu=str(data["gpu"]),
        arch=str(data["arch"]),
        precision=str(data["precision"]),
        description=str(problem["description"]),
        shapes=dict(problem["shapes"]),
        baseline=baseline,
        configs=configs,
        scoring=scoring,
        focus_config=focus_config,
        stage_gate=stage_gate,
        constraints=constraints,
        hints=hints,
        budget=budget,
        result_format=result_format,
        result_keys=result_keys,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Result parsing — dual-format support (new JSON + current human format)
# ─────────────────────────────────────────────────────────────────────────────


_KERNEL_RESULT_RE = re.compile(r"KERNEL_RESULT\s+(\{[^\n]*\})")
_QLEN_RE = re.compile(r"Q_LEN=(\d+):\s*([\d.]+)\s*TFLOPS")
_QSHORT_RE = re.compile(r"Q(\d+)=([\d.]+)")


def parse_kernel_output(stdout: str) -> dict[str, float]:
    """Parse kernel stdout for per-config TFLOPS results.

    Priority:
      1. KERNEL_RESULT JSON line              (new format)
      2. "Q_LEN=N: <float> TFLOPS"            (current v004/v012 printf)
      3. "Q<N>=<float>" inside a TFLOPS:line   (commit message body format)
      4. {} if nothing parses

    The format-3 fallback ONLY scans lines starting with "TFLOPS:" because
    real commit bodies often contain other Q<N>=<value> patterns on different
    lines (e.g. "Latency_ms: Q1=18.4us") which would otherwise overwrite the
    real TFLOPS values via last-write-wins. Caught against the v004 mirror —
    parser previously returned Q1=18.4 (the latency) instead of Q1=31.1.

    Returns dict keyed by canonical config name (e.g. "Q1", "Q4").
    """
    if not stdout:
        return {}

    # 1. JSON line (most precise; trust it if present and valid)
    m = _KERNEL_RESULT_RE.search(stdout)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict):
                return {str(k): float(v) for k, v in data.items()
                        if isinstance(v, (int, float))}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass  # fall through to regex parsers

    # 2. v004/v012 printf format — anchored on the literal "TFLOPS" suffix,
    # so it can scan the whole stdout safely.
    results: dict[str, float] = {}
    for match in _QLEN_RE.finditer(stdout):
        results[f"Q{match.group(1)}"] = float(match.group(2))
    if results:
        return results

    # 3. Commit message format — restricted to lines that start with "TFLOPS:"
    # to avoid swallowing latency/error/other Q<N>= patterns.
    for line in stdout.split("\n"):
        if line.lstrip().startswith("TFLOPS:"):
            for match in _QSHORT_RE.finditer(line):
                results[f"Q{match.group(1)}"] = float(match.group(2))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Scoring — pure functions, easy to unit-test
# ─────────────────────────────────────────────────────────────────────────────


def compute_score(
    results: dict[str, float],
    spec: TaskSpec,
) -> tuple[float, dict[str, float]]:
    """Compute aggregate score + per-config ratios.

    Returns:
      (aggregate_score, per_config_ratios)

    aggregate_score is the single number that drives stage gating and "is this
    kernel better than the previous one". per_config_ratios is for display in
    iteration prompts so the agent sees which config is the bottleneck.

    A config that's missing from `results` gets ratio 0.0 (worst case) — that
    way a kernel that doesn't even run a config can't accidentally win.
    """
    ratios: dict[str, float] = {}
    for cfg in spec.configs:
        if cfg.baseline_tflops <= 0:
            ratios[cfg.name] = 0.0
            continue
        tflops = results.get(cfg.name, 0.0)
        ratios[cfg.name] = tflops / cfg.baseline_tflops

    if not ratios:
        return 0.0, ratios

    if spec.scoring == "min_ratio":
        score = min(ratios.values())
    elif spec.scoring == "weighted_avg":
        total_w = sum(c.weight for c in spec.configs)
        if total_w <= 0:
            score = 0.0
        else:
            score = sum(c.weight * ratios[c.name] for c in spec.configs) / total_w
    elif spec.scoring == "focus":
        score = ratios.get(spec.focus_config, 0.0)
    else:
        score = 0.0

    return score, ratios


def should_advance_stage(
    score: float,
    ratios: dict[str, float],
    spec: TaskSpec,
) -> bool:
    """Stage gate decision: should we move from REPRODUCE to OPTIMIZE?

    score must reach spec.stage_gate.ratio. If stage_gate.strict is set, every
    config must also independently clear its own target_ratio (so the agent can't
    enter OPTIMIZE on the strength of one config carrying the aggregate).
    """
    if score < spec.stage_gate.ratio:
        return False
    if spec.stage_gate.strict:
        for cfg in spec.configs:
            if ratios.get(cfg.name, 0.0) < cfg.target_ratio:
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI for testing — `python3 task_spec.py path/to/task.yaml`
# ─────────────────────────────────────────────────────────────────────────────


def _format_state(spec: TaskSpec, results: dict[str, float]) -> str:
    score, ratios = compute_score(results, spec)
    lines = [
        f"  scoring     : {spec.scoring}",
        f"  score       : {score:.3f}",
        f"  advance?    : {should_advance_stage(score, ratios, spec)}",
        f"  per-config:",
    ]
    for cfg in spec.configs:
        tflops = results.get(cfg.name, 0.0)
        ratio = ratios.get(cfg.name, 0.0)
        marker = " ✓" if ratio >= cfg.target_ratio else ""
        lines.append(
            f"    {cfg.name}: {tflops:6.1f} / {cfg.baseline_tflops:6.1f} "
            f"= {ratio*100:5.1f}% (target {cfg.target_ratio*100:.0f}%){marker}"
        )
    return "\n".join(lines)


def _main() -> int:
    import sys
    if len(sys.argv) < 2:
        print("usage: python3 task_spec.py path/to/task.yaml", file=sys.stderr)
        return 2
    try:
        spec = load_task_spec(sys.argv[1])
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Loaded: {spec.name}")
    print(f"  gpu/arch    : {spec.gpu} / {spec.arch} / {spec.precision}")
    print(f"  configs     : {[c.name for c in spec.configs]}")
    print(f"  baselines   : {[c.baseline_tflops for c in spec.configs]}")
    print(f"  constraints : {len(spec.constraints)}")
    print(f"  hints       : {len(spec.hints)}")
    print(f"  stage_gate  : ratio={spec.stage_gate.ratio} strict={spec.stage_gate.strict}")
    print()

    print("Dry-run scoring:")
    print()
    print("  Case A — every config at exactly its baseline (ratio=1.0):")
    fake = {c.name: c.baseline_tflops for c in spec.configs}
    print(_format_state(spec, fake))
    print()

    print("  Case B — every config at 90% of baseline:")
    fake = {c.name: c.baseline_tflops * 0.9 for c in spec.configs}
    print(_format_state(spec, fake))
    print()

    print("  Case C — first config at 70%, rest at 110% (tests min_ratio vs avg):")
    fake = {}
    for i, c in enumerate(spec.configs):
        fake[c.name] = c.baseline_tflops * (0.70 if i == 0 else 1.10)
    print(_format_state(spec, fake))
    print()

    print("Parser test (KERNEL_RESULT JSON):")
    sample = 'noise\nKERNEL_RESULT {"Q1": 31.0, "Q2": 54.7, "Q4": 87.7}\nmore noise'
    print(f"  -> {parse_kernel_output(sample)}")

    print("Parser test (v004 printf format):")
    sample = "Q_LEN=1: 31.0 TFLOPS, 18.4 us\nQ_LEN=2: 54.7 TFLOPS, 20.9 us"
    print(f"  -> {parse_kernel_output(sample)}")

    print("Parser test (commit message format):")
    sample = "TFLOPS: Q1=31.0, Q2=54.7, Q4=87.7"
    print(f"  -> {parse_kernel_output(sample)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
