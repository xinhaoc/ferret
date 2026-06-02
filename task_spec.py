"""Task spec loader for ferret.

Loads workspace/task.yaml into a structured TaskSpec object with validation.
Standalone — imports nothing from orchestrator. Safe to test in isolation.

Usage:
    from .task_spec import (
        load_task_spec, compute_score, parse_kernel_output, parse_reference_output,
    )

    spec = load_task_spec("workspace/task.yaml")
    results = parse_kernel_output(commit_body)              # KERNEL_RESULT
    reference = parse_reference_output(commit_body)         # KERNEL_RESULT_REFERENCE
    score, ratios = compute_score(results, reference, spec) # ratio = kernel/reference

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
class IoTensor:
    """One kernel input/output tensor in MPK's format.

    A machine-readable, MPK-format I/O+shape contract so the dispatcher and the
    validator agree on the exact interface (the user's "明确接口规范" directive).
    shape entries may be symbolic strings ("K/128", "N", "M") or ints.
    """
    name: str                       # e.g. "A", "SFA", "D"
    shape: list[Any]                # e.g. ["M", "K"] or [128, "K/128"]; symbolic dims allowed
    dtype: str                      # fp8_e4m3 | bf16 | fp32 | int32 | ...
    layout: str = ""                # k_major | row_major | n_major | "" (unspecified)
    role: str = ""                  # optional, e.g. "act_scale_1x128", "weight_scale_128x128"
    prezeroed: bool = False         # optional; True = caller pre-zeroes (e.g. atomic-accum output)


@dataclass
class ReduceHint:
    """How the kernel's K-reduction (or any cross-CTA accumulation) should be done.

    A perf HINT for the coding agent, not a hard constraint. `method` is the
    accumulation strategy:
      - internal_atomic : inter-CTA accumulation via red.global atomics into a
                          pre-zeroed output, all inside ONE kernel (no reduce task).
      - tma_reduce      : use TMA reduction (cp.reduce.async.bulk / tcgen05 TMA
                          store-reduce) to accumulate partials in gmem/smem. PREFER
                          this when applicable — it is generally faster than naive
                          red.global atomics for the split-K reduce on Blackwell.
      - external_reduce : emit a SEPARATE reduce task/kernel that sums partials
                          (the builder-side split-K hop ferret usually wants to AVOID).
    """
    method: str = ""                # internal_atomic | tma_reduce | external_reduce | ""
    op: str = ""                    # optional, e.g. "red.global.add.noftz.bf16x2"
    separate_task: bool = False     # True = reduction is a distinct task/kernel (external_reduce)


@dataclass
class ConfigEntry:
    """One (shape, baseline) pair the kernel must satisfy."""
    name: str                       # e.g. "Q1"
    args: dict[str, Any]            # e.g. {"Q_LEN": 1}
    target_ratio: float = 0.95      # 1.0 = match reference, 0.95 = within 5%, 1.10 = beat by 10%
    weight: float = 1.0             # only used by scoring=weighted_avg
    target_latency_us: float | None = None  # optional ABSOLUTE latency goal (μs); None = ratio-only
    # baseline_tflops removed — baselines are now measured at runtime by the
    # agent's benchmark and emitted as KERNEL_RESULT_REFERENCE alongside
    # KERNEL_RESULT. See compute_score / orchestrator render.


@dataclass
class BaselineSpec:
    """The scoring baseline — what the agent's TFLOPS is measured against.

    This is NOT a code-reading hint. It names the fast, fixed comparison
    point (e.g. "cuBLAS", "trtllm-gen") that KERNEL_RESULT_REFERENCE must be
    measured against. Architectural reading material goes in TaskSpec.references.
    """
    source: str                     # name/path of the scoring reference (e.g. "cuBLAS")


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
    io_inputs: list[IoTensor] = field(default_factory=list)   # structured MPK-format input tensor contract (problem.io.inputs)
    io_outputs: list[IoTensor] = field(default_factory=list)  # structured MPK-format output tensor contract (problem.io.outputs)
    reduce: ReduceHint = field(default_factory=ReduceHint)    # K-reduction / cross-CTA accumulation hint (problem.reduce)
    references: list[str] = field(default_factory=list)  # REPRODUCE reading list — architectural templates (e.g. examples/tcgen05-gemm/). Separate from baseline: these are what to read, not what to beat.
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
# Recognized reduce-accumulation strategies (problem.reduce.method). tma_reduce
# is INTENTIONALLY first-class — the agent should be hinted to prefer it for the
# split-K reduce when applicable (generally faster than naive red.global atomics
# on Blackwell). Unknown methods are rejected so a typo doesn't silently drop the
# perf hint.
_VALID_REDUCE_METHODS = ("internal_atomic", "tma_reduce", "external_reduce")


def _parse_io_tensor(d: Any, where: str) -> IoTensor:
    """Parse one problem.io.{inputs,outputs}[] entry into an IoTensor."""
    if not isinstance(d, dict):
        raise ValueError(f"{where} must be a mapping")
    for req in ("name", "shape", "dtype"):
        if req not in d:
            raise ValueError(f"{where} missing required field: {req}")
    shape = d["shape"]
    if not isinstance(shape, list):
        raise ValueError(f"{where}.shape must be a list (ints or symbolic strings like 'K/128')")
    return IoTensor(
        name=str(d["name"]),
        shape=list(shape),
        dtype=str(d["dtype"]),
        layout=str(d.get("layout", "")),
        role=str(d.get("role", "")),
        prezeroed=bool(d.get("prezeroed", False)),
    )


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

    # problem.io — OPTIONAL structured MPK-format I/O contract. Back-compat:
    # absent => empty lists (old yamls keep loading).
    io_inputs: list[IoTensor] = []
    io_outputs: list[IoTensor] = []
    io_data = problem.get("io", {}) or {}
    if not isinstance(io_data, dict):
        raise ValueError("task spec.problem.io must be a mapping")
    inputs_data = io_data.get("inputs", []) or []
    outputs_data = io_data.get("outputs", []) or []
    if not isinstance(inputs_data, list):
        raise ValueError("task spec.problem.io.inputs must be a list")
    if not isinstance(outputs_data, list):
        raise ValueError("task spec.problem.io.outputs must be a list")
    for j, t in enumerate(inputs_data):
        io_inputs.append(_parse_io_tensor(t, f"problem.io.inputs[{j}]"))
    for j, t in enumerate(outputs_data):
        io_outputs.append(_parse_io_tensor(t, f"problem.io.outputs[{j}]"))

    # problem.reduce — OPTIONAL K-reduction / cross-CTA accumulation hint.
    # Back-compat: absent => default ReduceHint (empty method).
    reduce_data = problem.get("reduce", {}) or {}
    if not isinstance(reduce_data, dict):
        raise ValueError("task spec.problem.reduce must be a mapping")
    reduce_method = str(reduce_data.get("method", ""))
    if reduce_method and reduce_method not in _VALID_REDUCE_METHODS:
        raise ValueError(
            f"task spec.problem.reduce.method {reduce_method!r} invalid. "
            f"Must be one of {_VALID_REDUCE_METHODS} (or omitted)"
        )
    reduce = ReduceHint(
        method=reduce_method,
        op=str(reduce_data.get("op", "")),
        separate_task=bool(reduce_data.get("separate_task", False)),
    )

    # baseline block
    baseline_data = data["baseline"]
    if not isinstance(baseline_data, dict):
        raise ValueError("task spec.baseline must be a mapping")
    if "source" not in baseline_data:
        raise ValueError("task spec.baseline missing required field: source")
    baseline = BaselineSpec(
        source=str(baseline_data["source"]),
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
        try:
            target_ratio = float(c.get("target_ratio", 0.95))
        except (TypeError, ValueError) as e:
            raise ValueError(f"config[{i}] ({name}).target_ratio must be a number") from e
        try:
            weight = float(c.get("weight", 1.0))
        except (TypeError, ValueError) as e:
            raise ValueError(f"config[{i}] ({name}).weight must be a number") from e
        if target_ratio <= 0:
            raise ValueError(f"config[{i}] ({name}).target_ratio must be > 0")
        if weight < 0:
            raise ValueError(f"config[{i}] ({name}).weight must be >= 0")
        target_latency_us: float | None = None
        if c.get("target_latency_us") is not None:
            try:
                target_latency_us = float(c["target_latency_us"])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"config[{i}] ({name}).target_latency_us must be a number"
                ) from e
            if target_latency_us <= 0:
                raise ValueError(f"config[{i}] ({name}).target_latency_us must be > 0")
        # baseline_tflops silently ignored if present in old yaml — measured at runtime now
        configs.append(ConfigEntry(
            name=name,
            args=c["args"],
            target_ratio=target_ratio,
            weight=weight,
            target_latency_us=target_latency_us,
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

    # references — REPRODUCE reading list, separate from baseline
    references = list(data.get("references", []) or [])
    if not all(isinstance(s, str) for s in references):
        raise ValueError("task spec.references must be a list of strings")

    return TaskSpec(
        name=str(data["name"]),
        gpu=str(data["gpu"]),
        arch=str(data["arch"]),
        precision=str(data["precision"]),
        description=str(problem["description"]),
        shapes=dict(problem["shapes"]),
        baseline=baseline,
        configs=configs,
        io_inputs=io_inputs,
        io_outputs=io_outputs,
        reduce=reduce,
        references=references,
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


_KERNEL_RESULT_RE = re.compile(r"KERNEL_RESULT(?!_REFERENCE)\s+(\{[^\n]*\})")
_KERNEL_RESULT_REFERENCE_RE = re.compile(r"KERNEL_RESULT_REFERENCE\s+(\{[^\n]*\})")
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


_KERNEL_LATENCY_RE = re.compile(r"KERNEL_LATENCY_US\s+(\{[^\n]*\})")


def parse_latency_output(stdout: str) -> dict[str, float]:
    """Parse per-config measured latency (μs) — KERNEL_LATENCY_US JSON line.

    Only used when a config carries an absolute `target_latency_us`. The agent's
    benchmark emits, alongside KERNEL_RESULT / KERNEL_RESULT_REFERENCE:
        KERNEL_LATENCY_US {"<config>": <kernel-latency-us>, ...}
    Returns {} if the line is absent (latency gating then silently no-ops).
    """
    if not stdout:
        return {}
    m = _KERNEL_LATENCY_RE.search(stdout)
    if not m:
        return {}
    try:
        data = json.loads(m.group(1))
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()
                    if isinstance(v, (int, float))}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return {}


def parse_reference_output(stdout: str) -> dict[str, float]:
    """Parse REFERENCE baseline TFLOPS — KERNEL_RESULT_REFERENCE JSON line.

    The agent's benchmark must emit BOTH lines so ferret can score:
        KERNEL_RESULT {"<config>": <kernel-tflops>, ...}
        KERNEL_RESULT_REFERENCE {"<config>": <reference-tflops>, ...}

    Both should be measured on the same GPU with the same harness — fair
    comparison. The reference is whatever the agent picked as the bar
    (cuBLAS, trtllm-gen, etc.). No fallback: if KERNEL_RESULT_REFERENCE is
    missing, returns {} and scoring will report ratio 0 (treated like missing
    measurement).
    """
    if not stdout:
        return {}
    m = _KERNEL_RESULT_REFERENCE_RE.search(stdout)
    if not m:
        return {}
    try:
        data = json.loads(m.group(1))
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()
                    if isinstance(v, (int, float))}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Scoring — pure functions, easy to unit-test
# ─────────────────────────────────────────────────────────────────────────────


def compute_score(
    results: dict[str, float],
    reference: dict[str, float],
    spec: TaskSpec,
    latencies: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute aggregate score + per-config ratios.

    Args:
        results:    own-kernel TFLOPS per config (parse_kernel_output)
        reference:  baseline reference TFLOPS per config (parse_reference_output)
        spec:       task spec
        latencies:  optional own-kernel measured latency (μs) per config
                    (parse_latency_output). Only consulted for configs that set
                    `target_latency_us`. None / missing => latency branch no-ops
                    and scoring is the legacy TFLOPS-ratio behavior.

    Returns:
        (aggregate_score, per_config_ratios)
        ratio[cfg] = results[cfg] / reference[cfg]
        Missing reference (ref==0) → ratio 0 (can't score without reference).

        LATENCY BRANCH: if cfg.target_latency_us is set and a measured latency
        exists, a latency-attainment ratio (target_us / measured_us; >=1.0 means
        the absolute goal is met) is folded in by taking min(tflops_ratio,
        latency_ratio). The config is only "done" once BOTH the relative-TFLOPS
        bar and the absolute-latency goal are satisfied.

    aggregate_score is the single number that drives stage gating and "is this
    kernel better than the previous one". per_config_ratios is for display in
    iteration prompts so the agent sees which config is the bottleneck.
    """
    latencies = latencies or {}
    ratios: dict[str, float] = {}
    for cfg in spec.configs:
        ref = reference.get(cfg.name, 0.0)
        if ref <= 0:
            ratios[cfg.name] = 0.0
            continue
        tflops = results.get(cfg.name, 0.0)
        ratio = tflops / ref
        # Optional absolute-latency branch — only when this config sets a goal
        # AND we have a measurement for it.
        if cfg.target_latency_us is not None:
            meas_us = latencies.get(cfg.name, 0.0)
            if meas_us > 0:
                lat_ratio = cfg.target_latency_us / meas_us  # >=1.0 => goal met
                ratio = min(ratio, lat_ratio)
        ratios[cfg.name] = ratio

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

    The optional absolute-latency goal (config.target_latency_us) is honored
    transparently here: compute_score already folds latency-attainment into each
    config's ratio (min of TFLOPS-ratio and target_us/measured_us), so a config
    whose throughput is fine but whose latency is still above target stays below
    its target_ratio and (under strict) blocks advancing.
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


def _format_state(spec: TaskSpec, results: dict[str, float],
                  reference: dict[str, float]) -> str:
    score, ratios = compute_score(results, reference, spec)
    lines = [
        f"  scoring     : {spec.scoring}",
        f"  score       : {score:.3f}",
        f"  advance?    : {should_advance_stage(score, ratios, spec)}",
        f"  per-config:",
    ]
    for cfg in spec.configs:
        tflops = results.get(cfg.name, 0.0)
        ref = reference.get(cfg.name, 0.0)
        ratio = ratios.get(cfg.name, 0.0)
        marker = " ✓" if ratio >= cfg.target_ratio else ""
        lines.append(
            f"    {cfg.name}: {tflops:6.1f} / {ref:6.1f} "
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
    print(f"  target_ratios: {[c.target_ratio for c in spec.configs]}")
    print(f"  constraints : {len(spec.constraints)}")
    print(f"  hints       : {len(spec.hints)}")
    print(f"  stage_gate  : ratio={spec.stage_gate.ratio} strict={spec.stage_gate.strict}")
    print()

    print("Dry-run scoring (reference values are illustrative — measured at runtime):")
    print()
    fake_ref = {c.name: 100.0 for c in spec.configs}  # arbitrary reference
    print("  Case A — kernel == reference (ratio=1.0):")
    fake_kernel = {c.name: 100.0 for c in spec.configs}
    print(_format_state(spec, fake_kernel, fake_ref))
    print()

    print("  Case B — kernel at 90% of reference:")
    fake_kernel = {c.name: 90.0 for c in spec.configs}
    print(_format_state(spec, fake_kernel, fake_ref))
    print()

    print("  Case C — first config 70%, rest 110% (tests min_ratio):")
    fake_kernel = {c.name: (70.0 if i == 0 else 110.0)
                   for i, c in enumerate(spec.configs)}
    print(_format_state(spec, fake_kernel, fake_ref))
    print()

    print("Parser tests:")
    sample = 'noise\nKERNEL_RESULT {"Q1": 31.0, "Q2": 54.7}\nmore noise'
    print(f"  KERNEL_RESULT JSON          -> {parse_kernel_output(sample)}")
    sample_ref = 'KERNEL_RESULT_REFERENCE {"Q1": 25.0, "Q2": 50.0}'
    print(f"  KERNEL_RESULT_REFERENCE     -> {parse_reference_output(sample_ref)}")
    sample_both = sample + "\n" + sample_ref
    print(f"  Both lines (kernel parser)  -> {parse_kernel_output(sample_both)}")
    print(f"  Both lines (ref parser)     -> {parse_reference_output(sample_both)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
