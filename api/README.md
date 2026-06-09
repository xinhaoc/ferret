# ferret — API-form (motus) path  *(preserved original README)*

> **This is the ORIGINAL ferret README**, preserved here because `api/` holds the
> original API-form / `motus` invocation path. It is kept for reference only —
> **the Claude-Code agent ignores `api/`** (see the top-level `README.md` and
> `CLAUDE.md`). The only changes from the original: launch is now
> `python -m ferret.api.main` (the files moved under `api/`), and the layout
> below reflects the `api/` subpackage.

Autonomous CUDA kernel optimization agent. Structured task specs, per-config
scoring, two-stage REPRODUCE→OPTIMIZE workflow, git-tagged version tracking.

## Lineage

ferret is the clean rewrite of the `cuda_agent_v3` experiment. v3 produced real
wins on DeepSeek V3 MLA kernels (prefill, decode, multi-token decode) but
accumulated technical debt around multi-config scoring, agent-generated spec
files, and forgotten constraints in long sessions. ferret fixes those at the
architecture level:

- **Structured task.yaml**, authored by the user, is the single source of truth
  (problem, shapes, per-config baselines, constraints, hints, budget).
- **Per-config scoring** — `min_ratio` / `weighted_avg` / `focus`. No more
  `max()` across configs masquerading as "best TFLOPS".
- **Constraints re-injected every iteration**, not just at iteration 0.
- **Stage gate + budget driven by the spec**, not hardcoded constants.
- **`workspace/task.yaml` is read-only** — agent cannot clobber its own spec.

(The current production path replaced this `motus`/Anthropic-API form with the
Claude-Code mainthread + subagents — see the top-level README.)

## Requirements
- Python **3.12+** (`motus` dependency)
- NVIDIA CUDA toolkit on the machine where kernels are compiled/benchmarked
- An Anthropic API key in the environment (`ANTHROPIC_API_KEY`)

## Install
```bash
pip install lithosai-motus pyyaml
```
(run in place from the parent dir of the ferret repo)

## Usage
```bash
# From the PARENT of the ferret dir:
python -m ferret.api.main path/to/task.yaml
# or the launcher (resets the workspace, then launches):
api/scripts/run.sh path/to/task.yaml [--max-iterations N] [--no-detach]
```
ferret reads `workspace/task.yaml`, picks up the latest tagged kernel if
`workspace/.git` has tags, and runs the structured two-stage loop. Resume is
implicit from git history (no `--resume` flag).

Inspect state:
```bash
python -m ferret.state workspace/ tasks/your_task.yaml   # state.py stays at the ferret root
```

## Layout (under `api/`)
```
api/
├── main.py          entry point — validates spec + launches orchestrator
├── orchestrator.py  main loop, stage gate, prompt rendering
├── agents.py        ReAct (motus) agent + tool bindings
├── prompts.py       OPTIMIZER_PROMPT (system prompt)
├── cost_tracker.py  API cost telemetry
├── tools/
│   ├── compiler.py  nvcc compile wrapper
│   └── doc_loader.py reference reader
└── scripts/run.sh   workspace-reset + launch wrapper
```
Shared root modules (`task_spec.py`, `state.py`, `profile.py`,
`tools/profiler.py`) stay at the ferret root and are reused by the Claude-Code
path; `api/` imports them via `..`.

## How the run loop works
1. `main.py` loads + validates `task.yaml`, checks the baseline source path
   exists, logs the configs + scoring policy + budget.
2. Orchestrator `__init__` copies `self.spec`, sets up the ReAct agent.
3. `_first_turn` sends the structured first prompt: task description, shapes,
   baseline reference, per-config state, constraints, hints, git history.
4. Loop iterations each send: per-config status table with ← WORST marker,
   re-injected constraints, stage-specific advice, git footer.
5. Agent outputs are parsed for `KERNEL_RESULT` lines, aggregated per
   `spec.scoring`, compared against `spec.stage_gate.ratio` to decide
   REPRODUCE→OPTIMIZE transitions.
6. Budget exits: iterations, wall time, and tokens — all from `spec.budget`.
7. Every attempt is a git commit; only wins get tags; revert to last tag on fail.
