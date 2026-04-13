# ferret

Autonomous CUDA kernel optimization agent. Structured task specs, per-config
scoring, two-stage REPRODUCE→OPTIMIZE workflow, git-tagged version tracking.

## Lineage

ferret is the clean rewrite of the `cuda_agent_v3` experiment at
`lithos-cuda-example/examples/cuda_agent_v3/`. v3 produced real wins on
DeepSeek V3 MLA kernels (prefill, decode, multi-token decode) but accumulated
technical debt around multi-config scoring, agent-generated spec files, and
forgotten constraints in long sessions. ferret fixes those at the architecture
level:

- **Structured task.yaml**, authored by the user, is the single source of truth
  (problem, shapes, per-config baselines, constraints, hints, budget).
- **Per-config scoring** — `min_ratio` / `weighted_avg` / `focus`. No more
  `max()` across configs masquerading as "best TFLOPS".
- **Constraints re-injected every iteration**, not just at iteration 0.
- **Stage gate + budget driven by the spec**, not hardcoded constants.
- **`workspace/task.yaml` is read-only** — agent cannot clobber its own spec.

The v3 directory stays frozen as a reference.

## Requirements

- Python **3.12+** (motus dependency)
- NVIDIA CUDA toolkit on the machine where kernels are compiled/benchmarked
- An Anthropic API key in the environment (`ANTHROPIC_API_KEY`)

## Install

```bash
pip install lithosai-motus pyyaml
```

(ferret is not yet pip-installable itself — run in place from the parent dir.
A `src/ferret/` restructure is future work.)

## Usage

```bash
# From the parent of ferret/
cd ~/repos
python -m ferret.main ferret/tasks/mla-mtp-decode-q1to4-kv4096.yaml
```

ferret reads `workspace/task.yaml` (copying from the supplied path if needed),
picks up the latest tagged kernel if `workspace/.git` has tags, and runs the
structured two-stage loop. There is no `--resume` flag — resume is implicit
from git history.

### Authoring a new task

Copy `tasks/template.yaml`, fill in your problem description, shapes,
per-config baselines, constraints, and hints. Validate with:

```bash
python ferret/task_spec.py path/to/your_task.yaml
```

Then point ferret at it:

```bash
python -m ferret.main path/to/your_task.yaml
```

### Inspecting the current state

```bash
# Latest tag's per-config TFLOPS + spec-driven RunState
python -m ferret.state workspace/ tasks/your_task.yaml
```

## Layout

```
ferret/
├── main.py              entry point — validates spec + launches orchestrator
├── orchestrator.py      main loop, stage gate, prompt rendering
├── agents.py            ReAct agent + tool bindings
├── prompts.py           OPTIMIZER_PROMPT (system prompt)
├── task_spec.py         spec schema + loader + scoring + result parser
├── state.py             RunState + compute_state (bridge git → decision)
├── cost_tracker.py      API cost telemetry
├── pick_gpu.sh          multi-GPU picker for shared machines
├── tasks/               authored task.yaml specs (template + examples)
├── tools/               agent tool implementations (compiler, tester, …)
├── docs/                reference material the agent reads (PTX ISA, CUDA guide)
└── examples/            saved best kernels from prior runs
```

Deliberately missing: `resources/`. The vendored library sources (CUTLASS,
FlashInfer, FlashMLA, ThunderKittens, TRT-LLM, Triton) are gitignored and will
become git submodules. For now, copy them locally into `ferret/resources/`
before running.

## Submodule maintenance

Before committing after any submodule change (bump, rename, add, remove):

```bash
python scripts/check_resource_refs.py           # must exit 0
python scripts/check_resource_refs.py --verbose # shows per-submodule ref counts
```

The script walks every `*.md` and `*.py` under the repo and verifies that each
`resources/<submodule>/<path>` reference resolves to an existing file in the
pinned submodule. It also detects drift between `.gitmodules` and the on-disk
`resources/` tree (submodules declared but not cloned, orphan directories,
renamed without updating `.gitmodules`). Exit code 0 when clean, 1 on any
issue. Zero dependencies — stdlib only.

## How the run loop works

1. `main.py` loads + validates `task.yaml`, checks the baseline source path
   exists, logs the configs + scoring policy + budget.
2. Orchestrator `__init__` copies `self.spec`, sets up the ReAct agent.
3. `_first_turn` sends the structured first prompt: task description, shapes,
   baseline reference, per-config state (from `compute_state`), constraints,
   hints (once, then forgotten), and git history (if resuming).
4. Loop iterations (`_iterate`) each send: per-config status table with
   ← WORST marker, re-injected constraints, stage-specific advice, git footer.
5. After each iteration, agent outputs are parsed for `KERNEL_RESULT` lines
   (or `Q_LEN=N: X TFLOPS` legacy), aggregated per `spec.scoring`, compared
   against `spec.stage_gate.ratio` to decide REPRODUCE→OPTIMIZE transitions.
6. Budget exits: iterations, wall time, and tokens — all from `spec.budget`.
7. Agent saves every attempt as a git commit; only wins get tags. Revert to
   last tag after a failure.

## Current working spec

`tasks/mla-mtp-decode-q1to4-kv4096.yaml` — DeepSeek V3 MLA multi-token decode
on B200, Q_LEN ∈ {1,2,4}, KV=4096. Current best (imported from v4 mirror of v3's
v004): Q1=31.1 (91% of trtllm-gen baseline 34.3), Q2=54.7 (87%), Q4=87.7 (109%,
already beating baseline). Target is 100% across all three.

## Status

Under active development. Structured spec / orchestrator / prompts / CLI are
complete. End-to-end run on remote not yet attempted. See git log for per-step
commits — each commit is one focused change.
