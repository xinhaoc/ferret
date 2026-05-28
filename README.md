# ferret

Autonomous CUDA kernel optimization agent on top of **Claude Code**.
Structured task specs, per-config scoring, two-stage REPRODUCE→OPTIMIZE
workflow, git-tagged version tracking, six scoped subagents driving
planner / iterator / profiler / reviewer / codex-dispatcher / memory-keeper /
kernel-extractor responsibilities.

A ferret run is one `claude` process per workspace. The mainthread reads
`CLAUDE.md`, follows the loop discipline (§6.5), and delegates specialized
work to the subagents under `.claude/agents/`. Mirage consumes the final
`kernel.cuh` produced by `kernel-extractor` at convergence.

## Lineage

ferret v0.2 is the migration of v0.1's motus / Anthropic-API path
(`orchestrator.py` + `agents.py` + `main.py` + `prompts.py` +
`cost_tracker.py` — all removed in this cut) onto Claude Code subscription.
v0.1 lives in git history; the working subset that survives is the
**state CLI + task spec loader + scoring + ncu wrapper + profile CLI**.

## Requirements

- Python **3.12+**
- NVIDIA CUDA toolkit (B200 = Blackwell sm_100a on the dev cluster)
- `claude` CLI (Claude Code) on `PATH`, signed in with a subscription
- `codex` CLI on `PATH` (used by the `codex-dispatcher` subagent for ABI
  verification against Mirage headers)

## Install

```bash
pip install pyyaml         # ferret has zero runtime deps beyond stdlib + pyyaml
```

ferret runs in place — no pip install of the package itself. The launcher
exports `PYTHONPATH=$(dirname FERRET_DIR)` so subagents can `python -m
ferret.{state,profile,task_spec,cc_goal}` from any cwd.

## Usage

```bash
# Init + launch a workspace (creates ~/ferret/workspace3/ with its own .git,
# copies task.yaml, picks a GPU, exports env, exec claude with the
# /goal + --append-system-prompt channels wired up).
bash scripts/cc-run.sh 3 tasks/mla-mtp-decode-q1to4-kv4096.yaml
```

The mainthread reads `CLAUDE.md`, runs the session-start checklist, and
either dispatches to `planner` (cold start) or to `iterator` (resume).
It keeps iterating until `python3 -m ferret.state` reports `advance? True`
AND every config has the ✓ marker — at which point the `reviewer` invokes
`kernel-extractor` to emit a Mirage-ready `kernel.cuh`. The Mirage-side
dispatcher subagent (`~/mirage/.claude/agents/ferret-kernel-agent.md`)
picks up `workspace<N>/kernel.cuh` and `cp`'s it into the Mirage tree.

### Authoring a new task

Copy `tasks/template.yaml`, fill in problem description, shapes, baseline
SOTA name (just a label — the kernel measures it live), per-config target
ratios, constraints, and hints. Validate with:

```bash
PYTHONPATH=/home/$USER python3 task_spec.py tasks/your_task.yaml
```

The Mirage-side dispatcher can author this for you automatically — see
`~/mirage/.claude/agents/ferret-kernel-agent.md` (steps 1–4).

### Inspecting the current state

```bash
# Latest tag's per-config TFLOPS + spec-driven RunState
PYTHONPATH=/home/$USER python3 -m ferret.state workspace<N>/ workspace<N>/task.yaml

# Concrete goal text rendered from a task.yaml
PYTHONPATH=/home/$USER python3 -m ferret.cc_goal workspace<N>/

# One-shot ncu profile (in OPTIMIZE stage, after compile)
PYTHONPATH=/home/$USER python3 -m ferret.profile workspace<N>/
```

## Layout

```
ferret/
├── CLAUDE.md            mainthread system prompt — what to do in a session
├── cc_goal.py           task.yaml → concrete /goal text (SOTA + targets)
├── profile.py           ncu wrapper CLI (used by profiler subagent)
├── state.py             RunState + compute_state (git → decision)
├── task_spec.py         spec schema + loader + scoring + result parser
├── pick_gpu.sh          multi-GPU picker for shared machines
├── tools/               leaf helpers (ncu CSV parsing only)
├── tasks/               authored task.yaml specs (+ template.yaml)
├── baselines/           reference baseline.py scripts the kernel re-runs
├── examples/            saved best kernels from prior runs
├── docs/
│   ├── dev-memory/      gitignored, populated at runtime by memory-keeper
│   ├── dev-memory-seed/ committed template (cc-init bootstraps from here)
│   ├── architecture/    GPU-level limits (B200, etc.)
│   ├── patterns/        optimization techniques (swapab, split-K, chunked)
│   ├── ptx-isa-9.2/     instruction semantics
│   └── MAPPING.md       topic → reference file table
├── resources/           git-submodule library sources (CUTLASS, FlashInfer …)
├── scripts/
│   ├── cc-init.sh       create workspace<N>/ skeleton with own .git
│   ├── cc-run.sh        launch claude bound to workspace<N> + goal + env
│   └── check_resource_refs.py   submodule path sanity-check
├── .claude/agents/
│   ├── planner.md
│   ├── iterator.md
│   ├── profiler.md
│   ├── reviewer.md
│   ├── codex-dispatcher.md
│   ├── memory-keeper.md
│   └── kernel-extractor.md
└── workspace<N>/        (gitignored) per-run kernel.cu + kernel.cuh +
                         progress.md + own .git
```

## How the loop works (one workspace)

1. `cc-init.sh <N> <task.yaml>` creates `workspace<N>/` with own `.git`,
   copies task.yaml, writes a `progress.md` skeleton.
2. `cc-run.sh <N>` picks a GPU, exports `FERRET_WORKSPACE`/`PYTHONPATH`/
   `TMPDIR`, renders the goal from task.yaml via `ferret.cc_goal`, and
   `exec`s `claude` with both `--append-system-prompt "STANDING GOAL: …"`
   and a leading `/goal …` slash command (the latter installs a
   session-scoped Stop hook so the mainthread cannot exit before goal-
   met).
3. Mainthread runs the §0 session-start checklist, decides cold-start
   vs resume, dispatches to `planner` or `iterator`.
4. Each iteration: write/edit kernel.cu → nvcc → `./kernel` → observe
   `KERNEL_RESULT` / `KERNEL_RESULT_REFERENCE` lines → `git add … &&
   commit -m "v###: …" && git tag v###`.
5. After every tag, dispatch `reviewer`. Reviewer runs 4 checks (API
   alignment via `codex-dispatcher`, output keys, constraints, iterator
   follow-through), optionally escalates new host facts to
   `memory-keeper`, and at convergence dispatches `kernel-extractor`
   which writes `workspace<N>/kernel.cuh` (Mirage-ready, no host code).
6. Goal met → reviewer's last block records `convergence:` →
   mainthread exits cleanly.

## Submodule maintenance

```bash
python3 scripts/check_resource_refs.py           # must exit 0
python3 scripts/check_resource_refs.py --verbose # per-submodule ref counts
```

The script walks `*.md` / `*.py` and verifies every `resources/<sub>/<path>`
reference resolves in the pinned submodule. Catches drift between
`.gitmodules` and the on-disk tree.

## Status

`cc` branch contains the Claude-Code path. Smoke-tested end-to-end
(planner → 3× write→compile→run→tag→reviewer→codex-dispatcher,
convergence-triggered kernel-extractor producing `kernel.cuh`); see
`workspace1/progress.md` for the validation transcript.

## See also

- `~/mirage/.claude/agents/ferret-kernel-agent.md` — the mirage-side
  dispatcher (how mirage's mainthread invokes ferret).
- `tasks/template.yaml` — full task.yaml schema with annotations.
- `CLAUDE.md` — mainthread contract; §6.5 is the loop discipline.
