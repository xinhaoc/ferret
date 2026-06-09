# ferret

Autonomous CUDA kernel optimization agent on top of **Claude Code**.
Structured task specs, per-config scoring, two-stage REPRODUCE→OPTIMIZE
workflow, git-tagged version tracking, scoped subagents driving
planner / iterator / profiler / reviewer / codex-dispatcher / memory-keeper /
kernel-extractor responsibilities.

## How ferret is invoked (the dispatch model)

ferret is not run by hand in the normal flow — **it is dispatched as a separate
Claude Code session, on demand, from Mirage.**

1. **The whole coding agent runs as its own Claude Code session.** One ferret run
   = one headless `claude` mainthread bound to a single workspace
   (`FERRET_WORKSPACE=workspaceN`), launched by `scripts/cc-run.sh`. That session
   reads `CLAUDE.md`, follows the loop discipline, and drives the scoped subagents
   under `.claude/agents/`. It is a self-contained agent — its own context, its
   own git-tagged kernel history — not a library call inside the caller's process.

2. **You dispatch it via a Mirage subagent.** When Mirage needs a new/faster
   kernel, its mainthread invokes the
   [`ferret-kernel-agent` subagent](https://github.com/mirage-project/mirage/blob/mpk/.claude/agents/ferret-kernel-agent.md).
   That subagent translates the Mirage-side kernel requirement into a ferret
   `task.yaml`, picks a free `workspace[1-8]/`, launches `scripts/cc-run.sh` in
   non-interactive headless mode with the standing goal injected, monitors the
   workspace's git tags + `KERNEL_RESULT` lines until the goal is met or the
   wall-time budget expires, then hands the winning `workspace<N>/kernel.cuh`
   (Mirage-ready, no host code) back into the Mirage tree. So from Mirage's point
   of view, "optimize this kernel" is a single subagent call; ferret is the
   separate Claude-Code session that call spins up.

> **`api/` is NOT this path.** The directory `api/` preserves the older
> programmatic / `motus`-API invocation form (`python -m ferret.api.main`) for
> reference only — the Claude-Code agent **ignores** it. See `api/README.md`.

## Requirements

- Python **3.12+**
- NVIDIA CUDA toolkit (B200 = Blackwell sm_100a on the dev cluster)
- `claude` CLI (Claude Code) on `PATH`, signed in with a subscription
- `codex` CLI on `PATH` (used by the `codex-dispatcher` subagent for ABI
  verification against Mirage headers)

## Install

```bash
pip install pyyaml         # ferret has zero runtime deps beyond stdlib + pyyaml
git submodule update --init resources/kernelwiki && bash scripts/update_kernelwiki.sh
```

ferret runs in place — no pip install of the package itself. `cc-run.sh` exports
`PYTHONPATH=$(dirname FERRET_DIR)` so subagents can `python -m
ferret.{state,profile,task_spec,cc_goal}` from any cwd. The `resources/` library
sources + `resources/kernelwiki` are git submodules — run the submodule-update +
`update_kernelwiki.sh` once after cloning (KernelWiki content is NOT vendored,
only the upstream pointer is tracked).

## Usage

Normally a Mirage dispatch drives this (above). To launch a workspace directly:

```bash
# Init + launch workspace3 (own .git, copies task.yaml, picks a GPU, exports env,
# execs claude with the /goal + --append-system-prompt channels wired up).
bash scripts/cc-run.sh 3 tasks/mla-mtp-decode-q1to4-kv4096.yaml
```

The mainthread reads `CLAUDE.md`, runs the session-start checklist, and either
dispatches to `planner` (cold start) or `iterator` (resume). It keeps iterating
until `python3 -m ferret.state` reports `advance? True` AND every config has the
✓ marker — at which point `reviewer` invokes `kernel-extractor` to emit a
Mirage-ready `kernel.cuh`, which the Mirage-side dispatcher `cp`'s into the tree.

### Authoring a new task

Copy `tasks/template.yaml`, fill in problem description, shapes, baseline SOTA
name (a label — the kernel measures it live), per-config target ratios,
constraints, hints. Validate with:

```bash
PYTHONPATH=/home/$USER python3 task_spec.py tasks/your_task.yaml
```

The Mirage-side dispatcher authors this for you automatically (steps 1–4 of the
`ferret-kernel-agent` subagent).

### Inspecting the current state

```bash
PYTHONPATH=/home/$USER python3 -m ferret.state  workspace<N>/ workspace<N>/task.yaml  # per-config TFLOPS + RunState
PYTHONPATH=/home/$USER python3 -m ferret.cc_goal workspace<N>/                          # rendered /goal text
PYTHONPATH=/home/$USER python3 -m ferret.profile workspace<N>/                          # one-shot ncu profile
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
├── docs/                dev-memory(-seed) + architecture/patterns/ptx-isa refs
├── resources/           git-submodule library sources (CUTLASS, FlashInfer,
│                        … + kernelwiki, upstream-tracked, run the update script)
├── scripts/             cc-init.sh / cc-run.sh / update_kernelwiki.sh / check_resource_refs.py
├── .claude/agents/      planner · iterator · profiler · reviewer ·
│                        codex-dispatcher · memory-keeper · kernel-extractor
├── api/                 PRESERVED API-form (motus) path — agent IGNORES it (see api/README.md)
└── workspace<N>/        (gitignored) per-run kernel.cu + kernel.cuh + progress.md + own .git
```

## How the loop works (one workspace)

1. `cc-init.sh <N> <task.yaml>` creates `workspace<N>/` with its own `.git`,
   copies task.yaml, writes a `progress.md` skeleton.
2. `cc-run.sh <N>` picks a GPU, exports `FERRET_WORKSPACE`/`PYTHONPATH`/`TMPDIR`,
   renders the goal via `ferret.cc_goal`, and `exec`s `claude` with
   `--append-system-prompt "STANDING GOAL: …"` plus a leading `/goal …` slash
   command (installs a session-scoped Stop hook so the mainthread cannot exit
   before goal-met).
3. Mainthread runs the §0 session-start checklist, decides cold-start vs resume,
   dispatches to `planner` or `iterator`.
4. Each iteration: write/edit kernel.cu → nvcc → `./kernel` → observe
   `KERNEL_RESULT` / `KERNEL_RESULT_REFERENCE` → `git commit -m "v###: …" && git tag v###`.
5. After every tag, dispatch `reviewer` (4 checks: ABI alignment via
   `codex-dispatcher`, output keys, constraints, iterator follow-through;
   escalates new host facts to `memory-keeper`; at convergence dispatches
   `kernel-extractor` → `workspace<N>/kernel.cuh`).
6. Goal met → reviewer records `convergence:` → mainthread exits cleanly; the
   Mirage dispatcher collects `kernel.cuh`.

## Submodule maintenance

```bash
python3 scripts/check_resource_refs.py           # must exit 0 (verifies resources/<sub>/<path> refs)
python3 scripts/check_resource_refs.py --verbose # per-submodule ref counts
```

## Lineage

ferret v0.2 migrated v0.1's `motus` / Anthropic-API path
(`orchestrator.py` + `agents.py` + `main.py` + `prompts.py` + `cost_tracker.py`)
onto the Claude-Code subscription + subagents design. That v0.1 API form is
**preserved under `api/`** (not active; the agent ignores it). The surviving
shared core is the **state CLI + task spec loader + scoring + ncu wrapper +
profile CLI** at the repo root.

## See also

- [`ferret-kernel-agent` subagent](https://github.com/mirage-project/mirage/blob/mpk/.claude/agents/ferret-kernel-agent.md)
  — the Mirage-side dispatcher (how Mirage invokes ferret).
- `tasks/template.yaml` — full task.yaml schema with annotations.
- `CLAUDE.md` — mainthread contract; §6.5 is the loop discipline.
- `api/README.md` — the preserved API-form path (reference only).
```
