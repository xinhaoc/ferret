# ferret

**Autonomous CUDA-kernel agent. Give it a problem, get a kernel that beats the vendor library.**

![Paged GQA speedup over FlashInfer](docs/assets/paged_gqa_vs_flashinfer.png)

Above: an attention kernel produced by ferret on a fresh `task.yaml` —
**1.21× to 3.56× faster than FlashInfer** across all 16 (Q × seq) configurations
of Qwen3-30B-A3B (B200, bf16). 22 iterations, 1 self-contained binary, all 16
correctness-passing. Numbers measured against FlashInfer 0.6.8 using identical
2-sec warmup + 128 MB L2 flush + 300-iter median harness in both kernel and
baseline.

## What ferret does

ferret reads a YAML task spec (problem shape, baseline, target ratios) and runs
a structured **REPRODUCE → OPTIMIZE** loop with a Claude agent under the hood.
Each iteration the agent edits CUDA, ferret compiles + benchmarks + checks
correctness against an in-binary fp32 reference, then scores against the
baseline. Wins get git-tagged; regressions get reverted.

## How ferret is invoked (the dispatch model)

In the production flow ferret is not run by hand — **it is dispatched as a
separate Claude Code session, on demand, from Mirage.**

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

> **`api/` is the older path, not this one.** The directory `api/` preserves the
> original programmatic / `motus`-API invocation form (`python -m ferret.api.main`,
> incl. the `--remote-host` ssh+rsync routing) for reference — the Claude-Code
> agent **ignores** it. See `api/README.md`.

## Verified wins

Three representative workloads, each measured on B200 (one GPU, same physical
session), saved under `examples/`:

| Workload | Baseline | Speedup | Artifact |
|---|---|---|---|
| **Paged GQA decode** (Qwen3-30B-A3B, 16 configs Q×seq) | FlashInfer 0.6.8 | **1.21× – 3.56×** (geomean 1.93×) | [`paged-gqa-fused-qwen3/v011_fused_all16_115x.cu`](examples/paged-gqa-fused-qwen3/v011_fused_all16_115x.cu) |
| **FP8 MLA decode** (DeepSeek-V4 MODEL1, b=2, h_q=64) | FlashMLA | **1.14×** | [`fp8-mla-decode-dsv4/v033_partial_match_785.cu`](examples/fp8-mla-decode-dsv4/v033_partial_match_785.cu) |
| **Decode linear projections** (Qwen3-8B, M=16, GateUp) | cuBLAS BF16 | **1.17×** | [`qwen3-8b-decode-linear-bs16/v019_swapab_cg2_l2hints.cu`](examples/qwen3-8b-decode-linear-bs16/v019_swapab_cg2_l2hints.cu) |

All wins are reproducible from the saved `.cu` — each binary self-contains its
benchmark harness, fp32 correctness check, and (where applicable) the baseline
measurement.

## Requirements

- Python **3.12+**
- NVIDIA CUDA toolkit (B200 = Blackwell sm_100a on the dev cluster)
- `claude` CLI (Claude Code) on `PATH`, signed in with a subscription
- `codex` CLI on `PATH` (used by the `codex-dispatcher` subagent for ABI
  verification against Mirage headers)

## Install

```bash
pip install pyyaml         # the Claude-Code path has zero runtime deps beyond stdlib + pyyaml
git submodule update --init resources/kernelwiki && bash scripts/update_kernelwiki.sh
```

ferret runs in place — no pip install of the package itself. `cc-run.sh` exports
`PYTHONPATH=$(dirname FERRET_DIR)` so subagents can `python -m
ferret.{state,profile,task_spec,cc_goal}` from any cwd. The `resources/` library
sources + `resources/kernelwiki` are git submodules — run the submodule-update +
`update_kernelwiki.sh` once after cloning (KernelWiki content is NOT vendored,
only the upstream pointer is tracked). *(The `api/` path additionally needs
`lithosai-motus` + `ANTHROPIC_API_KEY` — see `api/README.md`.)*

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

Design choices that matter: **structured `task.yaml`** is the single source of
truth (the agent can't rewrite its own spec — `workspace/task.yaml` is
read-only); **per-config scoring** (`min_ratio` / `weighted_avg` / `focus`, no
`max()` masquerading as "best TFLOPS"); **constraints re-injected every
iteration**; **stage gate + budget driven by the spec**.

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
├── docs/                dev-memory(-seed) + architecture/patterns/ptx-isa refs + assets
├── resources/           git-submodule library sources (CUTLASS, FlashInfer,
│                        … + kernelwiki, upstream-tracked, run the update script)
├── scripts/             cc-init.sh / cc-run.sh / update_kernelwiki.sh / check_resource_refs.py
├── .claude/agents/      planner · iterator · profiler · reviewer ·
│                        codex-dispatcher · memory-keeper · kernel-extractor
├── api/                 PRESERVED API-form (motus) path — agent IGNORES it (see api/README.md)
└── workspace<N>/        (gitignored) per-run kernel.cu + kernel.cuh + progress.md + own .git
```

## Submodule maintenance

```bash
python3 scripts/check_resource_refs.py           # must exit 0 (verifies resources/<sub>/<path> refs)
python3 scripts/check_resource_refs.py --verbose # per-submodule ref counts
```

## Lineage

ferret v0.2 migrated v0.1's `motus` / Anthropic-API path
(`orchestrator.py` + `agents.py` + `main.py` + `prompts.py` + `cost_tracker.py`)
onto the Claude-Code subscription + subagents design. That v0.1 API form (incl.
the `--remote-host` ssh+rsync routing) is **preserved under `api/`** (not active;
the agent ignores it). The surviving shared core is the **state CLI + task spec
loader + scoring + ncu wrapper + profile CLI** at the repo root.

## See also

- [`ferret-kernel-agent` subagent](https://github.com/mirage-project/mirage/blob/mpk/.claude/agents/ferret-kernel-agent.md)
  — the Mirage-side dispatcher (how Mirage invokes ferret).
- `tasks/template.yaml` — full task.yaml schema with annotations.
- `CLAUDE.md` — mainthread contract; §6.5 is the loop discipline.
- `api/README.md` — the preserved API-form path (reference only).
```
