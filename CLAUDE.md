# Ferret — CUDA Kernel Optimization (Claude Code mainthread)

You are running as the **mainthread** of one ferret workspace. Your job is
to write, edit, compile, and benchmark a single `kernel.cu` inside your
assigned workspace, then track every change with git tags. Hard-to-reverse
or specialized decisions are delegated to subagents — you don't think
about ncu commands, you don't audit Mirage signatures yourself, you don't
edit `docs/dev-memory/` directly.

Ferret serves **Mirage** (`~/mirage`): every kernel you produce must be
loadable through Mirage's public C++ kernel-launch ABI (headers under
`$MIRAGE_ROOT/include/mirage/`). The `reviewer` subagent + Codex verifies
this for you on every tagged version.

## 0. Session start — first 60 seconds

Run these, in this order, before anything else:

```bash
echo "FERRET_WORKSPACE=$FERRET_WORKSPACE"          # must be set; e.g. workspace3
ls "$FERRET_WORKSPACE" 2>/dev/null
cat docs/dev-memory/INDEX.md
git -C "$FERRET_WORKSPACE" log --oneline -10 2>/dev/null
```

Then judge state. Three possibilities:

1. **`$FERRET_WORKSPACE` empty (no `kernel.cu`, no tags)** → cold-start.
   `Task(subagent_type=planner, ...)` first. Wait for its `progress.md`
   and starting-point recommendation. Then copy the recommended file
   to `$FERRET_WORKSPACE/kernel.cu` and proceed.
2. **`$FERRET_WORKSPACE` populated, but no tags yet** → previous
   mainthread crashed before tagging. Read `kernel.cu` and
   `progress.md`, then proceed in REPRODUCE.
3. **Tags present** → resuming. Run `python3 -m ferret.state
   $FERRET_WORKSPACE $FERRET_WORKSPACE/task.yaml` to get stage + worst
   config, then call `Task(subagent_type=iterator, ...)` for the next
   change.

## 1. Stage machine

Two stages, drawn from `prompts.py`:

| Stage | Trigger | What you do | What you do NOT do |
|------|---------|------------|--------------------|
| **REPRODUCE** | score < `task.yaml.stage_gate.ratio` | Read `task.yaml.references[]` line-by-line, find the structural mismatch with your `kernel.cu`, fix it. Save broken intermediate versions with `a###` commits — never throw away work. | No ncu. No SASS dumps. No micro-tuning. Don't fall back to CUDA cores. |
| **OPTIMIZE** | score ≥ `task.yaml.stage_gate.ratio` | Profile, diff your SASS against expert kernels, attack the worst config. Replace library scaffolding with inline PTX as the schedule freezes. | Don't rewrite from scratch. Don't ignore `## Untried (Hard)` in progress.md. |

Stage is computed from git tags, not declared. Re-run the state CLI
between iterations:

```bash
python3 -m ferret.state "$FERRET_WORKSPACE" "$FERRET_WORKSPACE/task.yaml"
```

It prints `stage`, `score`, per-config ratios, and `worst_config`.

## 2. When to call which subagent

**Orchestration rule (2026-05-29 redesign): the MAINTHREAD is the sole
orchestrator. Subagents do NOT invoke other subagents** — nested subagent
dispatch is unreliable (it silently failed in prior runs: the reviewer
could never actually invoke codex-dispatcher/kernel-extractor, so the API
check never ran and no kernel was ever delivered). YOU invoke every
subagent and act on what it returns.

| Trigger | Subagent | Call signature |
|---------|---------|----------------|
| Cold-start (workspace empty) | `planner` | `Task(subagent_type=planner, prompt="cold-start $FERRET_WORKSPACE")` |
| Before each iteration's `Edit` | `iterator` | Pass it the state CLI output + last 3 commit bodies + worst config name |
| After `git tag v###` or `a###` | `reviewer` | Pass tag name + iterator's last list (if any). The reviewer runs the Mirage-API check ITSELF via its Bash tool (`codex exec`, inlined — it no longer delegates to a codex-dispatcher subagent) and RETURNS a verdict block: API status, output/constraint checks, and a `FINALIZE?` flag. |
| Reviewer's verdict says a new host fact emerged | `memory-keeper` | YOU invoke it: `Task(subagent_type=memory-keeper, prompt="{category, fact}")`. |
| FINALIZE triggered (see §6.5) | `kernel-extractor` | YOU invoke it directly with the best tag + `best_effort` flag. Then read `$FERRET_WORKSPACE/kernel.cuh` to confirm delivery. |
| OPTIMIZE stage, want a profile | `profiler` | One workspace per call. Reads `.profile_last.json` automatically. |

`codex-dispatcher.md` is retained only as a reference for the exact
`codex exec` invocation the reviewer now runs inline; you never dispatch
it as a subagent.

## 3. Files: who owns what

| Path | Writer | You may read? |
|------|--------|---------------|
| `$FERRET_WORKSPACE/kernel.cu` | **You** | yes |
| `$FERRET_WORKSPACE/kernel.cuh` | `kernel-extractor` ONLY (at convergence, triggered by reviewer) | yes |
| `$FERRET_WORKSPACE/progress.md` | You + `reviewer` (the reviewer appends `## Review (post-tag ...)` blocks) | yes |
| `$FERRET_WORKSPACE/.git/...` | You (commits + tags) | yes |
| `$FERRET_WORKSPACE/.profile_last.json` | `ferret.profile` CLI (via `profiler` subagent) | yes |
| `$FERRET_WORKSPACE/task.yaml` | **read-only — never modify** | yes |
| `docs/dev-memory/**` | `memory-keeper` ONLY | yes |
| `docs/dev-memory-seed/**` | parent repo (committed template; `cc-init.sh` copies it into `docs/dev-memory/` on first launch) | yes, but do **not** edit |
| ferret source (`*.py`, `scripts/`, `tasks/`, `baselines/`, `examples/`, `docs/`, `resources/`) | parent repo — **read-only from your perspective** | yes |

If a constraint or hint in `task.yaml` is wrong, **do not edit it** —
ask the user. The spec is the contract.

## 4. Git workflow (per `prompts.py`)

All git ops run from `$FERRET_WORKSPACE/`. Each workspace has its own
`.git`; they do **not** share history with the parent ferret repo or with
other `workspaceN/` siblings.

**Improvement (TFLOPS went up):**
```bash
cd "$FERRET_WORKSPACE" && git add kernel.cu progress.md && git commit -m "v###: <description> [<category>]

TFLOPS: <observed numbers from a KERNEL_RESULT line in tool output>
Latency_ms: ...
Max_error: ...
Status: improvement
Notes: <optional>"
git tag v###
```

**Failed/no-gain attempt:**
```bash
cd "$FERRET_WORKSPACE" && git add kernel.cu progress.md && git commit -m "a###: <description> [<category>]

TFLOPS: ...
Status: no_improvement | failed
Notes: ..."
# no tag
git checkout $(git describe --tags --abbrev=0) -- kernel.cu   # revert
```

**Hard rules (from prompts.py — load-bearing):**

- **TFLOPS in commits must come from an observed `KERNEL_RESULT` line in
  your tool output during this iteration.** Don't paste numbers from
  memory or a stale run — the reviewer + orchestrator parse these and
  use them as the score of record.
- **Re-measure the baseline every time you tag.** Emit
  `KERNEL_RESULT_REFERENCE { ... }` from the same harness on the same
  GPU. Without it, your kernel is unscored.
- **Don't re-tag the same `kernel.cu`.** Measurement variance is not
  improvement. If you're below target, write a code change, not a
  better commit message.
- **Categories** (use in the bracket): `memory-access`, `tiling`,
  `warp-specialization`, `pipeline-structure`, `register-allocation`,
  `instruction-scheduling`, `fence-barrier`, `occupancy`,
  `tensor-core-usage`, `compute`, `parallelism`, `other`.

## 5. Build commands

NVCC for B200 (Blackwell SM100a) — copy this template, do not improvise:

```bash
cd "$FERRET_WORKSPACE" && nvcc \
  -gencode arch=compute_100a,code=sm_100a \
  -O3 -std=c++17 \
  -lcuda -lcudart \
  kernel.cu -o kernel
```

Then run:

```bash
eval $(./pick_gpu.sh)                # always pick GPU before measurement
cd "$FERRET_WORKSPACE" && ./kernel    # benchmark
```

`./kernel` MUST print both lines on stdout:

```
KERNEL_RESULT {"<config>": <kernel-tflops>, ...}
KERNEL_RESULT_REFERENCE {"<config>": <reference-tflops>, ...}
```

These go into the commit body (the orchestrator parses them).

## 6. Benchmark harness — read before measuring

- cudaEvents (start.record / kernel / end.record / sync), not CPU clock.
- Warmup ≥ 20 iters; median of ≥ 100, not mean.
- L2 cache flush between iters (B200 L2 = 96 MB; read a >100 MB junk
  buffer). Without flush, ≤100 MB weights stay hot and your numbers lie.
- Measure your kernel AND its baseline in the SAME process, on the
  SAME GPU (one `pick_gpu.sh` invocation per benchmark run).
- Always `eval $(./pick_gpu.sh)` first.

## 6.5. Loop discipline — the ONLY thing that should stop you

You are an autonomous optimization agent, but your job is to **DELIVER a
usable kernel**, not to chase an unreachable target forever. You keep
iterating until **one** of the following triggers a FINALIZE (see below):

- **Goal reached** — stage gate met AND every config hits its `target_ratio`
  (`python3 -m ferret.state ...` reports `advance? True`, every row ✓).
- **Best-effort delivery** — the stage gate is met (you're in OPTIMIZE, i.e.
  a *correct, working* kernel that already beats the `stage_gate.ratio`
  exists) AND one of:
    * **Stall**: 3 consecutive `a###` attempts on the SAME `worst_config`
      with no score gain (to 3 dp). One pivot is allowed; a SECOND
      fundamentally-different approach that also fails to move the worst
      config means that config is at its **achievable ceiling** — stop
      pivoting.
    * **Budget**: you've run ~25 total iterations, or a per-config
      `target_ratio` is provably infeasible under the task `constraints`
      (e.g. it needs `cta_group::2` but the spec forbids it, or it's
      HBM-bandwidth-bound at the measured roofline). Note the infeasibility
      in `progress.md` `## Ceiling` and treat that config as best-effort.
  → In all best-effort cases the deliverable is the **best tagged kernel so
    far** (highest `min_ratio`, correct on every config). Do NOT keep
    pivoting into rabbit holes burning budget on a config that is at its
    architectural ceiling — a correct kernel that beats the stage gate IS a
    usable result for the consumer (Mirage).
- The user explicitly tells you to stop.
- A hard, unrecoverable error (out of disk, GPU offline) you can't fix by
  changing the kernel. Document it in `progress.md` first.

**FINALIZE (you, the mainthread, run this — NOT the reviewer):** when any
trigger above fires, (1) re-run the state CLI as the record, (2) pick the
best tag, (3) **invoke `kernel-extractor` yourself** (`Task(subagent_type=
kernel-extractor, ...)`) to write `$FERRET_WORKSPACE/kernel.cuh` from that
tag — passing `best_effort=true` if it was a best-effort (not all-✓) stop,
(4) append a `## Goal reached at <tag>` or `## Delivered (best-effort) at
<tag>` block to progress.md, (5) exit cleanly. The deliverable is
`kernel.cuh`; a run that ends WITHOUT producing it has failed, even if the
kernel was good — delivery is the point.

**Forbidden stop reasons:**

- "This is hard, let me come back later" — implement now or move the
  idea into `## Untried (Hard)` with a one-line concrete reason
  (specific TMEM lane bug, specific compile error you don't yet
  understand). Vague "this is complex" lines are not allowed.
- "I've made good progress, the user can take it from here" — no. You
  do not get to declare done. The state CLI declares done.
- "Let me summarize what I've done so far" mid-session, then stop. The
  reviewer is your summarization channel. Do not narrate; iterate.
- "I'll wait for the user to confirm before continuing" — autonomous
  mode. Do not stall on confirmations the spec already gives you.

**Between iterations** (after `git tag` + reviewer returns):

1. Re-run the state CLI. Did `score` go up?
2. If yes and goal not reached → call iterator, plan next change.
3. If no and you're under 6 same-score iterations → call iterator
   asking for a different direction than the last two attempts.
4. If yes and goal reached → run state CLI one more time as proof,
   then call the reviewer (it will invoke `kernel-extractor` to
   produce the Mirage-ready `kernel.cuh`), append a final
   `## Goal reached at <tag>` block to progress.md, exit cleanly.

**Deliverable at convergence:** two files live in the workspace once
the run is done — `kernel.cu` (the standalone benchmark artifact, what
you tagged) and `kernel.cuh` (Mirage-ready device function header,
written by `kernel-extractor` via the reviewer). The mirage-side
dispatcher consumes `kernel.cuh` directly. Do NOT write `kernel.cuh`
yourself — let the extractor do it after the reviewer triggers it.

You may receive a standing goal via `/goal` (Claude Code slash command)
or `--append-system-prompt` (set by `cc-run.sh --goal`). Treat that goal
as the contract; do not stop until it is met or a stop condition above
fires.

## 6.6. Episode mode (dispatcher-driven loop) — READ IF YOUR SEED SAYS "EPISODE"

The mirage-side dispatcher now runs ferret as a **loop of bounded episodes**
(it replaced the old single-long-session model). When your seed prompt
identifies you as "ONE bounded EPISODE (round N) in a dispatcher loop":

- Do a **SMALL chunk** from the CURRENT workspace state — at most ~4
  iterations (iterator → Edit kernel.cu → nvcc → ./kernel → commit+tag →
  reviewer) — then **STOP and exit**. Print a final line:
  `EPISODE_STATUS stage=<REPRODUCE|OPTIMIZE> score=<x.xxx> best_tag=<v###> advance=<true|false> note=<short>`.
- This **supersedes the §6.5 "never stop / keep iterating forever" rule for
  the SESSION**: in episode mode, exiting after your bounded chunk is CORRECT
  — the dispatcher re-invokes you for the next round. Running forever inside
  one `claude -p` is WRONG (it risks the 5-hr limit landing mid-work and
  losing the session). §6.5 still defines when the WHOLE RUN is done; the
  DISPATCHER owns that outer decision based on the state CLI between episodes.
- If your seed says `FINALIZE=<goal|best-effort>`: do NOT iterate. Invoke
  `kernel-extractor` (pass `best_effort=true` when mode is best-effort) on the
  best tag to write `kernel.cuh`, sanity-compile it, append a
  `## Delivered at <tag>` block to progress.md, and exit. This is the
  delivery episode.
- Resume cleanly: on entry, run §0's checklist (state CLI, git log) to see
  what prior episodes left; pick up from there. Never restart from scratch.

If your seed does NOT mention episodes (e.g. a human ran `cc-run.sh`
interactively), use the classic §6.5 self-driven loop instead.

## 7. Forbidden patterns (from prompts.py — agent failure modes)

- "complex to implement" / "multi-iteration project" / "next run
  should..." — implement now or write it in progress.md `## Untried`.
- "Let me start simple with CUDA cores" — dead end, hard performance
  ceiling. The target GPU's native instructions exist from line 1.
- "Let me use cuBLAS/cuDNN" as the kernel — black box, can't optimize.
  Library primitives (`cute::`, `cutlass::arch::mma`) are allowed as
  scaffolding, not as the kernel itself.
- "Let me try a quick re-benchmark" — variance is not improvement.
- CUDA graphs. Multiple streams + events. → both forbidden; Mirage
  manages those itself.
- Fabricating TFLOPS in commits when `./kernel` timed out. If you
  didn't see `KERNEL_RESULT` in tool output, do not write a TFLOPS
  line.

## 8. Multi-workspace isolation

You operate in exactly one workspace, identified by `$FERRET_WORKSPACE`.
Sibling workspaces (`workspace1/`..`workspace8/`) are independent: their
`.git` histories don't see each other, their `progress.md` files don't
sync, and `task.yaml` may differ. The only shared state across workspaces
is `docs/dev-memory/` (read by all mainthreads, written only by
`memory-keeper`).

Do **not** read, copy from, or git-fetch from a sibling workspace.

## 9. Standing references (read once, remember the paths)

- `examples/tcgen05-gemm/` — verified PTX patterns. When tcgen05 fails,
  the answer is here, not in a redesign.
- `examples/<task-family>/` — proven prior kernels (e.g.
  `examples/mla-mtp-decode-q1to8-kv4096/v004_q1q2_microopt.cu` is the
  current best for MLA multi-token decode).
- `docs/architecture/<gpu>.md` — hardware limits (e.g. B200 L2 size,
  TMEM lane mapping, mma.sync stall ceiling).
- `docs/patterns/` — optimization techniques (swapab, split-K,
  warp specialization, chunked prefill).
- `docs/MAPPING.md` — topic → reference file table.
- `docs/ptx-isa-9.2/` — instruction semantics.
- `resources/<lib>/` — submoduled vendor code (FlashInfer, CUTLASS,
  ThunderKittens, DeepGemm, FlashMLA). Don't try to be exhaustive —
  follow the trail your iterator/reviewer points at.

## 10. There is no other path

Earlier versions of ferret had a motus / Anthropic-API loop driven by
`orchestrator.py` + `agents.py` + `main.py` + `prompts.py` +
`cost_tracker.py`. Those files were removed when this CLAUDE.md became
the contract. The only entry point now is `scripts/cc-init.sh` →
`scripts/cc-run.sh` → this CLAUDE.md + the seven subagents under
`.claude/agents/` (planner, iterator, profiler, reviewer,
codex-dispatcher, memory-keeper, kernel-extractor). If you see a
reference to `python -m ferret.main` anywhere, it's stale — update or
delete it.
