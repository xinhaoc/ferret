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

| Trigger | Subagent | Call signature |
|---------|---------|----------------|
| Cold-start (workspace empty) | `planner` | `Task(subagent_type=planner, prompt="cold-start $FERRET_WORKSPACE")` |
| Before each iteration's `Edit` | `iterator` | Pass it the state CLI output + last 3 commit bodies + worst config name |
| After `git tag v###` or `a###` | `reviewer` | Pass tag name + iterator's last list (if any) |
| OPTIMIZE stage, want a profile | `profiler` | One workspace per call. Reads `.profile_last.json` automatically. |
| Reviewer needs Mirage-API check | `reviewer` calls `codex-dispatcher` itself — you don't invoke it directly |
| Discovered host-level fact | `memory-keeper` | Pass `{category, fact}` — never edit `docs/dev-memory/` yourself |

**You never call `codex-dispatcher` or `memory-keeper` directly.** They
are invoked only via `reviewer`. Calling them from the mainthread
bypasses the audit trail and confuses the review record.

## 3. Files: who owns what

| Path | Writer | You may read? |
|------|--------|---------------|
| `$FERRET_WORKSPACE/kernel.cu` | **You** | yes |
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

## 10. The legacy motus path

`orchestrator.py`, `agents.py`, `main.py`, `prompts.py`, `cost_tracker.py`
implement the previous API-driven loop. They are kept for parity but are
not part of the Claude-Code mainthread path. Don't import from them and
don't `python -m ferret.main` — that's the legacy entry point.

The Claude-Code path is: `scripts/cc-init.sh` → `scripts/cc-run.sh` →
this `CLAUDE.md` + the six subagents under `.claude/agents/`.
