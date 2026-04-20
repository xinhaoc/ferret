# Ferret — CUDA Kernel Optimization Agent

## What is ferret

Autonomous CUDA kernel optimization agent. Takes a task.yaml spec, launches a motus ReActAgent that writes/edits/benchmarks kernels on B200 GPUs, tracks versions with git, and iterates until targets are met.

## Remote setup

- Machine: `catalyst-fleet1` (shared cluster, B200 GPUs)
- Ferret repo: `~/repos/ferret`
- Python: `/home/xinhaoc/miniconda3/bin/python3` (has motus, torch, flashinfer)
- Never use system `python3` — it lacks motus
- `scripts/run.sh` handles workspace init + launch. `PYTHON` env var overrides the python path.

## Launching a run

```bash
# Fresh start (wipes workspace):
~/repos/ferret/scripts/run.sh tasks/<task>.yaml --max-iterations 60

# Resume (keeps workspace + git history):
~/repos/ferret/scripts/run.sh tasks/<task>.yaml --max-iterations 60 --keep-workspace
```

`run.sh` does: verify HEAD isn't polluted with agent commits → wipe workspace (unless --keep-workspace) → git init inside workspace → launch with correct python.

## Workspace lifecycle

- `workspace/` has its own `.git` (separate from parent ferret repo)
- Agent commits `v###` tags for improvements, `a###` for failed attempts
- `workspace/` is in `.gitignore` — never leaks into parent repo
- **Wipe correctly**: `rm -rf workspace && mkdir workspace` (NOT `rm -rf workspace/*` — that misses `.git` dotfile)
- After wipe, always `cd workspace && git init -q`
- Save valuable workspaces to `legacy/<task-name>/` before wiping

## Creating a task.yaml

### Required checklist

1. **Task yaml** in `tasks/` with all required fields (see `tasks/template.yaml`)
2. **Baseline script** — agent needs something runnable to measure the reference. Put in `baselines/`. Example: `baselines/mla-mtp-decode/baseline.py --num-heads N`
3. **References** — list of readable example kernels and library source. Put proven working kernels first (agent reads top-down). Always include:
   - Best prior kernel for this task family (e.g. `examples/.../v037_tp2_swapab.cu`)
   - Relevant tcgen05/PTX examples from `examples/tcgen05-gemm/`
   - Library references from `resources/`
4. **Validate**: `python3 task_spec.py tasks/<task>.yaml` — loads and checks the spec
5. **Validate refs**: `python3 scripts/check_resource_refs.py` — ensures all referenced paths exist
6. **Commit + push immediately** — do NOT wait for the user to try launching and hit file-not-found

### baseline vs references (issue #1)

- `baseline.source`: the scoring target name (e.g. "cuBLAS", "trtllm-gen MLA decode"). Agent measures this in its benchmark and emits `KERNEL_RESULT_REFERENCE`. It's a label, NOT a filesystem path.
- `references`: list of filesystem paths the agent reads during REPRODUCE. Architectural templates. NOT the scoring baseline.
- `main.py` validates that each `references[]` path exists on disk. Does NOT validate `baseline.source` as a path.

### constraints vs hints

- **constraints**: hard framework rules, injected every iteration. E.g. "cta_group::1 only", "single stream", "no CUDA graphs". Do NOT put optimization suggestions here.
- **hints**: soft suggestions, first-turn only. Do NOT add your opinions about what the agent should or shouldn't try. Let the agent explore.

## Key architectural patterns

### swapab for small-M GEMM

When M is small (e.g. M=16 decode batch), tcgen05 MMA wastes lanes (M=64 minimum, 75% waste at M=16). Fix: transpose the GEMM so the large N goes into MMA M, small M goes into MMA N.

- Verified example: `examples/tcgen05-gemm/05b_cg2_swapab_small_m.cu` (PASS on B200)
- BLOCK_N=16 → illegal instruction. Minimum working BLOCK_N=32 for cg2.
- Best linear kernel: `examples/qwen3-8b-decode-linear-bs16/v006_cg1_swapab_l2hints.cu` (1.10 min_ratio vs cuBLAS)

### swapab for MLA decode (TP)

Same principle: with fewer heads per rank (TP=2→64, TP=4→32, TP=8→16), the M dimension (heads) is small. Swap so kv_len goes into MMA M (128, fully utilized) and heads go into MMA N.

Key difference from GEMM: attention has softmax between QK and PV. After swap-AB QK, S^T is in TMEM with kv_pos in rows. Softmax needs **cross-thread reduction** (per-head max + sum across all kv positions = across all threads).

- Verified: `examples/mla-mtp-decode-q1to8-kv4096/swapab_mla_regpv.cu` (QK MMA + cross-thread softmax + register PV, PASS on B200)
- Best TP=2 kernel: `examples/mla-mtp-decode-q1to8-kv4096/v037_tp2_swapab_unrolled_reduce.cu`
- Best TP=4 kernel: `examples/mla-mtp-decode-q1to8-kv4096/v007_tp4_swapab.cu`
- Best TP=8 kernel: `examples/mla-mtp-decode-q1to8-kv4096/v001_tp8_swapab.cu`

### cta_group::2

- Example 05 (`examples/tcgen05-gemm/05_two_sm_cluster_mma.cu`) splits M across 2 CTAs. Only works when M is large enough for ≥2 M-blocks.
- For small-M: cg2 with swapab works (verified in `05b_cg2_swapab_small_m.cu`). MMA_M=256 along N_original, MMA_N=32 (padded from 16).
- cg2 MMA M minimum is 128 (vs 64 for cg1). Doubles M-waste if used without swapab at small M.

## Common pitfalls

### Agent behavior

- **Noise tagging**: agent re-benchmarks unchanged kernel hoping for lucky numbers. Prompt rule added: "Do NOT re-benchmark and re-tag the same kernel.cu."
- **Score gaming**: agent picks favorable reference measurements. Reference numbers should come from the baseline script run consistently, not cherry-picked.
- **CUTLASS wrapping**: agent wraps `GemmUniversalAdapter::run()` instead of writing own `__global__`. Provide hand-written PTX examples in `examples/tcgen05-gemm/` so agent has a readable non-CUTLASS path.
- **cg2 failure loop**: agent tries cg2 6+ times via incremental patches, all fail on same bug. Fix: provide a verified working cg2 example (like `05b`).

### Infrastructure

- **Wall-time overshoot**: budget check fires between iterations only. Long iterations (80 steps) blow past budget. ferret resets agent on 400 errors and replays first-turn, extending the run further (v3 would stop, ferret continues).
- **GPU variance**: `pick_gpu.sh` picks different GPUs per command. Kernel and reference measured on different GPUs → noisy ratio. Lock to one GPU per run or measure both in same harness.
- **Disk full**: `/home` is shared 28T filesystem. Check `df -h` before long runs.
- **flashinfer install**: needed for MLA baselines. Currently installed via miniconda.

## File layout

```
ferret/
  main.py              # entry point
  orchestrator.py      # iteration loop, prompt construction
  prompts.py           # system prompts (REPRODUCE/OPTIMIZE)
  task_spec.py         # task.yaml loader + scoring
  state.py             # git-based run state (tags → scores)
  agents.py            # motus ReActAgent construction
  scripts/run.sh       # verified-clean launcher
  scripts/check_resource_refs.py  # validates resource paths
  tasks/               # task.yaml specs
  baselines/           # baseline measurement scripts
  examples/            # proven kernels (agent reads these)
  resources/           # vendored libraries (git submodules)
  docs/                # architecture docs, PTX ISA, patterns
  legacy/              # archived workspaces from prior runs
  workspace/           # active run (agent-managed, gitignored)
```

## Results summary

| Task | min_ratio vs baseline | kernel |
|---|---|---|
| qwen3-bs16 linear (cg1) | 1.10 vs cuBLAS | v006_cg1_swapab_l2hints.cu |
| qwen3-bs16 linear (cg2) | 1.07 vs cuBLAS | v019_swapab_cg2_l2hints.cu |
| MLA decode TP=2 (64 heads) | 1.05 vs trtllm-gen | v037_tp2_swapab_unrolled_reduce.cu |
| MLA decode TP=4 (32 heads) | 1.20 vs trtllm-gen | v007_tp4_swapab.cu |
| MLA decode TP=8 (16 heads) | 1.19 vs trtllm-gen | v001_tp8_swapab.cu |
