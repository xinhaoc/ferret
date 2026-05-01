# Ferret — CUDA Kernel Optimization Agent

Autonomous CUDA kernel optimization agent. Takes a task.yaml spec, launches a motus ReActAgent on B200 GPUs, writes/edits/benchmarks kernels, tracks versions with git.

## Remote setup

- Machine: `catalyst-fleet1` (shared cluster, B200 GPUs)
- Ferret repo: `~/repos/ferret`
- Python: check `scripts/run.sh` for PYTHON path (was miniconda, may change)
- `scripts/run.sh` handles workspace init + launch

## Launching a run

```bash
# Fresh start (wipes workspace):
~/repos/ferret/scripts/run.sh tasks/<task>.yaml --max-iterations 60

# Resume (keeps workspace + git history) — DEFAULT CHOICE for continuing work:
~/repos/ferret/scripts/run.sh tasks/<task>.yaml --max-iterations 60 --keep-workspace
```

**Always default to `--keep-workspace` unless explicitly starting a new task.** Fresh start throws away hours of agent work. Every time you give a fresh command when it should be resume, hours of GPU time and API tokens are wasted.

Before launching: `cd ~/repos/ferret && git pull --ff-only` to get latest code.

## Workspace lifecycle

- `workspace/` has its own `.git` (separate from parent ferret repo)
- Agent commits `v###` tags for improvements, `a###` for failed attempts
- `workspace/` is in `.gitignore`
- **Wipe correctly**: `rm -rf workspace && mkdir workspace` (NOT `rm -rf workspace/*` — misses `.git` dotfile, agent then reads old git history from parent repo)
- Save valuable workspaces to `legacy/<task-name>/` before wiping
- Save best kernels to `examples/` so they persist across workspace wipes

## Creating a task.yaml

### Checklist

1. **Task yaml** in `tasks/` — see `tasks/template.yaml`
2. **Baseline script** in `baselines/` — must be runnable, agent calls it for KERNEL_RESULT_REFERENCE
3. **References** — proven working kernels first (agent reads top-down), then library source
4. **Validate**: `python3 task_spec.py tasks/<task>.yaml`
5. **Validate refs**: `python3 scripts/check_resource_refs.py`
6. **Commit + push immediately** — never wait for user to discover files are missing

### baseline.source

A label (e.g. "cuBLAS", "FA2 unabsorbed"), NOT a filesystem path. `main.py` validates `references[]` paths, not `baseline.source`. The agent reads `baseline.source` to know WHAT to measure, then runs the baseline script and uses its output for KERNEL_RESULT_REFERENCE.

### references

Filesystem paths the agent reads during REPRODUCE. Put the most relevant example first — agent reads top-down. Always include:
- Best prior kernel for this task family
- Relevant tcgen05/PTX examples from `examples/tcgen05-gemm/`

### constraints vs hints

- **constraints**: framework rules only. "cta_group::1 only", "single stream", "no CUDA graphs". Injected every iteration. Do NOT put optimization suggestions here — that constrains the agent's exploration.
- **hints**: first-turn only. Keep factual. Do NOT add opinions about what the agent should or shouldn't try.

## Critical lessons (from painful experience)

### FLOPS formula consistency

The #1 recurring bug. The kernel's benchmark and the baseline script MUST use the same FLOPS formula. If one uses `2 * B * H * S * S * D` (standard multiply-accumulate) and the other uses `B * H * S * S * D` (1x), all ratios are 2x inflated. Check BOTH formulas before trusting any ratio.

The kernel's formula is in `kernel.cu` (search for `double fl=`). The baseline's is in `baselines/*/baseline*.py` (search for `flops =`). They must match.

### Agent hardcodes reference numbers

The agent often measures the baseline once, hardcodes KERNEL_RESULT_REFERENCE in kernel.cu's benchmark, then never re-measures. If the FLOPS formula changes (yours or the agent's), the hardcoded reference becomes stale. The prompt says "run baselines/ script every time you tag" but agents ignore this.

Verify: after each run, check that KERNEL_RESULT_REFERENCE in the commit body matches what the baseline script actually produces on the same GPU.

### Agent picks the wrong baseline when multiple are printed

If the baseline script prints multiple references (e.g. trtllm + FA2), the agent picks whichever is most favorable. Fix: print only the target baseline. Remove alternatives or clearly label which one to use for scoring.

### MLA prefill: absorbed vs unabsorbed

- **Absorbed** (D_QK=576, D_V=512): skips kv_b_proj decompression. 3x more compute per head. Wins at short S (S≤1024) where memory dominates. Loses at long S where compute dominates.
- **Unabsorbed** (D_QK=192, D_V=128): standard MHA after decompression. What vLLM/SGLang deploy for prefill. Less compute, faster at long S.
- `trtllm_batch_decode_with_kv_cache_mla` is the ONLY trtllm API for absorbed-form MLA (handles asymmetric QK/V dims). Despite "decode" in the name, it works for any Q_LEN.
- `trtllm_batch_context_with_kv_cache` does NOT support MLA (requires headDimQk == headDimV).
- For unabsorbed prefill baseline: `BatchPrefillWithRaggedKVCacheWrapper` (FA2 JIT) or `single_prefill_with_kv_cache` (CUTLASS SM100a, faster on B200).

### FlashInfer has multiple FA2 implementations

On B200:
- `BatchPrefillWithRaggedKVCacheWrapper` → FA2 JIT kernel (what SGLang uses in production)
- `single_prefill_with_kv_cache` → CUTLASS SM100a kernel (faster, ~5-10% on most configs)
- `determine_attention_backend()` returns `"fa2"` on SM100 (not FA3, not CUTLASS auto)

Compare against the strongest baseline, not just what's convenient.

## Key architectural patterns

### swapab for small-M GEMM

tcgen05 MMA M≥64. At M=16: 75% waste. Fix: transpose so large N → MMA M, small M → MMA N.
- Verified: `examples/tcgen05-gemm/05b_cg2_swapab_small_m.cu`
- BLOCK_N=16 → illegal instruction. Minimum BLOCK_N=32 for cg2.

### swapab for MLA decode (TP)

Same principle for heads. Swap so kv_len → MMA M (fully utilized), heads → MMA N.
Cross-thread softmax needed after swap-AB QK (column-wise reduction in TMEM).
- Verified: `examples/mla-mtp-decode-q1to8-kv4096/swapab_mla_regpv.cu`

### cta_group::2

Example 05 splits M across 2 CTAs. Only works when M large enough for ≥2 M-blocks.
For small-M with swapab: cg2 MMA_M=256, MMA_N=32. Verified in `05b`.

### Chunked prefill

Same kernel as full prefill but with `q_len ≠ kv_len`. Three changes:
1. Parameters: `int S` → `int q_len, int kv_len, int q_start`
2. Grid: tile over `q_len/BM` (not `S/BM`)
3. Causal mask: `kvend = min(kv_len, q_start + qs + BM)`

Small chunks (256) have SM under-utilization — agent uses BM=32 + split-K.

## Agent failure modes

- **Noise tagging**: re-benchmarks same kernel, tags measurement variance as improvement. Prompt rule exists but not enforced.
- **Score gaming**: re-runs benchmark until lucky numbers, commits with "make sure scoring commit has good numbers."
- **CUTLASS wrapping**: wraps `GemmUniversalAdapter::run()` instead of writing `__global__`. Provide hand-written PTX examples so agent has a non-CUTLASS path.
- **cg2 failure loop**: tries cg2 6+ times via edit patches, same bug every time. Fix: provide a verified working cg2 example as starting point.
- **Ignores hints**: agent sees "try unabsorbed" hint but keeps optimizing absorbed form. May need stronger prompt or task restructuring (separate task for each approach).
- **Hardcoded references**: stores baseline TFLOPS in kernel.cu, never re-measures. Leads to stale ratios when formulas change.

## Infrastructure notes

- **Wall-time overshoot**: budget check fires between iterations only. ferret resets agent on 400 errors and continues (v3 would stop). Runs can go 2x over budget.
- **Context window**: 750K tokens. Agent runs 80 steps per iteration, multiple iterations before context fills. Much longer runs than v3 (which had 150K effective limit).
- **GPU variance**: `pick_gpu.sh` picks different GPUs per command. Measure kernel AND baseline on same GPU in same run.
- **Disk**: `/home` is shared 28T. Check `df -h` before long runs.

## File layout

```
ferret/
  main.py, orchestrator.py, prompts.py, task_spec.py, state.py, agents.py
  scripts/run.sh               # launcher
  scripts/check_resource_refs.py
  tasks/                       # task.yaml specs
  baselines/                   # baseline measurement scripts
  examples/                    # proven kernels (persist across workspace wipes)
    tcgen05-gemm/              # PTX reference progression (00-07 + swapab)
    qwen3-8b-decode-linear-bs16/   # linear GEMM kernels
    mla-mtp-decode-q1to8-kv4096/   # MLA decode kernels
    mla-prefill-b1-s1024/          # MLA prefill kernels
  resources/                   # vendored libraries (git submodules)
  docs/                        # architecture docs, PTX ISA, patterns
  legacy/                      # archived workspaces
  workspace/                   # active run (gitignored)
```

## Results

| Task | ratio | baseline | kernel file |
|---|---|---|---|
| Linear GEMM M=16 (cg1) | 1.10 | cuBLAS | `examples/qwen3-8b-decode-linear-bs16/v006_cg1_swapab_l2hints.cu` |
| MLA decode TP=2 | 1.05 | trtllm-gen | `examples/mla-mtp-decode-q1to8-kv4096/v037_tp2_swapab_unrolled_reduce.cu` |
| MLA decode TP=4 | 1.20 | trtllm-gen | `examples/mla-mtp-decode-q1to8-kv4096/v007_tp4_swapab.cu` |
| MLA decode TP=8 | 1.19 | trtllm-gen | `examples/mla-mtp-decode-q1to8-kv4096/v001_tp8_swapab.cu` |
| MLA prefill TP=8 absorbed | 1.19 (S≤1024) | FA2 | `examples/mla-prefill-b1-s1024/v024_tp8_absorbed_mmasync.cu` |
| MLA prefill TP=8 unabsorbed | 1.16-2.36 | FA2 batch | `examples/mla-prefill-b1-s1024/v006_tp8_unabsorbed.cu` |
| MLA chunked prefill TP=8 | ~tied | CUTLASS SM100a | `examples/mla-prefill-b1-s1024/v019_tp8_unabsorbed_chunked.cu` |
| MLA chunked prefill TP=8 | 1.06-1.36 | FA2 batch | same kernel |
