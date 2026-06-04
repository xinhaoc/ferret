---
name: kernelwiki
description: Use BEFORE/while writing or stuck on an MPK CUDA kernel (Blackwell SM100 / Hopper SM90) to pull relevance-ranked SOTA prior-art — the closest external kernel (DeepGEMM/CUTLASS/FlashInfer/vLLM/SGLang/FlashMLA), its verbatim reference code, and a perf_claim to anchor target_ratio. Query at planner cold-start (closest SOTA template) and at iterator stall (by performance symptom, e.g. low-sm-utilization). NOT for host-side/framework integration, distributed (DeepEP/EPLB/TP-comm), or generic CUDA Q&A.
argument-hint: "[op precision arch keywords] | [--symptom low-sm-utilization] | [page-id]"
allowed-tools: "Bash Read Grep"
---

# kernelwiki — query SOTA kernel prior-art for ferret

A local, OFFLINE knowledge base vendored as the `resources/kernelwiki` submodule
under ferret (2179 merged PRs + 48 synthesis pages, Blackwell/Hopper). Use it so ferret seeds from
**external SOTA** (the user's rule: refs = external SOTA, not our in-tree buggy
kernel) and anchors `target_ratio` to a real number — instead of guessing or
relying only on the frozen `examples/<family>/` winner.

## 3-command runbook (all offline; run via Bash)

```bash
KW="${FERRET_ROOT:-$HOME/ferret}/resources/kernelwiki"
# 1. RANK closest SOTA pages for this task (op + precision + arch keywords):
python3 $KW/scripts/query.py "fp8 dense gemm decode skinny-m" --type kernel --architecture sm100 --compact --limit 5
#    stuck on a bottleneck? query by SYMPTOM instead:
python3 $KW/scripts/query.py "" --symptom low-sm-utilization --compact --limit 5    # also: tail-effect, register-pressure, memory-bound
# 2. READ the chosen page + its one-hop PR provenance (named baseline + perf_claim to anchor target_ratio):
python3 $KW/scripts/get_page.py <page-id> --follow-sources
# 3. PULL verbatim CUDA to seed kernel.cu / the task.yaml references[]:
python3 $KW/scripts/get_page.py <page-id> --include-code
```
(Run `query.py --help` / `get_page.py --help` if a flag is unclear — flags evolve.)

## How to use the result
- **Planner cold-start:** cite the page id + its 6-field perf_claim in `progress.md`
  ("SOTA prior-art (KernelWiki): <id>, claim=<...>"); optionally save the
  `--include-code` dump as a candidate starting file. This replaces hand-wiring
  `task.yaml.references[]` for a new task.
- **Iterator on STALL only** (not every iteration — latency/prompt bloat): when a
  config is stuck (stall≥2) query by the bottleneck symptom for a grounded next move.

## CAVEATS (encode these — they are why a naive copy regresses)
1. **target_ratio bar stays the REAL in-tree `mediumm`** (ferret's existing rule).
   The wiki perf_claim is a SANITY CEILING / target HINT only — it was reported
   upstream, NOT measured on this B200, so never quote it as "achieved" (the
   KERNEL_RESULT-observed-this-iteration rule still governs).
2. **M=1 / skinny-M decode goal:** the strongest GEMM pages (DeepGEMM) are
   LARGE-M-tuned — copying their mainloop WORSENS the M=1 under-occupancy. For
   M=1 pull the **skinny-M / tile-scheduling / CLC / tail-effect** pages instead.
3. **Scope = the compute half only** (kernel-level, Blackwell-first). It cannot
   inform the ~60μs system/scheduler-overlap or TP-comm half of the 282→150μs gap.
4. Wiki cutoff is dated (`data/refresh-cutoff.yaml`); to refresh the corpus run
   `scripts/update_kernelwiki.sh` (upstream-sync + optional gh-ingest; see
   `docs/kernelwiki-refresh.md` for the mechanics).
