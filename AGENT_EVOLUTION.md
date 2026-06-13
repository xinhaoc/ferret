---
name: CUDA agent V1→V2→V3 evolution
description: Complete history of three CUDA kernel optimization agent architectures, what worked, what didn't, key findings and results
type: project
originSessionId: 5e23aceb-bdf7-4712-80cb-76fdd688059d
---
## Three Agent Versions

All live in `lithos-cuda-example/examples/`:
- `cuda_agent/` — V1 (single-agent iterative with ferret/focused-fix)
- `cuda_agent_v2/` — V2 (tree-based search with candidate layers)
- `cuda_agent_v3/` — V3 (V1 base + focus agents + later simplified to "V3-lite" two-stage)

### V1: Single-Agent Iterative (`cuda_agent/`, ~2500 lines)

**Architecture**: One main optimizer agent iterating in a loop. Ferret reviewer + focused fix agents for stuck situations.

**Key files**: orchestrator.py (923), agents.py (367), prompts.py (193), ferret.py (253), focused_fix.py (416), workspace.py (257), researcher.py (80), deep_optimize.py (403), analysis.py, main.py (158)

**Flow**:
1. Planning: 5-turn forced sequence (read docs → select references → spawn researcher subagents → measure baselines → design plan → write seed kernel)
2. Iteration loop: synthesize_guidance() profiles kernel → agent gets profile + lineage + attempts → agent edits/tests/saves
3. Stall: after 4 failures or <2% gains, forces research turn
4. Ferret: fires after 2 sessions of no gain, reviews conversation for dodged priorities, outputs DEBUG/OPTIMIZE/NONE
5. Focused fix: narrow agent implements one specific fix (cannot pivot), max 60 steps, writes to focused_fix_log.md

**Version tracking**: lineage.json + attempts/log.json (custom JSON)

**Tools**: 15 for optimizer (write/edit/read_kernel, run_command, run_ncu, save_version, save_attempt, read_docs/reference/mapping, read_sass, list/glob_files, select_references, think)

**Key design**: save_version re-benchmarks the kernel (caused double-benchmark bug + different GPU via pick_gpu.sh). Researcher subagents study references in parallel. Decision entry check blocks edits until notes.md has iteration entry. Deferral tracking across sessions detects if agent keeps flagging same priority without attempting it.

**Results**: BF16 GEMM reached 395 TFLOPS (v020 of ~23 versions). MLA decode reached 1.8 TFLOPS with tcgen05 but stuck on PV swizzle/barrier issues.

**Problems**: Read-think-read-think through 6 workflow files. Research subagent summaries ignored. save_version double-benchmark. Workspace path bugs. Agent avoids hard tcgen05 debugging.

---

### V2: Tree-Based Search (`cuda_agent_v2/`, ~1800 lines)

**Architecture**: Completely different paradigm — candidate generation upfront, then tree search with layered exploration and pruning.

**Key files**: models.py (46), main.py (249), candidate_gen.py (274), kernel_editor.py (227), search_manager.py (246), branch_runner.py (289), pruning.py (226), pruner.py (146), proofread.py (97)

**Data model**: Layer → Node (branch) → Leaf (trial). Tree state saved to JSON.

**Flow**:
1. Phase 1 — Candidate Generation: read-only agent brainstorms ALL possible single changes exhaustively. Outputs candidates.md
2. Phase 2 — Organization: candidates parsed, grouped by category, ordered strategically (structural first: tiling/pipeline/epilogue → memory → compiler knobs)
3. Phase 3 — Tree Search per layer: expand (survivors × candidates + no_change per survivor) → run each branch via kernel_editor agent → prune (soft ceiling via LLM scorer + hard ceiling by TFLOPS)

**Specialized agents**:
- Candidate gen: read-only, no edit tools, exhaustive brainstorm
- Kernel editor: narrow mandate (implement ONE change), edit-before-compile guard, max 50 steps, structured result format (SUCCESS/IMPLEMENTATION_FAILED/STRUCTURAL_FAILURE)
- Branch scorer: evaluate no-improvement branches 1-5 scale for "room to grow", reads NCU data, cache=NONE (fresh per branch)

**Key innovations**:
- "no_change" branches: parent kernel always survives to next layer even if all candidates regress
- Baseline stripping: editor prompt removes baselines/targets to prevent performance anchoring
- Soft+hard pruning: LLM scores no-improvement branches for potential, hard cap by TFLOPS
- Strategic layer ordering: structural changes first enable later optimizations
- Pre-flight checks (proofread.py): 21 sanity checks before running search

**Results**: Applied to BF16 GEMM optimization. Systematic but expensive — each layer expands survivors × candidates.

**Problems**: High cost (many agent calls per layer). Candidate gen quality depends on initial brainstorm. No iterative debugging — each branch is one-shot (implement + test, no retry). Less flexible than V1 for complex multi-step debugging.

---

### V3: V1 Base + Focus Agents, Later Simplified (`cuda_agent_v3/`, ~3500 lines total)

V3 directory contains TWO generations:
- **V3-original** (prompts.py + orchestrator.py + agents.py): V1 base enhanced with focus_agent.py for parallel direction exploration
- **V3-lite** (prompts_v2.py + orchestrator_v2.py + agents_v2.py): radical simplification to two-stage design

#### V3-original (orchestrator.py, ~985 lines)

Same as V1 but adds:
- **focus_agent.py** (268 lines): multi-session deep exploration of one direction. Cannot switch directions. Sub-workspaces in branches/. Best result promoted to main lineage. 3 sessions max per direction.
- **ferret.py** rewritten: stateless, reads filtered conversation + tool_calls since last call, outputs SPAWN with ranked directions or NONE
- **orchestrator.py** wired: ferret trigger → parse decision → dispatch focus agents → promote results

**Flow addition**: On resume, if no gain in 2 sessions → ferret reviews → SPAWN dispatches focus agents on top-ranked directions → results promoted back

#### V3-lite (orchestrator_v2.py, ~403 lines) — THE CURRENT SYSTEM

**Architecture**: Two stages (REPRODUCE → OPTIMIZE) with minimal orchestration.

**Key files**: prompts_v2.py (234), orchestrator_v2.py (403), agents_v2.py (165), main_v2.py (103)

**Two stages**:
1. REPRODUCE (best < 90% of baseline): Don't profile. Study baseline source. Reproduce its architecture. Don't fall back to CUDA cores.
2. OPTIMIZE (best ≥ 90%): Profile-driven. Read broadly. Try bold changes. Check progress.md for untried ideas.

**Key simplifications from V1**:
- No multi-turn forced planning sequence
- No research subagents — agent reads references directly
- No ferret, no focus agents
- No workflow files — workflow inline in system prompt
- Git-based version tracking (tags for improvements, commits for attempts)
- progress.md (append-only) replaces notes.md (overwritten each session)
- 11 tools (removed save_version/save_attempt/select_references)
- ~300 lines orchestrator vs ~1000

**Hardcoded values** (must change per task):
- `_baseline_source`: path to reference code in resources/
- `_get_baseline_tflops()`: returns hardcoded number

**Resume injects**: spec.yaml, full git history (commit messages), baseline hint (REPRODUCE only)
**Iteration injects**: stage-aware prompt, numbers (best, baseline, percentage)
**OPTIMIZE iteration**: "Read workspace/progress.md first. Try the hard ones you've been avoiding."

**Results**:
- MLA decode: 46.2 TFLOPS, 44% above CUTLASS baseline (but unfused 5-kernel, batch timing, split-K correctness bug at sk≠32)
- MLA prefill: 260 TFLOPS, 3% above FlashInfer FA2 baseline (fused single kernel, verified fair comparison)

---

## Key Behavioral Findings (across all versions)

1. **Agent reproduces baseline's instruction set**: FA2 baseline (mma.sync) → agent writes mma.sync. CUTLASS baseline (tcgen05) → agent writes tcgen05. Never switches spontaneously.

2. **Agent avoids hard rewrites**: Recognizes tcgen05 would help, writes "try tcgen05" in notes, then picks a simpler micro-optimization instead. Every time. Across all versions.

3. **80 steps too long without check-in**: Agent drifts by step 30. System prompt buried under 300K tokens by step 50. Iteration prompt (from orchestrator) is more powerful than system prompt.

4. **SASS dumps cause analysis paralysis**: Full SASS → 20+ steps analyzing → proposes 10 ideas → implements none.

5. **notes.md gets overwritten**: Agent rewrites notes each session, losing plans. Fixed with append-only progress.md in V3-lite.

6. **Double-benchmark bug** (V1/V3-original): save_version re-ran benchmark on potentially different GPU. Fixed in V3-lite (agent records own numbers in git commits).

7. **Baseline determines everything**: The choice of baseline is the single most important design decision. It determines instruction set, architecture, and performance ceiling.

8. **Prompt says X, agent does Y**: "Don't fall back to CUDA cores" — agent falls back. "Read the baseline source" — agent reads it once, never returns. "One change at a time" — agent makes multiple changes. Prompts set initial direction but lose influence over time.

9. **Progress tracking across sessions**: Git history shows what was tried. progress.md (append-only) shows plans and analysis that weren't committed. Both needed for continuity.

10. **NCU broken by /tmp permissions**: On this cluster, ncu failed with "Unknown error on device 0". Fix: `TMPDIR=/tmp/$USER`. Wasted multiple sessions.

## Key Technical Findings

1. **mma.sync ceiling on B200**: ~33% MIO scoreboard stalls from ldmatrix 2-way bank conflicts (pigeonhole: 16 rows × 16B > 128B bank period). Only tcgen05 can fix it.

2. **tcgen05 TMEM for MLA**: D_V=512 needs all 512 TMEM columns. Can't fit S (QK scores) and O (output) simultaneously without split/time-multiplex.

3. **FlashInfer FA2 on B200**: Uses mma.sync (SM80), not wgmma (SM90, broken on B200) or tcgen05 (SM100, not wired up in FlashInfer 0.6.7 for MLA prefill). The "fa3" backend generates SM90 code that can't run on SM100.

4. **Split-K correctness**: Decode v024 only correct at sk=32 (1 tile per CTA). Online softmax accumulation across multiple tiles has a bug in the correction factor.

5. **Batch vs per-iteration timing**: Batch timing (N iterations in one event pair) amortizes launch overhead. Significant for multi-kernel pipelines. CUTLASS baseline uses per-iteration. Must match methodology for fair comparison.

## What Worked Best

- **V3-lite's two-stage design**: Simple, clear, effective. Agent knows exactly what to do in each stage.
- **Git for version tracking**: Natural, agent already knows it, no custom JSON formats.
- **Baseline source injection**: Pointing to specific files (not directories) helps reproduction.
- **progress.md append-only**: Carries plans across sessions without overwriting.
- **Stall detection + forced research** (V1): Good idea, but V3-lite's shorter iterations make it less necessary.

## What Didn't Work

- **Research subagents** (V1): Summaries often ignored by main agent.
- **6 workflow files** (V1): Caused read-think-read-think loops.
- **Ferret + focus agents** (V1/V3-original): Complex machinery, never conclusively proved value.
- **Tree search** (V2): Systematic but expensive, one-shot branches can't debug complex issues.
- **"Don't fall back" prompts**: Agent rationalizes around any instruction.
- **SASS analysis tool**: Returns too much data, agent gets analysis paralysis.

## File Locations

- V1: `lithos-cuda-example/examples/cuda_agent/`
- V2: `lithos-cuda-example/examples/cuda_agent_v2/`
- V3: `lithos-cuda-example/examples/cuda_agent_v3/` (both original + lite)
- Entry: `cuda_agent_v3/main_v2.py` (run from cuda_agent_v3/ directory)
- Remote: `<host>:~/repos/lithos/examples/cuda_agent_v3/`
- Saved workspaces: workspace_gemm, workspace_v2_run1, workspace_mla_decode_v2, workspace (current prefill)
- Best kernels: v024.cu (decode, 46.2 TFLOPS), v016_prefill.cu (prefill, 260 TFLOPS)
