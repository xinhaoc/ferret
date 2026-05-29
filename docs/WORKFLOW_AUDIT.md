# Ferret kernel-agent workflow audit + hardening plan

Owner: mirage main agent. Goal (user, 2026-05-29): make the ferret kernel-agent
workflow reliably produce a **good, performant kernel end-to-end within ~a day**,
using "write a Split-K Dense GEMM" as the acceptance test. MPK optimization is
paused until this toolchain is solid. Don't rush — budget for 1-2 days.

## The pipeline (7 stages)

| # | Stage | Component(s) | Usability check (how to verify) | Status / known defect |
|---|-------|--------------|----------------------------------|-----------------------|
| S1 | Dispatch + launch | mirage `ferret-kernel-agent` → `cc-run.sh` → `cc-init.sh` → `claude` mainthread | smoke: one dispatch spawns a live mainthread on a clean ws+GPU, goal injected, first tag appears | dispatcher early-return **FIXED**; HEAD-guard catastrophic-advice **FIXED**. Mostly OK; needs a smoke re-verify. |
| S2 | Cold-start plan | `planner` | progress.md has a sound Plan + correct starting file + Mirage ABI captured | Worked in ws6 (found the MPK kernel + B36 bug + the fix). Gap: no target-feasibility check. |
| S3 | Iterate | `iterator` + mainthread write→nvcc→bench→tag loop | valid KERNEL_RESULT lines + git tags + score progresses; no rabbit-hole | **D2**: ws6 rabbit-holed a003-a008 (swapAB) for ~most of 835k tokens; §6.5 stall rule says "pivot, never stop" + there is no budget / iteration cap / diminishing-returns harvest. |
| S4 | Per-tag review | `reviewer` + `codex-dispatcher` | API/output/constraint checks correct; codex ACTUALLY runs | **D3**: codex-dispatcher returned "codex_unavailable" every run (ws3, ws6) → "API: NOT VERIFIED" always → the Mirage-ABI check never actually executed. |
| S5 | Memory | `memory-keeper` | appends useful host facts to docs/dev-memory | lightly exercised (quirks.md/tips.md grew); low risk. |
| S6 | Convergence → delivery | `reviewer` step-5 + `kernel-extractor` | produces a valid, ABI-matching `kernel.cuh` that compiles | **D1 + D5**: gate requires `advance? True` AND every config ✓ (BOTH in reviewer step-5 and kernel-extractor preconditions). One infeasible per-config target ⇒ never converges ⇒ extractor NEVER RAN in any run ⇒ the entire .cu→.cuh delivery is UNTESTED. |
| S7 | Integration (mirage side) | cp kernel.cuh → MPK + build + measure | drops into MPK, JIT-compiles, runs, helps | Untested via the flow (no kernel.cuh was ever delivered). |

## Core defects (ranked)

- **D1 — all-or-nothing convergence (S6).** `ferret.state advance? True` + every
  ✓ is required by BOTH reviewer step-5 and kernel-extractor preconditions. A
  single infeasible target (o_proj M=1 can't beat cuBLAS `cta_group::2` while
  ferret is constrained to `cta_group::1`) ⇒ permanent non-convergence ⇒ a
  working kernel (ws6 v002) is never delivered. **This is the #1 flaw.**
- **D2 — no budget/stall harvest (S3).** §6.5: after 6 same-score iters, "pivot,
  do NOT stop." No iteration/token cap, no "harvest the best usable kernel and
  stop." Directly caused the 835k-token swapAB rabbit hole.
- **D3 — codex-dispatcher unavailable (S4).** API verification never ran. Either
  the subagent lacks the Task tool to invoke codex, the `codex` CLI is missing,
  or the invocation is mis-wired. The ABI is currently checked only by manual
  read, not by codex.
- **D4 — infeasible targets authored (dispatch/S2).** My dispatcher set per-config
  `target_ratio` (e.g. 1.10 vs cuBLAS on o_proj M=1) that is architecturally
  unreachable under the constraints → guarantees D1.
- **D5 — kernel-extractor never exercised (S6).** Unknown whether extraction
  even works (produces a compiling, ABI-correct .cuh). Must test in isolation.

## Milestones

- **M1 — Launch reliability (S1).** Smoke-verify a clean dispatch → live
  mainthread → first tag. (Largely done; confirm.)
- **M2 — Per-subagent unit usability (S2-S5).** Exercise each subagent in
  isolation; **fix D3 (codex-dispatcher)**. Cheap: use ws6's existing artifacts.
- **M3 — Convergence + delivery works (S6).** Fix **D1** (add a "deliver best
  usable kernel" path: extract on stage-gate-met + stall/budget, not only
  all-✓) and **verify D5** by running kernel-extractor on ws6's v002 NOW (no new
  run) — does it yield a compiling, ABI-correct kernel.cuh?
- **M4 — Budget/stall discipline (S3).** Fix **D2**: iteration + wall/token cap;
  on stall or cap, harvest-best + stop instead of pivoting forever.
- **M5 — Feasible task authoring (D4).** Dispatcher sets achievable per-config
  targets (or marks a config "best-effort / architecture-capped") + frames the
  goal as "deliver a consumer-usable kernel," not "beat external SOTA on every
  shape."
- **M6 — END-TO-END acceptance.** Dispatch the Split-K Dense GEMM task (feasible
  targets, fixed workflow) → ferret delivers a good kernel.cuh within ~a day →
  cp into MPK → JIT-compiles + runs (correctness smoke). THIS is the user's
  success bar.

## Cheap unit-check strategy (no new ferret run needed for M2/M3-verify)

ws6 already exercised S2-S4 + has a v002 kernel.cu. Exercise the BROKEN/UNTESTED
stages in isolation against that existing artifact:
- S4/D3: invoke codex-dispatcher directly on ws6 → does it run or return
  codex_unavailable? Root-cause + fix.
- S6/D5: invoke kernel-extractor on ws6 v002 (bypass the convergence gate
  manually) → does it produce a compiling, ABI-correct kernel.cuh? This both
  tests extraction AND yields the very kernel we want.

## Status log
- 2026-05-29: audit written. Prior fixes landed: dispatcher blocks-to-converge
  (Step 5 rewrite), HEAD-guard safe-advice + FERRET_ALLOW_AGENT_HEAD bypass,
  stray-commit relabel. Next: M2 cheap checks (codex + extractor on ws6 v002).

## 2026-05-29 — CRITICAL workflow gap found via the o_proj crash

Hand-grafting ferret ws6 v002's B36 fix into MPK's kernel + enabling
MPK_DSV3_DECODE_OPROJ_SPLITK=1 → STILL crashes in MPK ("illegal memory
access"), n=3. ferret validated v002 STANDALONE (own __global__ launcher,
stress grid, NS=5), but MPK invokes the task_impl with NS=3 + the persistent-
worker model (num_workers=136, worker_idx from the scheduler, total != 
n_tiles*SPLIT_K). The standalone validation did NOT cover MPK's actual call.

WORKFLOW REQUIREMENT (add to acceptance): ferret's standalone benchmark
harness MUST mirror how MPK actually calls the kernel — same template params
(NS the register passes), same worker model (num_workers + worker_idx +
multi-tile-iter via total>num_workers), same pre-zeroed-output + grid
semantics. A kernel that's "validated standalone" but crashes in MPK is NOT
delivered. Either: (a) the task.yaml harness benchmarks at MPK's NS + worker
model, or (b) the acceptance includes an in-MPK smoke (cp kernel.cuh + a tiny
TP=4 run) before declaring success. Reverted the hand-graft — ferret will
produce the MPK-faithful version through the fixed workflow.

## 2026-05-29 — workflow redesign edits landed (component-fix phase)
Done (all prompt/spec edits, no GPU):
- CLAUDE.md §6.5: FINALIZE = goal-reached OR best-effort (stage-gate met +
  stall[3 attempts]/budget[~25 iters]/infeasible-target). Mainthread invokes
  kernel-extractor itself; deliver-best is the contract. Fixes D1+D2.
- CLAUDE.md §2: mainthread is sole orchestrator; no nested subagent dispatch.
  Fixes the root cause of D3 + D5 (nothing was ever delivered).
- reviewer.md step1: runs `codex exec` inline via Bash + manual-Read fallback
  (never silent NOT VERIFIED). Fixes D3.
- reviewer.md step5: returns FINALIZE verdict (goal-reached/best-effort-ready/
  no); does NOT invoke extractor.
- kernel-extractor.md: best_effort mode — deliver best correct tag without
  requiring every ✓ (default mode still strict). Fixes D1/D5.
- task dense-fp8-gemm-decode-splitk.yaml: target_ratio 1.00 (not-worse bar,
  >=1.10 stretch) + MPK-FAITHFUL HARNESS constraint (NS=3, num_workers=136
  worker model, multi-tile-iter exercised, pre-zeroed output) so a validated
  kernel actually runs in MPK. Fixes D4 + the crash-gap.

NEXT (per-stage verify, then combine — user's plan):
- M3-verify/D5: dispatch ferret to RUN the extractor on a delivered kernel —
  confirm .cu→.cuh produces a compiling, ABI-correct header (UNTESTED stage).
- M6 end-to-end: dispatch the split-K task through the fixed workflow → ferret
  delivers a good kernel.cuh → cp to MPK → in-MPK smoke (no crash + perf).
- Then generalize the dispatcher (mirage ferret-kernel-agent.md) feasible-target
  + MPK-faithful-harness framing for future tasks.

## 2026-05-29 — ORCHESTRATION-LOOP redesign (user's orchestrator.py-in-dispatcher)
The dispatcher (mirage ferret-kernel-agent) is now the durable loop controller
(replaces ferret v0.1 orchestrator.py). Architecture:
- Each `cc-run.sh ... --prompt` = ONE bounded `claude -p` episode that does ~4
  iterations then EXITS (print mode returns). Foreground/blocking — control
  returns to the dispatcher = its decision point.
- Dispatcher Step-5 LOOP: run episode → state CLI → decide {advance?True →
  finalize(goal); OPTIMIZE+stall≥3 or near MAX_ROUNDS → finalize(best-effort);
  else next round}. MAX_ROUNDS=8, STALL=3. Finalize = an episode that invokes
  kernel-extractor → kernel.cuh.
- ferret CLAUDE.md §6.6: episode mode — bounded chunk + EPISODE_STATUS + exit;
  supersedes the in-session "never stop"; robust to the 5-hr limit landing
  BETWEEN episodes (dispatcher just resumes). FINALIZE=<mode> seed → extract.
Why better: no 835k-token runaway; clean checkpoint+decide each round; resumable;
sidesteps nested-dispatch entirely (dispatcher is depth-0, invokes everything).

NEXT: first end-to-end validation run (bounded MAX_ROUNDS) on the feasible+
MPK-faithful split-K task → does the loop deliver a compiling kernel.cuh?

## 2026-05-29 — CALIBRATION phase (M5): reviewer overturned the baseline + cleanup
ablation-logic-reviewer audited my "re-scope split-K task" reasoning. Key corrections
(all evidence-verified by me afterward):
- WRONG BASELINE: ferret's harness benches **cuBLASLt** (ws6/kernel.cu:22 "KERNEL_RESULT_
  REFERENCE = in-process cuBLASLt"), task.yaml mislabels it "DeepGEMM". Neither is the
  kernel being replaced. The user's bar is "not worse than MPK's CURRENT dense smallm/
  mediumm". mediumm beats DeepGEMM 1.14-3.82x → cuBLAS/DeepGEMM is the WRONG denominator.
  → Re-baseline the task to MPK's mediumm/smallm. (M5 fix.)
- SELECTION IS BY max_seq_length, NOT M: builder.py:717 `smallm if max_seq_length<=512
  else mediumm`. Decode workload (long ctx) ⇒ MEDIUMM is what's replaced. And MPK runs
  M=mbt capacity w/ runtime gating, not literal M=1 — so bench M in {1,4,128}.
- o_proj 71% mechanism is NOT "K=2048 too short for split-K" (ferret's own ablation: SPLIT_K
  is FLAT on o_proj ⇒ not K-reduction-bound ⇒ it's M=1 MMA-row-waste/atomic-contention +
  under-iteration + strong cuBLAS denominator). Don't drop o_proj for the wrong reason;
  re-baseline it to mediumm (where mediumm is most under-utilized → split-K likely WINS).
- Don't game the gate by deleting o_proj; it's only safe because builder has a mediumm
  fallback (MPK_DSV3_DECODE_OPROJ_SPLITK opt-in). Keep min_ratio but with mediumm denominators.

ACTION (M5 calibration, in flight): codex bg run (id b0h3o9pya) benching split-K(v002) +
mediumm + smallm + cuBLAS on qkv_a/gate_up/o_proj × M{1,4,128}, validated vs host FP32,
→ /tmp/calib_out.md. Base = git show v002:kernel.cu (ws6 tree is dirty on swapAB a008).
Then re-author tasks/dense-fp8-gemm-decode-splitk.yaml: baseline=mediumm, per-config
target=beat-mediumm, → e2e loop run.

CLEANUP: killed a runaway ferret mainthread (PID 1122104) still grinding ws6's OBSOLETE
task (110% vs cuBLAS, infeasible) from a prior session — the exact D2 pivot-forever failure
the redesign fixes. Plus stale tail -f monitors + an nvcc probe. ws6 left dirty on a008.

## 2026-05-29 — CALIBRATION RESULT (split-K v002 vs the REAL baseline mediumm)
Subagent built calib_scratch/calib.cu (v002 harness + mediumm/smallm reference lines,
local header copies to dodge NVSHMEM link; GEMM bodies unchanged), ran on GPU 4, all
PASS host-FP32 (rel err ~0.002-0.003). TFLOPS / ratio_vs_mediumm:
  qkv_a  M1 : splitk 2.18  mediumm 0.95  → 2.29x   |  M4: 8.70 / 3.80 → 2.29x | M128: 194.7/114.7 → 1.70x
  gate_up M1: splitk 3.19  mediumm 1.79  → 1.78x   |  M128: 262.1/215.7 → 1.22x
  o_proj M1 : splitk 2.05  mediumm 1.60  → 1.28x   |  M4: 8.19/6.77 → 1.21x  | M128: 152.3/203.5 → 0.75x (LOSES)
mediumm ≈ smallm everywhere (only NE differs). cuBLAS is 2.3-2.5x of mediumm (the
"71% of cuBLAS" o_proj number was the WRONG denominator — vs mediumm split-K WINS).

CONCLUSION (vs user's bar "not worse / >=10% better than the replaced kernel = mediumm"):
split-K beats mediumm 1.21-2.29x at the DECODE regime (M=1-4) on all 3 shapes. Only loss
is o_proj M=128 (0.75x) — large-M, ample N-tiles, no need to split K.

⚠ CAVEAT (load-bearing, almost missed): split-K measured at NS=5/NE=2 (harness default);
MPK calls the split-K task at NS=3 — the exact gap that crashed v002 in MPK. So these
ratios are NS=5; the MPK-faithful NS=3 split-K is UNMEASURED + the multi-tile-iter
correctness at NS=3 is the unresolved B36 bug. The e2e ferret run (MPK-faithful NS=3
harness) is still needed to PRODUCE a kernel that wins AND is correct at NS=3.
→ dispatching ablation-logic-reviewer to audit this interpretation before re-authoring
the task + launching the e2e run.

## 2026-05-29 — Explore ground-truth shapes + TASK RE-AUTHORED (M5 done) → launching M6
Explore pinned the real TP=4 decode dense-GEMM shapes (cited builder.py/task_register.cc):
  qkv_a   : N=2176 K=7168           NS=3 NE=4 mediumm
  gate_up : N=9216 (NOT 4096!) K=7168
  o_proj  : N=7168 K=1792 per-rank (NOT 2048/16384!) — RowParallel shards K 7168/4
  compile-M=128 (mbt); at TPOT active_rows=1, but the kernel COMPUTES a full 128-row
  MMA tile (write-gated to active_rows) → decode cost = ONE 128-row row-tile latency,
  NOT M=1 TFLOPS (reviewer hole A confirmed: common.cuh ~314 scales-gated, ~328-357 MMA
  unconditional, ~362 write-gated).
Implication: split-K's benefit ∝ K. qkv_a/gate_up (K=7168) WIN at M=128 (calib 1.70x/1.22x
vs mediumm). o_proj (K=1792) too short → LOSES (0.75x) → stays on mediumm.
Re-authored tasks/dense-fp8-gemm-decode-splitk.yaml: baseline=mediumm@NS=3, seed harness=
calib_scratch/calib.cu, bench M=128, TARGET qkv_a_M128 + gate_up_M128 (min_ratio, >=1.00),
o_proj+M4 reported-only (weight 0), NS=3/NE=2 MPK-faithful, multi-tile-iter correctness +
INVALID-on-mismatch mandated. Both audits back this; the prior cuBLAS/DeepGEMM baseline +
M=1 + o_proj-in-min_ratio were all wrong (would never deliver).

NOW: M6 first e2e validation — dispatch ferret-kernel-agent (orchestration loop, bounded
MAX_ROUNDS=3) on ws7 → does the loop run bounded episodes, state-check+decide, and DELIVER
a kernel.cuh (the never-tested stage)? Then a longer run if the mechanism holds.

## 2026-05-29 — M6 END-TO-END VALIDATION: SUCCESS (first round) ✅
Dispatched ferret-kernel-agent (orchestration loop, MAX_ROUNDS=3) on ws7 + the re-authored
task. Result (VERIFIED by me against ground truth, not just subagent self-report):
- Loop mechanism WORKS: round1 (cold-start→planner→seed calib.cu→NS5→3 edits→compile→run→
  tag v001→reviewer, ~11min, EPISODE_STATUS printed, exit0); round2 (FINALIZE=goal→
  kernel-extractor→kernel.cuh, ~1.5min). Decision logic fired (advance?True→deliver). No hang.
- DELIVERY STAGE (D5, never before tested) WORKS: ws7/kernel.cuh (18805B) sanity-compiles
  (nvcc --device-c exit0), ABI byte-identical to MPK fp8_gemm_dense_decode_splitk_sm100_task_impl
  <BN,NS,NE,SPLIT_K>. DIFF vs in-tree = ONLY the B36 fix (ph-toggle → continuous gk/gki phase)
  + license header. Surgical, correct-looking.
- PERF vs mediumm@<128,NS=3,NE=4> (v001 KERNEL_RESULT, MPK-faithful NS=3 harness):
  qkv_a_M128 194.97/121.86 = 1.60x ✓ ; gate_up_M128 485.74/412.55 = 1.18x ✓
  (both >> ≥10% bar). o_proj tripwire 1.00x (holds, stays on mediumm). qkv_a_M4 2.29x.
- B36 multi-tile-iter: validated at NS=3; gate_up (72*2=144>136 workers) exercises multi-wave;
  host-FP32 rel err <1e-2 all configs.

⇒ The ferret WORKFLOW is validated end-to-end; the user's STANDALONE acceptance (ferret
produces a SplitK kernel, standalone ≥10% better than the replaced mediumm) is MET.

RESIDUAL RISK: the earlier MANUAL graft of this SAME continuous-counter fix CRASHED in MPK.
ferret's passes the MPK-FAITHFUL STANDALONE harness (NS=3, nw=136, multi-tile-iter) — far
better de-risked — but "works in MPK" is only PROVEN by integration (reviewer's "end at
in-MPK per-MoE-layer wallclock"). MPK integration = resuming the PAUSED MPK-perf phase → user's call.

WORKFLOW FRICTION to fix (harness improvements, the user's actual goal):
1. task_spec.py: scoring=min_ratio does NOT skip weight=0 configs → a reported-only tripwire
   <1.0 falsely blocks OPTIMIZE. Fix: min_ratio should skip weight==0 (or add reported_only
   flag). Workaround used: scoring=weighted_avg (the linter/user applied this + target_ratio
   0.0→0.01). REAL FIX still pending in task_spec.py.
2. Validator rejects target_ratio=0.0 (needs >0). Fixed in yaml to 0.01; consider allowing 0.
3. Round-1 claude -p stdout went to the bg-task output file, not workspace7.log (the tee ran
   inside a run_in_background Bash). Dispatcher should write to workspace${N}_round${r}.log.

## 2026-05-29 — IN-MPK CRASH-TEST: ferret VINDICATED; crash is pre-existing/excluded-path
Ran the in-MPK TP=4 crash-test (o_proj split-K path, MPK_DSV3_DECODE_OPROJ_SPLITK=1) via the
opportunistic poller (grabbed quad 1,2,3,4 at poll 140). RESULT: CRASH "illegal memory access"
at the NVSHMEM barrier (rank2, RC=255) — SAME class as the pre-ferret kernel.

ablation-logic-reviewer + 2 free checks corrected my (wrong) UE8M0-scale-byte theory:
- The installed kernel uses FP32 scales indexed by GLOBAL k-tile (kg) — NO UE8M0 packing/sf_id.
  The "byte-0" bug is a DIFFERENT kernel (linear_fp8_swapAB) + a silent-miscompare, not an OOB.
- CHECK 1: ferret's diff = ONLY gk/gki phase counters; red.global.add ADDRESSING grep = 0 (byte-
  identical to the pre-ferret kernel that ALSO crashed at TP=4). ⇒ crash is PRE-EXISTING +
  B36-ORTHOGONAL. ferret fixed B36 correctly + introduced nothing. WORKFLOW VINDICATED.
- Journal (2026-05-27 compute-sanitizer): TP=1 clean, TP=4 OOB localized to the NVSHMEM
  RowParallel partial-buffer + AllReduce path; cause UNPINNED (never root-caused). o_proj-specific.
- CHECK 2: the crash path is `residual is not None` → _new_tp_partial + _allreduce_residual
  (RowParallel/NVSHMEM) = o_proj ONLY. qkv_a (down-proj) + gate_up (ColumnParallel) are
  residual=None → plain output → NOT the danger path. So the TARGETS are safe.

STATUS: ferret WORKFLOW validated (produces correct, fast, MPK-faithful-standalone kernels +
correctly fixed B36). Standalone goal MET (1.60x/1.18x vs mediumm). The in-MPK crash is a
pre-existing, excluded-path (o_proj RowParallel NVSHMEM) bug — NOT ferret's kernel.

WORKFLOW LESSON: the MPK-faithful standalone harness was faithful on NS/workers/multi-tile-iter
but CANNOT model the TP>1 NVSHMEM RowParallel partial-buffer output context. For RowParallel-
consumed kernels, the robust acceptance gate is an in-MPK TP=4 smoke (which caught it), not an
ever-more-faithful standalone harness. Encode this in the dispatcher (done: Step-0 quality bar
already says "end at in-MPK metric" + "model the TP>1 NVSHMEM output context").

REMAINING for (1b) confirm-speedup: wire qkv_a/gate_up (safe, residual=None, K=7168) through the
INTERNAL decode_splitk kernel (mirror o_proj _emit_decode_splitk; pre-zero output; split_k=8/2)
+ in-MPK TP=4 smoke (cosine vs env-off baseline) + per-MoE-layer wallclock. Substantial NEW
builder wiring + GPU-contended → arguably the first perf-phase task. Surfaced to user.
