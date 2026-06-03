# In-MPK kernel-correctness validation harness

Closes the **"standalone-correct but in-MPK-crash" gap** — the root of the
SplitK Heisenbug, where a kernel passes ferret's own standalone host-reference
check but crashes (`Invalid __global__ read`, illegal memory access) or
silently miscompares once it runs through the *real* MPK compile pipeline
(`graph.cc` dispatch → `task_register.cc` codegen → `tma.cuh` descriptors →
megakernel `nvcc` → scheduler dispatch).

The Kernel Agent invokes this at convergence (see
`.claude/agents/mpk-validator.md`) to **self-validate** a candidate kernel
through that real path on a single, exclusive GPU before delivery.

Driver: `scripts/mpk_validate.sh <WS_INDEX> <KERNEL_NAME> <TEST_DRIVER>`.

---

## Pattern A vs Pattern B — the decision

```
Does an MPK *layer* already route to this kernel's task header?
  (i.e. is there a `<kernel>_layer(...)` in persistent_kernel.py AND a
   register_<kernel>_task in task_register.cc AND a dispatch in graph.cc?)

  ── YES ───────────────────────────────────────────►  PATTERN A   (PREFERRED)
  └─ NO ────────────────────────────────────────────►  PATTERN B   (fallback)
```

### Pattern A — full MPK scheduler + megakernel  (HIGH fidelity, PREFERRED)

Reuse or clone the matching `test_*_testmode.py` and run it through
`PersistentKernel(test_mode=True)`. This is the **highest-fidelity** check: it
exercises the EXACT code path the megakernel runs — scheduler dispatch, the
`task_register.cc` codegen snippet, `tma.cuh` descriptor creation, and the
single-GPU megakernel `nvcc` JIT. If the kernel maps to an existing layer,
ALWAYS use Pattern A — it is the only path that catches scheduler/codegen-level
crashes (which is precisely the Heisenbug class).

The canonical model is
`tests/runtime_python/blackwell/sm100_fp8_gemm_dense/test_fp8_gemm_dense_qkva_splitk_v2_testmode.py`
(the in-MPK test for ferret ws3's split-K kernel). The minimal example is
`tests/runtime_python/test_mode/test_rmsnorm_testmode.py`. Invoke the
`test-mode` skill in the MPK repo for the canonical authoring guide.

To clone for a new kernel, copy the matching `test_*_testmode.py`, swap the
`pk.<your_layer>(...)` call + shapes, and KEEP the two non-negotiable guards
below.

### Pattern B — CUDAExtension wrapper  (lower lift, no existing layer)

When the kernel has **no** existing MPK layer (a brand-new op the builder
doesn't call yet), fall back to a hand-written `__global__` that calls the
ferret kernel's `__device__ ... task_impl(...)` and builds the TMA descriptors
on the host the way `tma.cuh` would. This bypasses the scheduler/megakernel but
still runs the **real device code** (tcgen05, mbarrier protocol, swizzle math —
the part that crashes), so it still catches the device-side Heisenbug.

Cribbed from
`tests/runtime_python/blackwell/sm100_fp8_group_gemm_decode/{runtime_kernel_wrapper.cu,setup.py}`.

Files here: `runtime_kernel_wrapper.cu.tmpl` + `setup.py.tmpl`. Copy both into a
fresh dir, fill the `@@PLACEHOLDERS@@` (the wrapper header documents each), and
add a `test_<name>.py` driver (contract below). Then point `mpk_validate.sh` at
the `setup.py`.

---

## The two NON-NEGOTIABLE guards (apply to BOTH patterns)

These are what make the harness honest. A decode-gated FP8 GEMM that early-exits
to all-zero output is the textbook way a broken kernel *looks* like a pass.

### 1. Sentinel-fill guard (catches "early-exit to zero looks like a pass")

Pre-fill the output with a poison value so a kernel that never writes is
**visible** rather than masquerading as a (wrong) all-zero match. The poison
value MUST be **BF16-exact** — a power of two such as `-1024.0`. Do **NOT** use
`-987.0`: BF16 has 8 mantissa bits, so `-987.0` rounds to `-988.0` and an
exact-equality scan `== -987.0` then matches **zero** untouched rows, silently
defeating the guard (the count stays 0 even on a full no-write).

```python
SENTINEL = -1024.0  # BF16-exact (power of two); -987.0 rounds to -988.0 -> BAD
output = torch.full((M, N), SENTINEL, device="cuda", dtype=torch.bfloat16)
...
sentinel_rows = (output.float() == SENTINEL).all(dim=1).sum().item()
passed = cos > 0.99 and sentinel_rows == 0
print(f"... cos={cos:.6f}  sentinel_rows={sentinel_rows}  -> {'PASS' if passed else 'FAIL'}")
```

`mpk_validate.sh` independently re-greps `sentinel_rows=` and FAILS if any line
reports `> 0`, regardless of what the driver printed — so even a buggy driver
cannot hide a no-write.

### 2. Decode-gate drive guard (makes decode-gated kernels actually execute)

> **THIS IS THE GUARD THAT WAS WRONG.** The old advice — set
> `qo_indptr = arange(M+1)` via `meta_tensors` — does NOT work and silently
> FALSE-FAILs every decode-gated kernel. Use the request-state form below.

Decode-gated kernels (the SplitK family) read, in their `task_register.cc`
codegen snippet:

```c++
int q_len_       = qo_indptr_buffer[1] - qo_indptr_buffer[0];
if (q_len_ > 8) return;                                  // prefill -> skip
int active_rows_ = qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];
if (min(active_rows_, M) <= 0) return;                   // nothing -> skip
```

**`qo_indptr_buffer` is NOT a settable static input.** In MODE_OFFLINE
(test_mode runs MODE_OFFLINE) the runtime owns it through TWO writes that both
land AFTER you set `meta_tensors`:

1. `init_kernel` (`persistent_kernel.cuh`) **zeros** the whole
   `qo_indptr_buffer` at init time.
2. `prepare_next_batch` fires at the first `EVENT_END_OF_TASK_GRAPH`, *before*
   the first real task-graph iteration, and **rebuilds** `qo_indptr_buffer`
   from scratch out of the request scheduler state (`tokens.shape[0]` =>
   `total_num_requests`, plus `step`, `prompt_lengths`, `num_new_tokens`).

So `meta_tensors["qo_indptr_buffer"] = arange(M+1)` is discarded twice over —
the kernel sees whatever `prepare_next_batch` computed (with the test_mode
defaults: ONE prefill request of `q_len = M` => `q_len_ > 8` => early-exit, or
`active_rows = 0`). The output stays sentinel and you get a vacuous FALSE FAIL.

**The WORKING contract: drive the request state** so `prepare_next_batch`
emits **M single-token DECODE requests in one batch**. That makes it write
`qo_indptr_buffer = [0,1,2,...,M]` at execution time => `q_len=1 ≤ 8` (decode
gate passes) and `active_rows = M`:

```python
PAGE_SIZE = 128
M = ...                                  # compile-M == # decode rows under test
params["max_num_batched_requests"] = M   # all M requests in ONE batch
params["max_num_batched_tokens"]   = M
params["max_seq_length"]           = max(PAGE_SIZE * 2, M)
params["max_num_pages"]            = max(M, 4)   # 1 page/req at step=1; no wrap
params["page_size"]                = PAGE_SIZE

qo = torch.zeros(M + 1, dtype=torch.int32, device="cuda")  # read back after pk()
params["meta_tensors"] = {
    "qo_indptr_buffer": qo,
    # tokens.shape[0] == M  => total_num_requests = M
    "tokens":         torch.zeros(M, params["max_seq_length"],
                                  dtype=torch.int64, device="cuda"),
    "step":           torch.ones(M, dtype=torch.int32, device="cuda"),  # decode
    "prompt_lengths": torch.ones(M, dtype=torch.int32, device="cuda"),  # step>=plen
    "num_new_tokens": torch.ones(M, dtype=torch.int32, device="cuda"),  # 1 tok/req
}
```

**Then VERIFY it held at execution time** (the runtime uses the SAME `qo`
pointer, so reading it back after `pk()` shows what the kernel gated on):

```python
pk(); torch.cuda.synchronize()
qo_rt = qo.cpu().tolist()
q_len_rt, active_rows_rt = qo_rt[1] - qo_rt[0], qo_rt[-1]
gate_ok = (q_len_rt <= 8) and (active_rows_rt == M)   # expect q_len=1, rows=M
passed = cos > 0.99 and sentinel_rows == 0 and gate_ok
```

If `gate_ok` is False the kernel early-exited and the run is **vacuous** — that
is a HARNESS bug (mis-driven request state), NOT a kernel defect. Never report
a kernel FAIL from a vacuous run.

The canonical, working implementation of BOTH guards — AND the profiler path
below — is
`tests/runtime_python/blackwell/sm100_fp8_gemm_dense/test_fp8_gemm_dense_qkva_splitk_v2_testmode.py`
(study its `run()` — it drives the request state, reads `qo` back, profiles the
kernel, and runs a mediumm baseline for a ratio).

For Pattern B (no MPK scheduler, hand-rolled `__global__`): there is no
`prepare_next_batch`, so the gate inputs are whatever YOU pass as kernel args.
Pass the `q_len` / `active_rows` (or the equivalent `m_indices` / `qo_indptr`)
args directly as the decode-passing values (`q_len=1`, `active_rows=M`) — i.e.
the `arange`/explicit form is fine *only* in Pattern B, where nothing rewrites
it.

---

## PERFORMANCE — the in-MPK single-kernel latency (test mode's MAIN purpose)

Correctness is necessary but NOT sufficient: a kernel can be fast standalone
(ferret's dedicated-worker bench) yet slow in the shared-worker megakernel. The
faithful number is the kernel's **WALL-SPAN** inside the real MPK run, which
test mode exposes via the profiler. Make every test-mode driver report it; the
validator contract now REQUIRES a WALL-SPAN perf number alongside the
correctness gate.

### KERNEL-LATENCY METRIC = WALL-SPAN, NOT median (the bimodal-CTA pitfall)

The profiler CSV's per-task `duration_ns` is a **per-CTA** span. At decode these
kernels are **BIMODAL**: the kernel launches `grid_dim` (e.g. 128) CTAs sized
for the compile-time M=mbt, but only `ceil(active_rows * N / tile)` of them do
real work — the rest idle-exit in <1us (decode has active_rows=1). So the
MEDIAN duration_ns is an *idle CTA*: for the mediumm dense-GEMM, median ≈ 0.66us
while the real work takes ≈ 29us. Ranking kernels by median is therefore
GROSSLY wrong — split-K vs mediumm by median is ratio ≈ **0.06x** (it would
declare the FASTER kernel "16x slower").

The correct kernel latency is the **WALL-SPAN** = `max(end_ts) - min(begin_ts)`
over the task's events — wallclock from the first CTA starting to the last CTA
finishing. By WALL-SPAN: split-K = 22.27us, mediumm = 29.31us ⇒ ratio = **1.32x**
(split-K faster), which matches reality. `scripts/parse_profile.py --stat wall`
returns this (with 32-bit %globaltimer wrap correction); `--stat all` also
includes `wall_ns`/`wall_us`. **Drive every WIN/SLOWER decision off WALL-SPAN;
median/max are secondary characterization of the per-CTA work split only.**

### How a driver emits the trace (opt-in, before compile)

```python
device = "cuda"
# Absolutize the trace stem into the per-config compile_dir so the CSV is found
# regardless of the process cwd (mpk_validate.sh runs the driver from $MIRAGE_ROOT).
trace_stem = os.path.join(compile_dir, f"trace_{kernel}")
params["profiler_tensor"] = torch.zeros(3000 * 128, dtype=torch.uint64, device=device)
params["trace_name"]      = trace_stem        # writes trace_stem.csv / .perfetto-trace
# ... attach tensors, register the layer, pk.compile(output_dir=compile_dir) ...
pk(); torch.cuda.synchronize()                # CSV exists after this
```

The buffer MUST be `uint64` on CUDA; `3000*128` is the demo-conventional size
(2 entries per task event). See the MPK `test-mode` skill, section "Profiling".

### How a driver reads the WALL-SPAN back

Run `scripts/parse_profile.py <csv> <TASK_NAME> --stat wall` (JSON out with
`wall_ns`/`wall_us`/`count`), or `--stat all` (which adds `wall_ns`/`wall_us`
alongside `min_ns`/`max_ns`/`avg_ns`/`median_ns`). `TASK_NAME` is the TaskType
enum name as it appears in the CSV — e.g. `TASK_FP8_GEMM_DENSE_QKVA_SPLITK_SM100`
for the split-K candidate, `TASK_FP8_GEMM_DENSE_MEDIUMM_SM100` for the mediumm
baseline; `--list` enumerates what ran if you're unsure. Print a
machine-greppable line `mpk_validate.sh` scrapes (WALL_us is the latency metric;
median/max are secondary, do NOT rank on them):

```python
# PERF: kernel=<TASK_NAME> count=.. WALL_us=.. (median_us=.. max_us=.. avg_us=..)
```

### Report a RATIO, not just an absolute

To know whether the candidate actually beats what it replaces *in-MPK*, run the
BASELINE kernel through the SAME test-mode harness at the SAME shape (same
A/B/scales/reference, so the two WALL-SPANs are directly comparable) and print
(field names are `*_wall_us`; ratio = mediumm_wall/splitk_wall):

```python
# PERF_SUMMARY: splitk_wall_us=<f> mediumm_wall_us=<f> ratio=<f>   # >1 = splitk faster
```

`mpk_validate.sh` scrapes `PERF_SUMMARY:` (preferred — WALL-SPAN field names,
with back-compat fallback to the legacy `splitk_us`/`mediumm_us`) or the last
`PERF: WALL_us` and surfaces `perf_us=`/`baseline_us=`/`ratio=` (all WALL-SPAN)
in its verdict line. `perf_us=-` means the driver was not profiling-enabled —
fix it. Perf is reported ALONGSIDE correctness; cos+sentinel still gate
PASS/FAIL. The canonical driver
(`tests/runtime_python/blackwell/sm100_fp8_gemm_dense/test_fp8_gemm_dense_qkva_splitk_v2_testmode.py`)
runs the qkv_a shape through BOTH `kernel="splitk"` and `kernel="mediumm"` and
prints the WALL-SPAN `PERF_SUMMARY:` ratio — copy its structure.

---

## Pattern B driver contract

A Pattern B dir MUST ship a `test_<name>.py` (or `run.py`) that
`mpk_validate.sh` runs after `build_ext`. It must:

1. import the built module, build real FP8 tensors + per-block scales;
2. apply **both guards** above: a BF16-exact `-1024.0` sentinel-fill output,
   and — since Pattern B has no scheduler/prepare_next_batch — pass the
   decode-gate args (`q_len=1`, `active_rows=M`, or the `arange` `m_indices`/
   `qo_indptr`) DIRECTLY as kernel args (guard #2's Pattern-B note);
3. compute an FP32 reference and a cosine similarity;
4. print machine-greppable lines: `cos=<float>` and `sentinel_rows=<int>`,
   plus a `PASS`/`FAIL` token. (`mpk_validate.sh` keys on `cos=` / `sentinel_rows=`.)

---

## What `mpk_validate.sh` does

1. `cp $WS/kernel.cuh $MIRAGE_ROOT/include/.../tasks/<family>/<KERNEL_NAME>.cuh`
   (backing up any existing file first).
2. **GPU pick is torch-probe + exclusivity**, NOT just `nvidia-smi` mem% —
   MPK needs an exclusive GPU, and `nvidia-smi` can show a GPU "free" that then
   fails `torch.cuda.init()` with `cudaErrorDevicesUnavailable`. It ranks out
   GPUs hosting any compute process or with util ≥ 5%, then torch-probes the
   survivors and takes the first that actually initializes CUDA.
3. runs the test driver (Pattern A `.py` or Pattern B `setup.py`+driver),
   capturing the log.
4. parses the verdict: PASS iff **no crash/timeout** AND **no CUDA sentinel
   error** in the log AND **every `cos=` > 0.99** AND **`sentinel_rows=` == 0**
   on every line AND no `FAIL`/`Traceback`.
5. scrapes the **PERFORMANCE** WALL-SPAN number(s): a `PERF_SUMMARY:` line
   (`splitk_wall_us`/`mediumm_wall_us`/`ratio`, back-compat to legacy
   `splitk_us`/`mediumm_us`) if present, else the candidate's `PERF: WALL_us`.
   WALL-SPAN is the latency metric, NOT median (decode kernels are bimodal — the
   median is an idle CTA; see the PERFORMANCE section). Reported ALONGSIDE the
   verdict (does NOT change PASS/FAIL); a missing perf number is a WARNING.
6. **reverts the `.cuh` copy** (restores the backup) so the MPK tree is never
   left dirty — this is a *validator*, not an integrator. `--keep-on-pass`
   overrides on success; `--no-revert` keeps it regardless.

Verdict line (always emitted, exit 0=PASS / 1=FAIL / 2=harness-error;
`perf_us`/`baseline_us`/`ratio` are `-` if the driver was not profiling-enabled):

```
MPK_VALIDATE: <PASS|FAIL> kernel=<name> gpu=<N> cos=<x> sentinel_rows=<n> \
    perf_us=<f|-> baseline_us=<f|-> ratio=<f|-> reason=<...>
```
