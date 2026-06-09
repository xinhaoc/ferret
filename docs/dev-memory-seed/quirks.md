# quirks.md — library / cluster footguns (committed generic template)

> This is the generic committed seed. The live runtime copy at
> `docs/dev-memory/quirks.md` is gitignored, bootstrapped from this file,
> and appended to by the `memory-keeper` subagent. Entries here are the
> machine-agnostic, durably-useful footguns; keep host paths, usernames,
> device UUIDs, and PCIe IDs out (use `$MIRAGE_ROOT` / `$FERRET_ROOT` /
> `$USER` placeholders).

Things that are true *now* but may change with library upgrades. The
`memory-keeper` subagent is responsible for updating entries here when a
new ferret run discovers a new footgun. Append-only; conflicts add a
fresh `Updated YYYY-MM-DD:` block underneath.

- DeepGEMM grouped-FP8 baselines written for SM90 do NOT transfer to B200
  (SM100) unchanged. An SM90-style call (`disable_ue8m0_cast=True` + float
  scales + unpadded `m_indices`) is wrong on SM100: B200 DeepGEMM requires
  `disable_ue8m0_cast=False` and per-expert M padded to 128 (the "padded
  layout"). The SM100-correct grouped reference is padded-128/expert +
  ue8m0, full API per iteration with a 128 MB L2 flush and a median over
  ~100 iters.

- DeepGEMM's SM100 FP8 GEMM path can emit NaN output when its JIT ue8m0
  scale-pack kernel is broken (e.g. a `_C.so` built against the wrong
  CPython ABI). BF16 GEMM output stays correct. Because GEMM timing is
  data-independent, such a build is still valid as a *performance*
  reference even while its FP8 numerics are garbage — just don't trust it
  for correctness. Build DeepGEMM against the same Python/torch ABI you
  run it with.

- When using DeepGEMM as a perf reference for grouped FP8 GEMM on B200,
  the full API includes a per-call scale transform (order ~30–40 us for
  the DSv3 gate_up / down shapes) that ferret kernels pre-pack on host
  before the timed region — this matches Mirage's intended usage
  (`transpose_scale_sm100` is a separate task). Compare GEMM-only-vs-GEMM
  when isolating kernel speed, and full-API-vs-full-API when checking the
  task contract.

- B200 node-wide CUDA fault: when a single GPU falls off the bus,
  `nvidia-smi` reports `Unable to determine the device handle for GPU<N>:
  <pci>: Unknown Error`, and then a trivial `cudaMalloc` (128 MB) returns
  `cudaErrorUnknown` on EVERY visible GPU — even clean, idle ones — because
  one GPU falling off poisons driver/NVML state for the whole node.
  Symptom in ferret kernels: `CUDA(flush):unknown error` at the first
  `cudaMalloc`/`cudaMemset`. This is NOT a kernel bug and is NOT fixable by
  editing `kernel.cu`; it needs an admin GPU/driver reset
  (`nvidia-smi --gpu-reset` on the faulted GPU, or a node reboot).
  Detection: build a 5-line program with `cudaMalloc(&p, 1<<27)` +
  `cudaMemset` and check for `unknown error`, or grep
  `nvidia-smi --query-gpu=index,memory.free --format=csv,noheader` for the
  "Unable to determine device handle" message. Wait and retry the minimal
  malloc test before assuming a kernel regression. Note: `/tmp` is shared
  across users' containers on a multi-tenant host — use a unique filename
  (e.g. `/tmp/$USER_test.cu`) to avoid collisions.

- B200 SM100a, fp8 dense GEMM at the qkv_a decode shape
  (M=128, N=2176, K=7168) with pipeline NS=3: the bare split-K MMA latency
  (reduction stripped) is roughly FLAT at ~32us regardless of K-tiles per
  CTA (measured 14/28/56 tiles at SPLIT_K=8/4/2 => all ~31–34us in
  throughput mode); mediumm@NS=3 = ~32.8us. CONSEQUENCE: splitting K does
  NOT reduce compute latency at this shape — the GEMM is bound by a fixed
  ~32us cost, not the serial K-loop, so the premise "mediumm serializes
  K=7168, split cuts it 2x/4x" does NOT hold at NS=3. An internal split-K
  kernel with a *correct* reduction (exclusive FP32 partials + last-arriver
  read-back) therefore caps at ~1.00x vs mediumm@NS=3 even with zero
  reduction overhead; the larger speedups some tasks quote came from a
  forbidden fire-and-forget `red.add.bf16x2` epilogue (no read-back tail)
  and/or NS=5. The reduction read-back must be COALESCED (column-major
  partial layout `n*BM+et`, NOT row-major `et*BN+n`) — row-major layout
  cost an early split-K ~36us(S2)/88us(S4) of uncoalesced reduction;
  column-major cut the S4 reduction ~3x.

- 2026-06-04 — **MPK megakernel TMEM co-residency crash (GENERAL, framework-level).** Any kernel using `tcgen05.alloc` that lands in Mirage's persistent megakernel co-resides on-SM with other tcgen05 tasks; at DSv3 decode specifically with MLA-TP-decode, which allocs the FULL 512-col TMEM pool (D_V=512, no relinquish, held alloc→dealloc — verified mla_mtp_decode_tp{2,4,8}_sm100.cuh). The SM has 512 TMEM cols total. STANDALONE ferret benchmarks CANNOT see this (GPU to itself, no co-resident MLA-TP), so a kernel that is correct+fast standalone can still IMA/rc=255 at MULTI-RANK (TP>=2) DECODE. Root-caused 2026-06-04 (mirage workflow splitk-crash-rootcause-compare): every FP8 split-K crashed multi-rank; BF16 split-K (linear_sm100_mpk.cuh SplitK=true) did NOT. Two differentiators: (1) FP8 used CROSS-WARP tcgen05 alloc(warp2)/dealloc(warp0) — violates the CuTe same-warp permit invariant, drifts under ITS+scheduler jitter → corrupt/zero taddr → IMA; BF16 is warp0/warp0 SAME-WARP. (2) FP8 TMEM footprint = 256 cols (TCA=NE*BN, N baked into accumulator) vs BF16 = 32 cols (MMA_N=16, N tiled by the GRID, TMEM N-independent); 256 co-resident with MLA-TP's 512 widens the crash window. RULES for any MPK tcgen05 kernel: (a) alloc AND dealloc from the SAME warp, warp-uniform (the MMA warp may differ — only the permit must be same-warp); (b) minimize TMEM cols via grid-tiled-N (MMA_N small) not N-baked accumulator; (c) the crash is co-residency-only — flag it for the Mirage main agent to validate at multi-rank decode, NOT just standalone. NOTE: mediumm is ALSO cross-warp + works, but it is SINGLE-CTA-per-tile — cross-warp only bites under split-K's multi-CTA concurrency + the 512-col co-residency. Source: mirage experiment_history INDEX/journal 2026-06-04, workflow wf_3bb6586f.
