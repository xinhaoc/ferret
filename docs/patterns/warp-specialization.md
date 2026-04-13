# Warp Specialization

Assigning different warps to different roles (producer vs consumer) to overlap memory and compute.

## Why Warp Specialization Works

Without specialization: all threads do `load → barrier → compute → barrier → load`. The barriers serialize memory and compute — total time = load_time + compute_time.

With specialization: producer warps load stage N+1 while consumer warps compute on stage N. No barrier between them — they run in parallel. Total time ≈ max(load_time, compute_time).

**The fundamental win**: memory latency hidden by compute, compute latency hidden by memory. On a well-balanced kernel, this nearly doubles throughput.

## Why Pingpong (Multiple Consumers) Works

With 1 consumer warpgroup: after WGMMA finishes, the consumer waits for the next tile to arrive. Tensor cores are idle during this wait.

With 2 consumers alternating: while Consumer0 waits for its next data, Consumer1 computes on its data. The tensor cores are never idle — there's always a consumer ready to issue the next MMA.

**The win**: tensor core utilization approaches 100%. The cost: more threads → fewer registers per consumer → may limit tile size.

## Why Register Redistribution Matters

SM90 has 64K registers per SM, shared by ALL warps. Without redistribution, producer warps hold ~128 regs each — wasted on TMA-issuing code that barely uses them.

With `setmaxnreg`: producer drops to 24 regs, freed registers go to consumers (240 regs). Enables larger MMA accumulator tiles → better compute/memory ratio → higher throughput.

Example: 2 WGs on Hopper
- Without redistribution: each WG gets ~128 regs. Consumer can't hold a 64×256 accumulator.
- With redistribution: producer=24, consumer=240. Consumer holds a much larger accumulator tile.

## Why Blackwell's TMEM Changes Everything

On Hopper: 128 threads × 240 regs = 30,720 registers locked per consumer warpgroup just for MMA accumulators. With 2 consumers + 1 producer = 3 warpgroups, most of the SM's register file is consumed.

On Blackwell: TMEM (256KB, separate hardware memory) holds accumulators. Consumer warps don't need registers for MMA results at all. This frees the register file for:
- More pipeline stages (more live buffers)
- Larger epilogue tiles
- More concurrent warps (higher occupancy)
- The MMA itself needs only 1 thread to issue — 31 threads in the MMA warp are free for other work

---

## Hopper (SM90): Warpgroup is the Unit

### Warpgroup = 4 warps = 128 threads (ALWAYS)

From `cutlass-4.4.2/include/cutlass/cutlass.h`:
```cpp
static const int NumThreadsPerWarpGroup = 128;
static const int NumWarpsPerWarpGroup = 4;
```

This is a **hardware requirement**. `wgmma.mma_async.sync.aligned` requires all 128 threads to participate synchronously. You cannot have 2 warps doing MMA — it's always 4.

### WGMMA thread model

From `cutlass-4.4.2/include/cute/arch/mma_sm90_gmma.hpp`:

```cpp
// All 128 threads must execute these synchronously:
wgmma.fence.sync.aligned       // fence
wgmma.mma_async.sync.aligned   // MMA
wgmma.commit_group.sync.aligned // commit
wgmma.wait_group.sync.aligned   // wait
```

The accumulator registers are **distributed across all 128 threads**. For `MMA_64x8x16_F16`, each of 128 threads holds 2 registers → collectively 64×8 output tile. The consumer warpgroup needs 160-256 registers per thread for these accumulators.

### Typical Hopper configurations

| Config | Warps | Producer | Consumer | Producer Regs | Consumer Regs |
|---|---|---|---|---|---|
| 1 prod + 1 cons | 8 (2 WGs) | WG0 (4 warps) | WG1 (4 warps) | 24-56 | 232-256 |
| 1 prod + 2 cons (pingpong) | 12 (3 WGs) | WG0 (4 warps) | WG1, WG2 (4 warps each) | 24-40 | 232-240 |
| 1 prod + 3 cons | 16 (4 WGs) | WG0 (4 warps) | WG1-3 (4 warps each) | 32 | 160 |

**You cannot break the 4-warp warpgroup boundary.** Asymmetric configs like "2 warps producer, 6 warps consumer" are impossible because WGMMA requires exactly 128 threads.

From CUTLASS pingpong GEMM `sm90_gemm_tma_warpspecialized_pingpong.hpp`:
```cpp
static_assert(NumMMAThreads == 128, "Pingpong kernel must have TiledMMA operating using 128 threads.");
static_assert(MaxThreadsPerBlock == 384, "Pingpong kernel must have 384 threads in total.");
```

## Blackwell (SM100): Warp is the Unit

### tcgen05 MMA: issued by a SINGLE thread

From `cutlass-4.4.2/include/cute/arch/mma_sm100_umma.hpp`:

```cpp
// Only the elected thread issues the MMA
if (cute::elect_one_sync()) {
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;"
        ...
    );
}
```

**The minimum thread count for a tcgen05 MMA is 1 thread.** The instruction is NOT a cooperative warp/warpgroup operation. One elected thread issues the command, the tensor core hardware handles execution asynchronously.

### Accumulator goes to TMEM, NOT registers

From `cute/arch/mma_sm100_umma.hpp`:

```cpp
// SM90 (Hopper): accumulator in REGISTERS
using CRegisters = uint32_t[2];  // each of 128 threads holds 2 regs

// SM100 (Blackwell): accumulator in TMEM
using DRegisters = void;         // no register output
using CRegisters = uint32_t[1];  // TMEM address, not data
```

TMEM = dedicated on-chip memory (128 rows × 512 columns × 32-bit, from `tmem_allocator_sm100.hpp`):
```cpp
using MAX_CAPACITY_BITS = Int<128*512*32>;  // 256 KB per SM
```

This eliminates register pressure on consumer warps. On Hopper, consumers needed 224+ regs for accumulators. On Blackwell, accumulators live in TMEM.

### CUTLASS SM100: Per-Warp Roles (not per-warpgroup)

From `cutlass-4.4.2/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp`:

```cpp
// Each role = 1 WARP (32 threads), NOT 1 warpgroup
static constexpr uint32_t NumSchedThreads        = NumThreadsPerWarp; // 32
static constexpr uint32_t NumMMAThreads          = NumThreadsPerWarp; // 32
static constexpr uint32_t NumMainloopLoadThreads = NumThreadsPerWarp; // 32
static constexpr uint32_t NumEpilogueLoadThreads = NumThreadsPerWarp; // 32

enum class WarpCategory : int32_t {
    MMA          = 0,  // 1 warp: issues tcgen05.mma
    Sched        = 1,  // 1 warp: tile scheduling
    MainloopLoad = 2,  // 1 warp: TMA loads
    EpilogueLoad = 3,  // 1 warp: epilogue data movement
    Epilogue     = 4   // N warps: TMEM→registers→global store
};

// Role by warp index
int warp_idx = canonical_warp_idx_sync();
WarpCategory warp_category = warp_idx < static_cast<int>(WarpCategory::Epilogue)
    ? WarpCategory(warp_idx) : WarpCategory::Epilogue;
```

**Fundamental departure from Hopper**: 4 specialized roles each get 1 warp, not 4 warpgroups. The MMA warp only needs 1 thread to issue `tcgen05.mma`; the other 31 threads in that warp are idle during MMA issue.

### cta_group::1 vs cta_group::2 (1SM vs 2SM MMA)

**cta_group::1** — single CTA/SM:
- M = 64 or 128, N = 8-256
- One CTA owns the TMEM allocation

**cta_group::2** — two CTAs cooperate across 2 SMs:
- M = 128 or 256 (doubles the M dimension)
- Leader CTA issues the instruction
- Hardware reads A from leader's SMEM, B from both CTAs' SMEM via DSMEM

```cpp
// Single SM
asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, ...");

// Two SMs cooperating
asm volatile("tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, ...");
```

### ThunderKittens Blackwell (real examples)

From `thunderkittens-main/kernels/gemm/bf16_b200/` educational levels:

**Level 07** — 4 warps total (128 threads), warp-specialized:
- Warp 0: TMA loader (1 thread issues TMA)
- Warp 1: MMA issuer (1 thread issues tcgen05.mma)
- Warps 2-3: Epilogue (TMEM → registers → store)

**Level 09** — 2-CTA cluster, 12 warps (3 warpgroups):
- Warpgroup 2: Producer (TMA + MMA issuers, register decrease to 56)
- Warpgroups 0-1: Consumers (epilogue, register increase to 224, read from TMEM)

```cpp
static constexpr int NUM_CONSUMERS = 2;
static constexpr int NUM_WARPS = (NUM_CONSUMERS + 1) * 4;  // 12 warps
```

## Hopper Role Assignment Examples

### FlashAttention

From `flash-attention-fa4-v4.0.0.beta8/hopper/flash_fwd_kernel_sm90.h`:

```cpp
int warp_group_idx = cutlass::canonical_warp_group_idx();

// Register redistribution
static constexpr uint32_t LoadRegisterRequirement = Use_TMA_KV ? 24 : 40;
static constexpr uint32_t MmaRegisterRequirement = Use_TMA_KV ? 240 : 232;

if (warp_group_idx == 0) {  // Producer
    cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
    // TMA load loop...
} else {  // Consumer (1-3 warpgroups)
    cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
    // WGMMA compute loop...
}
```

### DeepGemm

From `deepgemm-2.1.1.post3/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh`:

```cpp
__global__ __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)

if (warp_idx >= kNumMathThreads / 32) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if (warp_idx == kNumMathThreads/32 && elect_one_sync()) {
        // Only 1 thread issues TMA loads
    }
} else {
    cutlass::arch::warpgroup_reg_alloc<240>();
    // WGMMA loop...
}
```

## Pingpong Scheduling (Multiple Consumers)

When 2+ consumer warpgroups exist, they alternate to avoid shared memory contention.

### FlashAttention named barrier pingpong

From `flash-attention-fa4-v4.0.0.beta8/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`:

```cpp
enum class FwdNamedBarriers {
    WarpSchedulerWG1 = 1,  // WG1's turn
    WarpSchedulerWG2 = 2,  // WG2's turn
};

// Each WG waits on its OWN barrier, signals the NEXT WG's barrier
void warp_scheduler_barrier_sync() {
    NamedBarrier::sync(2 * NumThreadsPerWarpGroup,
        FwdNamedBarriers::WarpSchedulerWG1 - 1 + canonical_warp_group_idx());
}
void warp_scheduler_barrier_arrive() {
    int cur = canonical_warp_group_idx() - 1;
    int next = (cur < NumMmaWarpGroups - 1) ? cur + 1 : 0;
    NamedBarrier::arrive(2 * NumThreadsPerWarpGroup,
        FwdNamedBarriers::WarpSchedulerWG1 + next);
}

// Usage in inner loop:
warp_scheduler_barrier_sync();     // wait for my turn
flash::gemm<true, -1>(...);       // QK GEMM (async)
flash::gemm<false, -1>(...);      // PV GEMM (async)
warp_scheduler_barrier_arrive();   // signal next WG
```

## The Fundamental Shift: Hopper → Blackwell

| Aspect | Hopper (SM90) | Blackwell (SM100) |
|---|---|---|
| **Specialization unit** | Warpgroup (4 warps = 128 threads) | Warp (1 warp = 32 threads) |
| **MMA thread requirement** | All 128 threads synchronously | 1 elected thread |
| **Accumulator storage** | Distributed across 128 thread registers | TMEM (dedicated hardware memory) |
| **Consumer register pressure** | High (224-256 regs for accumulators) | Low (TMEM holds accumulators) |
| **Producer register pressure** | Low (24-48 regs) | Low (24-48 regs) |
| **Asymmetric warp allocation** | No (must be multiples of 4 warps) | Yes (any warp count per role) |
| **MMA pipeline commit** | Thread-based (`wgmma.commit_group`) | Hardware-based (`tcgen05.commit`) |
| **Warp roles** | Producer WG + Consumer WG(s) | MMA warp + Sched warp + Load warp + Epilogue warp(s) |
| **Flexibility** | Low (warpgroup boundary fixed) | High (per-warp role assignment) |
