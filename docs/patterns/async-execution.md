# Async Execution Model for Hopper/Blackwell Kernels

## Evolution: Ampere → Hopper → Blackwell

### Ampere (SM80): cp.async + commit/wait groups
Thread-scoped. Each thread issues copies, commits into groups, waits for groups.
```
cp.async.ca.shared.global [smem], [gmem], 16;
cp.async.commit_group;
cp.async.wait_group N;
```

### Hopper (SM90): TMA + mbarrier + wgmma
Three major additions:
1. **TMA**: Hardware copies entire tiles. One thread issues, hardware does the rest. No registers for data.
2. **Transaction barriers**: mbarrier tracks both thread arrivals AND byte counts. TMA auto-signals on completion.
3. **WGMMA**: Async warpgroup MMA reading directly from shared memory.

### Blackwell (SM100): tcgen05 + TMEM
- **tcgen05**: Replaces wgmma. Results go to Tensor Memory (TMEM), not registers.
- **tcgen05.commit.mbarrier::arrive**: MMA hardware directly signals barriers on completion.

## The Producer-Consumer Model

### Two Barriers Per Pipeline Stage

Each stage has TWO barriers:

| Barrier | Type | Who signals | Who waits | Meaning |
|---|---|---|---|---|
| **FullBarrier** | `ClusterTransactionBarrier` | Producer (via TMA hardware) | Consumer | "Data is ready" |
| **EmptyBarrier** | `ClusterBarrier` | Consumer | Producer | "Buffer is free" |

From CUTLASS `PipelineTmaAsync`:
```cpp
struct SharedStorage {
    FullBarrier full_barrier_[Stages];
    EmptyBarrier empty_barrier_[Stages];
};
```

### The Cycle

```
PRODUCER                              CONSUMER
1. producer_acquire(stage)
   - Wait on empty_barrier[stage]
   - arrive_and_expect_tx(bytes)
     on full_barrier[stage]

2. Issue TMA load → smem[stage]
   (TMA hardware auto-signals
    full_barrier on completion)

3. Advance to next stage            1. consumer_wait(stage)
                                       - Wait on full_barrier[stage]

                                    2. Compute on smem[stage]

                                    3. consumer_release(stage)
                                       - Arrive on empty_barrier[stage]
```

### Phase Tracking

Each barrier has a phase bit (0 or 1) that flips each cycle. The `PipelineState` tracks stage index + phase:

```cpp
void operator++() {
    ++index_;
    if (index_ == Stages) {
        index_ = 0;
        phase_ ^= 1;  // flip phase when wrapping
    }
}
```

**Critical**: Producer starts with **phase = 1** (opposite to consumer's initial phase = 0), because buffers are initially empty.

## Barrier Primitives

### ClusterBarrier (EmptyBarrier)

```
mbarrier.init.shared::cta.b64 [addr], arrive_count;
mbarrier.arrive.shared::cta.b64 _, [addr];
// Wait (spin):
LAB_WAIT:
  mbarrier.try_wait.parity.shared::cta.b64 P1, [addr], phase, ticks;
  @P1 bra DONE;
  bra LAB_WAIT;
DONE:
```

### ClusterTransactionBarrier (FullBarrier)

Adds transaction-byte tracking for TMA:

```
// Set expected bytes + arrive (one leader thread):
mbarrier.arrive.expect_tx.shared::cta.b64 _, [addr], transaction_bytes;

// Set expected bytes without arriving:
mbarrier.expect_tx.shared::cta.b64 [addr], transaction_bytes;
```

The barrier flips phase only when BOTH:
- All thread arrivals completed
- All expected transaction bytes completed

### Barrier Lifecycle (One Pipeline Stage)

```
1. INIT (once at kernel start):
   mbarrier.init full_barrier[stage], producer_arrive_count
   mbarrier.init empty_barrier[stage], consumer_arrive_count
   fence.mbarrier_init.release.cluster
   __syncthreads()

2. PRODUCER ACQUIRE:
   empty_barrier[stage].wait(phase)           // buffer free
   full_barrier[stage].arrive_and_expect_tx(bytes)  // set expected bytes

3. TMA LOAD:
   cp.async.bulk.tensor ... , [full_barrier_addr]   // TMA signals barrier on done

4. CONSUMER WAIT:
   full_barrier[stage].wait(phase)            // data ready

5. CONSUMER COMPUTE: (read from smem[stage])

6. CONSUMER RELEASE:
   empty_barrier[stage].arrive()              // buffer free for reuse
```

## TMA Integration

### How TMA Signals Barriers

The producer passes the full_barrier address to TMA. Hardware:
1. Reads from global memory
2. Writes to shared memory
3. **Automatically signals** full_barrier by completing expected bytes

```cpp
// FlashAttention pattern:
copy(params.tma_load_K.with(
    *pipeline_k.producer_get_barrier(smem_pipe_write),  // barrier addr
    mcast_mask_kv,
    TMA::CacheHintSm90::EVICT_LAST
), tKgK_TMA(_, n_block), tKsK_TMA(_, smem_pipe_write.index()));
```

PTX generated:
```
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    [smem_addr], [tensor_desc, coords], [mbar_addr], cta_mask;
```

### Transaction Bytes Must Be Exact

```cpp
static constexpr uint32_t TmaTransactionBytesK =
    static_cast<uint32_t>(size(take<0,2>(SmemLayoutK{})) * sizeof_bits_v<Element> / 8);
```

- Too many bytes → barrier never flips → **deadlock**
- Too few bytes → barrier flips early → **data corruption**

### cp.async Fallback (Non-TMA)

When TMA can't be used (e.g., paged KV cache):
```cpp
pipeline_k.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
// PTX: cp.async.mbarrier.arrive.shared::cta.b64 [barrier_addr];
```

## WGMMA Fence/Commit/Wait

WGMMA is asynchronous — issued, committed, and waited on separately.

### Three Key Instructions

```cpp
warpgroup_arrive();         // wgmma.fence.sync.aligned — shared mem visible to wgmma
// ... issue wgmma.mma_async instructions ...
warpgroup_commit_batch();   // wgmma.commit_group.sync.aligned — commit to a group
warpgroup_wait<N>();        // wgmma.wait_group.sync.aligned N — wait until ≤N groups remain
```

### Compiler Fence (Critical)

```cpp
warpgroup_fence_operand(accum);  // asm volatile("" : "+f"(reg) :: "memory")
```

Prevents compiler from reordering register reads before wgmma completes. **Must bracket every wgmma sequence.**

### Canonical Sequence

```cpp
warpgroup_fence_operand(accum);     // compiler: don't move accum reads before here
warpgroup_arrive();                  // fence: shared memory ready
cute::gemm(tiled_mma, A, B, accum); // issue wgmma instructions
warpgroup_commit_batch();            // commit group
warpgroup_wait<0>();                 // wait for completion
warpgroup_fence_operand(accum);      // compiler: accum now safe to read
```

### Overlapping with `wg_wait=-1`

FlashAttention defers waits to overlap compute:
```cpp
// Start QK gemm, don't wait:
flash::gemm<true, -1>(tiled_mma_qk, Q, K, S);   // wg_wait=-1

// Start PV gemm while QK is in flight:
flash::gemm<false, -1>(tiled_mma_pv, P, V, O);

warpgroup_wait<1>();  // wait for QK (1 group = PV still in flight)
// ... softmax on S ...
warpgroup_wait<0>();  // now wait for PV
```

## Warp Specialization

### Thread Role Assignment

```cpp
if (warp_group_idx == 0) {  // PRODUCER
    cutlass::arch::warpgroup_reg_dealloc<24>();   // give up registers
    // TMA load loop
} else {  // CONSUMER
    cutlass::arch::warpgroup_reg_alloc<240>();    // claim registers
    // MMA compute loop
}
```

### Register Redistribution

Hopper can redistribute registers at runtime:
- Producer warpgroup: ~24-40 registers (only needs to issue TMA commands)
- Consumer warpgroups: up to 240 registers (large accumulator tiles)

This is done via:
```
setmaxnreg.inc.sync.aligned.u32 240;  // consumer claims
setmaxnreg.dec.sync.aligned.u32 24;   // producer releases
```

### Named Barriers for Cross-Warpgroup Coordination

FlashAttention uses named barriers beyond the pipeline:
```cpp
enum class FwdNamedBarriers {
    QueryEmpty = 0,       // Q smem free to overwrite
    WarpSchedulerWG1 = 1, // round-robin between consumer warpgroups
    WarpSchedulerWG2 = 2,
    PFull = 6,            // P written to smem (for SS MmaPV)
    PEmpty = 7,           // P smem buffer free
};
```

## Pipeline Stages (Multi-Stage Buffering)

With N stages, N-1 loads overlap with 1 compute:

```
Stage 0: [LOAD 0] → [COMPUTE 0] → [LOAD N] → ...
Stage 1:            [LOAD 1]    → [COMPUTE 1] → ...
Stage 2:                         [LOAD 2]    → ...
```

Typical values: 2-4 stages, limited by shared memory capacity.

## Blackwell Extensions

### tcgen05.commit to mbarrier

MMA hardware directly signals barriers:
```
tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [barrier_addr];
```

Eliminates `wgmma.wait_group` + explicit arrive.

### 2×1SM MMA

Two adjacent SMs cooperate on one MMA. Requires cross-SM barrier coordination — always signal on the leader SM (even-numbered).

## Memory Fences

| Fence | PTX | When to use |
|---|---|---|
| `fence_barrier_init` | `fence.mbarrier_init.release.cluster` | After barrier init, before any use |
| `fence_view_async_shared` | `fence.proxy.async.shared::cta` | After async write to smem (STSM, TMA), before reading |
| `fence_proxy_async_global` | `fence.proxy.async.global` | Before TMA store |

## Common Pitfalls

1. **Wrong transaction bytes → deadlock or corruption**. Must exactly match `size(smem_tile) * sizeof(element)`.

2. **Missing fence_barrier_init**. After `mbarrier.init`, MUST call `fence.mbarrier_init.release.cluster` + sync.

3. **Missing fence_view_async_shared**. After writing smem via STSM/TMA, MUST fence before GMMA reads.

4. **Multiple leaders calling arrive_and_expect_tx**. Only ONE thread should call. Multiple → double arrive → premature barrier flip.

5. **Consumer release before warpgroup_wait**.
   ```
   // WRONG:
   pipeline.consumer_release(state);  // producer overwrites smem
   warpgroup_wait<0>();               // wgmma still reading!

   // CORRECT:
   warpgroup_wait<0>();               // wgmma done
   pipeline.consumer_release(state);  // now safe
   ```

6. **Missing warpgroup_fence_operand**. Without it, compiler may read accumulator registers before wgmma completes.

7. **Phase bit confusion**. Wrong phase = instant return (stale data) or infinite wait (deadlock). Phase = `(total_iterations / Stages) % 2`.

8. **Producer exits before consumer finishes**. Producer must call `producer_tail()` — wait for all stages to be released.

## CUTLASS Pipeline Classes

| Class | Full Barrier | Empty Barrier | Use |
|---|---|---|---|
| `PipelineTmaAsync` | TransactionBarrier | ClusterBarrier | TMA loads (Hopper) |
| `PipelineAsync` | ClusterBarrier | ClusterBarrier | cp.async loads |
| `PipelineTransactionAsync` | TransactionBarrier | ClusterBarrier | Non-TMA with tx tracking |
| `PipelineTmaStore` | — | — | TMA stores (commit/wait groups) |
| `PipelineUmmaAsync` | TransactionBarrier | ClusterBarrier | tcgen05 MMA (Blackwell) |
