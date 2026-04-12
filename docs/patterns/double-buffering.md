# Software Pipelining (Double Buffering)

## The Problem

Without pipelining, compute waits for memory and memory waits for compute — they're serialized. Pipelining overlaps them: while computing on stage N, load stage N+1.

## Ampere: cp.async Pipeline (3-stage)

From `cutlass-4.4.2/examples/cute/tutorial/sgemm_sm80.cu`:

### Setup: 3 shared memory buffers

```cpp
auto bP = Int<3>{};  // 3 pipeline stages
auto sA = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));  // (BLK_M, BLK_K, PIPE=3)
```

### Prefill: load stages 0 and 1 before the loop

```cpp
for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));  // cp.async global→shared
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
    cp_async_fence();          // cp.async.commit_group
    --k_tile_count;
    if (k_tile_count > 0) { ++k_tile_next; }
}
```

### Mainloop: two levels of pipelining

```cpp
int smem_pipe_read = 0;
int smem_pipe_write = K_PIPE_MAX-1;

while (k_tile_count > -(K_PIPE_MAX-1)) {
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        if (k_block == K_BLOCK_MAX - 1) {
            // Advance read pointer, wait for next stage
            tXsA_p = tXsA(_,_,_,smem_pipe_read);
            cp_async_wait<K_PIPE_MAX-2>();   // wait until ≤1 group in flight
            __syncthreads();
        }

        // Register pipeline: load next k_block from shared→regs (ldmatrix)
        copy(s2r_atom, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));

        // Issue NEW cp.async for next tile on first k_block
        if (k_block == 0) {
            copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
            cp_async_fence();                // cp.async.commit_group
            smem_pipe_write = smem_pipe_read;
            smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1;
        }

        // Compute: MMA on current k_block
        gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
    }
}
```

**Two levels**: (1) SMEM pipeline: 3 stages via `cp_async_fence/wait_group`. (2) Register pipeline: within each stage, ldmatrix for next sub-tile overlaps with MMA on current sub-tile.

## Hopper: TMA + mbarrier Pipeline

From `cutlass-4.4.2/include/cutlass/pipeline/sm90_pipeline.hpp` (PipelineTmaAsync):

### Key difference from cp.async

cp.async uses ordered groups (`commit_group` / `wait_group<N>`) — thread-local, same threads load and compute.

TMA uses **per-stage mbarrier pairs** — separate warp groups for producer (TMA loads) and consumer (MMA compute). TMA hardware auto-signals the barrier on completion.

```cpp
struct SharedStorage {
    FullBarrier full_barrier_[Stages];    // "data is ready" (transaction-tracking mbarrier)
    EmptyBarrier empty_barrier_[Stages];  // "buffer is free" (regular mbarrier)
};
```

### Producer: acquire → issue TMA → (TMA auto-commits)

```cpp
void producer_acquire(uint32_t stage, uint32_t phase) {
    empty_barrier_ptr_[stage].wait(phase);                          // wait: buffer free
    if (params_.is_leader) {
        full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes);  // set expected bytes
    }
}
// TMA load issued here — hardware signals full_barrier on completion
// producer_commit is a NOP for TMA (hardware does it)
```

### Consumer: wait → compute → release

```cpp
void consumer_wait(uint32_t stage, uint32_t phase) {
    full_barrier_ptr_[stage].wait(phase);    // wait: data arrived
}

void consumer_release(uint32_t stage) {
    empty_barrier_ptr_[stage].arrive(...);   // signal: buffer free
}
```

### Phase tracking

```cpp
struct PipelineState {
    int index_ = 0;
    uint32_t phase_ = 0;
    void operator++() {
        ++index_;
        if (index_ == Stages) { index_ = 0; phase_ ^= 1; }  // phase flips on wrap
    }
};
```

## FlashAttention: Overlapped QK/PV GEMMs (2-stage)

From `flash-attention-fa4-v4.0.0.beta4/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`:

FlashAttention uses `kStages=2` with separate K and V pipelines. The key innovation is **IntraWGOverlap**: the consumer interleaves QK and PV GEMMs across iterations.

### Producer (WG0): TMA loads K and V

```cpp
auto load_K = [&](int n_block, auto const& smem_pipe_write, ...) {
    pipeline_k.producer_acquire(smem_pipe_write);
    copy(params.tma_load_K.with(
        *pipeline_k.producer_get_barrier(smem_pipe_write),  // TMA signals this mbarrier
        mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
    tKgK_TMA(_, n_block_idx), tKsK_TMA(_, smem_pipe_write.index()));
};
// Similar for load_V
```

### Consumer (WG1-3): Overlapped fwd_step

```cpp
auto fwd_step = [&](int n_block, auto mask_fn, ...) {
    ++smem_pipe_read;

    // Wait for K, issue QK GEMM (don't wait for result: wg_wait=-1)
    consumer_wait(pipeline_k, smem_pipe_read);
    flash::gemm<true, -1>(tiled_mma_qk, tSrQ, tSrK(...), tSrS);

    // While QK is in flight, wait for V, issue PV GEMM for PREVIOUS iteration
    consumer_wait(pipeline_v, smem_pipe_read_v);
    flash::gemm<false, -1>(tiled_mma_pv, tOrP, tOrV(...), tOrO);

    // Now wait for QK (warpgroup_wait<1> = 1 group still in flight = PV)
    warpgroup_wait<1>();
    pipeline_k.consumer_release(smem_pipe_read);

    // Softmax on QK scores (while PV still in flight)
    softmax.online_softmax(tSrS);

    // Wait for PV
    warpgroup_wait<0>();
    pipeline_v.consumer_release(smem_pipe_read_v);

    // Convert softmax output for next iteration's PV GEMM
    convert_type_out(tSrS, tOrP);
};
```

**Three things overlapped**: iteration N's QK GEMM, iteration N-1's PV GEMM, and iteration N's softmax — all in one `fwd_step`.

## ThunderKittens: 4-Stage with LCF Framework

From `thunderkittens-main/kernels/gemm/bf16_h100/bf16_h100_gemm.cu`:

```cpp
static constexpr int INPUT_PIPE_STAGES = 4;
static constexpr int NUM_CONSUMER_WARPS = M_BLOCK * 4;  // 8 consumer warps
static constexpr int PRODUCER_BARRIER_ARRIVALS = 1;      // 1 producer warpgroup
```

### Producer: TMA loads via kittens semaphores

```cpp
struct producer {
    __device__ static void load(producer_load_args<layout> args) {
        if (warpgroup::laneid() == 0) {
            tma::expect(args.inputs_arrived, args.input);
            for (int i = 0; i < M_BLOCK; i++)
                tma::load_async(args.input.a[i], args.globals.A,
                    {args.coord.x+i, args.iter}, args.inputs_arrived);
            for (int i = 0; i < N_BLOCK; i++)
                tma::load_async(args.input.b[i], args.globals.B,
                    {args.iter, args.coord.y+i}, args.inputs_arrived);
        }
    }
};
```

### Consumer: WGMMA + signal

```cpp
struct consumer {
    __device__ static void compute(consumer_compute_args<layout> args) {
        warpgroup::mma_AB(args.state.accum,
            args.input.a[warpgroup::groupid()],
            reinterpret_cast<wide_tile&>(args.input.b));
        warpgroup::mma_async_wait();
        if (warp::laneid() == 0) arrive(args.inputs_finished);  // release buffer
    }
};
```

### Pipeline orchestration (LCF framework)

From `thunderkittens-main/prototype/lcf/lcf.cuh`:

```cpp
// Producer: fill pipeline, then iterate
for (load_iter = 0; load_iter < INPUT_PIPE_STAGES-1; load_iter++) {
    wait(inputs_finished[ring], get_phasebit<1>(bitfield, ring));  // buffer free
    lcft::producer::load({..., inputs_arrived[ring], ...});         // TMA load
    ring = ring_advance<INPUT_PIPE_STAGES>(ring);
}
for (; load_iter < num_iters; load_iter++) {
    wait(inputs_finished[ring], get_phasebit<1>(bitfield, ring));
    lcft::producer::load({..., inputs_arrived[ring], ...});
    ring = ring_advance<INPUT_PIPE_STAGES>(ring);
}

// Consumer: wait for data, compute, release
for (int it = 0; it < num_iters; it++) {
    wait(inputs_arrived[ring], get_phasebit<0>(bitfield, ring));   // data ready
    lcft::consumer::compute({..., inputs_finished[ring], ...});     // WGMMA + release
    ring = ring_advance<INPUT_PIPE_STAGES>(ring);
}
```

## Architecture Comparison

| Aspect | Ampere cp.async | Hopper TMA | FlashAttention FA4 | ThunderKittens |
|---|---|---|---|---|
| Stages | 3 | Configurable | 2 | 4 |
| Sync | `commit_group` / `wait_group<N>` | Per-stage mbarrier pairs | CUTLASS PipelineTmaAsync | Kittens semaphores |
| Producer/Consumer | Same threads | Separate warp groups | WG0=producer, WG1-3=consumer | Last WG=producer, rest=consumer |
| Commit | `cp_async_fence()` | NOP (TMA auto-signals) | NOP (TMA auto-signals) | `tma::expect` + `tma::load_async` |
| Register budget | Shared | Producer: 24-48, Consumer: 224-240 | Producer: 24, Consumer: 240 | Producer: 40, Consumer: 232 |

For the full async execution model (barriers, fences, pitfalls), see `docs/patterns/async-execution.md`.

