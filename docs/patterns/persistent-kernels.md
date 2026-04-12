# Persistent Kernels

## Why Persistent Kernels Work

Standard launch: one CTA per output tile. `N_tiles` CTAs are launched, scheduled by hardware.

Persistent launch: exactly `NUM_SMs` CTAs (one per SM). Each CTA loops, pulling tiles from a scheduler. Benefits:

1. **Zero re-launch overhead** — kernel launch + CTA scheduling done once, not per tile
2. **L2 warmth** — CTA keeps L2 cache lines warm across tile boundaries
3. **Pipeline continuity** — TMA/WGMMA pipeline state carries forward, no drain-refill penalty between tiles

## Pattern: The Persistent Loop

Every implementation follows the same structure:

```
launch NUM_SMs CTAs
each CTA:
    while scheduler.get_next_tile():
        load tile data
        compute
        store result
```

## DeepGemm: stride-based scheduler

From `deepgemm-2.1.1/deep_gemm/include/deep_gemm/common/scheduler.cuh`:

```cpp
__device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
    const auto next_block_idx = (++current_iter) * kNumSMs + blockIdx.x;
    if (next_block_idx >= num_blocks)
        return false;
    get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
    return true;
}
```

CTA `i` processes tiles `i, i+kNumSMs, i+2*kNumSMs, ...`. Both TMA and math warp-groups execute the same loop:

```cpp
// TMA thread:
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    // issue TMA loads for all K tiles
}

// Math warp-groups:
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    // WGMMA accumulation, then TMA store
}
```

## Triton: `tl.range` persistent loop

From `triton-3.6.0/python/tutorials/09-persistent-matmul.py`:

```python
def matmul_kernel_persistent(..., NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(axis=0)
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, ...)
        # ... full GEMM tile compute + store
```

Grid: `min(NUM_SMS, num_tiles)`. The `flatten=True` flag lets the compiler overlap epilogue stores with the next tile's prologue loads across the loop boundary.

```python
grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),)
```

## ThunderKittens: LCF task loop

From `thunderkittens-main/prototype/lcf/lcf.cuh`:

```cpp
// Grid = 132 CTAs (H100 SM count)
template<bool PERSISTENT_GRID=true>
__host__ static inline dim3 grid(int M, int N, int K) {
    return dim3(PERSISTENT_GRID ? 132 : M*N/(M_BLOCK*N_BLOCK*...));
}
```

The kernel loops with `task_iter`:

```cpp
for (int task_iter = 0; true; task_iter++) {
    int num_iters = -1;
    lcft::common_setup(unif);  // maps task_iter*gridDim.x + blockIdx.x → tile coords
    if (num_iters < 0) break;  // no more work
    // ... producer loads or consumer computes for num_iters k-tiles
}
```

Work assignment in `common_setup`:
```cpp
int task_id = args.task_iter * gridDim.x + blockIdx.x;
if (task_id >= Rblocks * Cblocks) {
    args.num_iters = -1;  // STOP
    return;
}
```

## CUTLASS: PersistentTileScheduler

From `cutlass-4.4.2/include/cutlass/gemm/kernel/static_tile_scheduler.hpp`:

```cpp
// Each CTA gets a linear index, strides by total grid size
StaticPersistentTileScheduler(Params const& params_) {
    current_work_linear_idx_ = blockIdx.x + blockIdx.y * gridDim.x;
    total_grid_size_ = gridDim.x * gridDim.y * gridDim.z;
}

void advance_to_next_work(uint32_t advance_count = 1) {
    current_work_linear_idx_ += total_grid_size_ * advance_count;
}

WorkTileInfo get_current_work_for_linear_idx(uint64_t linear_idx) const {
    if (linear_idx >= scheduler_params.blocks_per_problem_)
        return WorkTileInfo::invalid_work_tile();
    auto [work_idx_m, work_idx_n] = get_work_idx_m_and_n(...);  // with grid swizzle
    return {work_idx_m, work_idx_n, work_idx_l, true};
}
```

Used in the kernel main loop:
```cpp
auto work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});
while (work_tile_info.is_valid()) {
    // compute tile...
    scheduler.advance_to_next_work();
    work_tile_info = scheduler.fetch_next_work(work_tile_info, ...);
}
```

`PersistentScheduler` is the **default** for SM90+ in CUTLASS. Grid is capped at SM count.

## When to Use Persistent Kernels

- **Always for Hopper/Blackwell** — all modern reference implementations use them
- **Large tile counts** — the benefit grows with more tiles (more launch overhead saved)
- **Pipeline-heavy kernels** — the pipeline drain/refill between tiles is eliminated
- **Combined with grid swizzling** (see `docs/patterns/grid-swizzling.md`) for L2 cache reuse
