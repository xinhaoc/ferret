# Grid Swizzling (L2 Tile Ordering)

## Why Tile Ordering Matters

In matmul `C = A × B`, tile (m, n) reads row m of A and column n of B. With naive row-major CTA ordering:
- CTAs walk left-to-right across N, then down M
- Row 0 loads A[0] but needs ALL columns of B
- Row 1 loads A[1] and ALL columns of B again
- B data evicted from L2 before it's reused

With grouped ordering (column-major within groups of G M-rows):
- G consecutive CTAs share the same B column tile in L2
- Consecutive groups share overlapping A row tiles
- L2 hit rate increases dramatically

Triton reports **10%+ improvement** (220 → 245 TFLOPS on A100) from this alone.

## The Core Idea

Group G consecutive M-tiles together. Within each group, iterate column-major (down M before across N). This means G tiles share the same B data in L2.

```
Naive (row-major):          Grouped (G=4):
[0  1  2  3  4  5]         [0  4  8  12 16 20]
[6  7  8  9  10 11]        [1  5  9  13 17 21]
[12 13 14 15 16 17]        [2  6  10 14 18 22]
[18 19 20 21 22 23]        [3  7  11 15 19 23]
```

In the grouped version, CTAs 0-3 all read the same B column → L2 reuse.

## Triton: GROUP_SIZE_M

From `triton-3.6.0/python/tutorials/03-matrix-multiplication.py`:

```python
pid = tl.program_id(axis=0)
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

Default: `GROUP_SIZE_M = 8` across all autotuned configs.

From the tutorial: "In a 9×9 block matmul, row-major ordering requires loading 90 blocks into SRAM for the first 9 output blocks, but grouped ordering only needs 54 blocks."

## ThunderKittens: SUPER_M

From `thunderkittens-main/kernels/gemm/bf16_h100/bf16_h100_gemm.cu`:

```cpp
template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct matmul_template {
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (M_BLOCK*64);
        int Cblocks = args.globals.C.cols() / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M;
        int super_repeat = SUPER_M * Cblocks;
        int task_id = args.task_iter * gridDim.x + blockIdx.x;

        if (task_id < super_rows * Cblocks)
            args.common.coord = {
                SUPER_M*(task_id/super_repeat) + task_id%SUPER_M,    // M: column-major within group
                (task_id%super_repeat)/SUPER_M                        // N: advances after SUPER_M rows
            };
        else { /* handle remainder rows */ }
    }
};
```

Default: `SUPER_M = 12`. Different configs tested: `SUPER_M=8`, `SUPER_M=12`.

## CUTLASS: Log-Swizzle

From `cutlass-4.4.2/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp`:

```cpp
static CUTLASS_DEVICE cute::tuple<int32_t, int32_t>
get_work_idx_m_and_n(uint64_t blk_per_grid_dim, ..., int32_t log_swizzle_size, ...) {
    uint64_t offset = cluster_id & ((1 << log_swizzle_size) - 1);
    uint64_t extra = cluster_id >> log_swizzle_size;
    divmod_cluster_blk_major(cluster_idx_minor_div_swizzle, cluster_idx_major, extra);
    cluster_idx_minor = cluster_idx_minor_div_swizzle * (1 << log_swizzle_size) + offset;
}
```

The swizzle size is chosen automatically based on problem dimensions:

```cpp
// From tile_scheduler_params.h
static int32_t get_log_swizzle_size(int problem_ctas_m, int problem_ctas_n, int max_swizzle_size) {
    int min_cta_dim = min(problem_ctas_m, problem_ctas_n);
    if (max_swizzle_size >= 8 && min_cta_dim >= 6) return 3;  // groups of 8
    else if (max_swizzle_size >= 4 && min_cta_dim >= 3) return 2;  // groups of 4
    else if (max_swizzle_size >= 2 && min_cta_dim >= 2) return 1;  // groups of 2
    else return 0;  // no swizzle
}
```

## DeepGemm: L2-Optimized Group Size

From `deepgemm-2.1.1.post3/deep_gemm/include/deep_gemm/common/scheduler.cuh`:

```cpp
template <...>
static constexpr uint32_t get_num_1d_blocks_per_group() {
    uint32_t num_best_blocks = 0, min_usage = max;
    for (const auto& candidate : {8u, 16u}) {
        // Estimate L2 working set: group_size * tile_cols + num_groups * tile_rows
        const auto& usage = kIsMulticastOnA ?
            candidate * BLOCK_N + ceil_div(kNumSMs, candidate) * BLOCK_M :
            candidate * BLOCK_M + ceil_div(kNumSMs, candidate) * BLOCK_N;
        if (usage < min_usage)
            min_usage = usage, num_best_blocks = candidate;
    }
    return num_best_blocks;
}
```

DeepGemm goes further than the others: it **minimizes the L2 footprint** by choosing between group size 8 and 16 based on which creates a smaller working set. The formula estimates live L2 data: `group_size` tiles in one dimension + `num_concurrent_groups` tiles in the other.

The swizzle also aligns with TMA multicast — paired CTAs share the same row or column for 2-way multicast.

## Choosing the Group Size

| Implementation | Default | Selection Method |
|---|---|---|
| Triton | GROUP_SIZE_M = 8 | Fixed, autotuned |
| ThunderKittens | SUPER_M = 12 | Fixed per config (8 or 12) |
| CUTLASS | 2^log_swizzle_size (2, 4, or 8) | Automatic based on problem shape |
| DeepGemm | 8 or 16 | Compile-time L2 footprint minimization |

Rule of thumb: **8 is a good starting point.** CUTLASS's heuristic (larger groups for larger problems) is sound. DeepGemm's L2 minimization is the most principled.

## Interaction with Persistent Kernels

Grid swizzling is almost always combined with persistent kernels (see `docs/patterns/persistent-kernels.md`):
- Persistent kernel → fixed number of CTAs cycling through tiles
- Grid swizzle → tiles are ordered for L2 reuse
- Together: each CTA processes L2-adjacent tiles in sequence, maximizing cache hits

All four implementations (Triton, ThunderKittens, CUTLASS, DeepGemm) use both patterns together.
