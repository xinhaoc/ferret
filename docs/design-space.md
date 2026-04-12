# GEMM Kernel Design Space

Complete optimization dimensions for GEMM kernels on modern NVIDIA GPUs.
Each candidate is a JOINT configuration — all dimensions together define one kernel.

---

## 1. Problem Formulation

- **transpose_a**: whether A is transposed (N or T)
- **transpose_b**: whether B is transposed (N or T)
- **swap_ab**: compute C^T = B^T × A^T instead of C = A × B (swaps role of M and N)
- **split_k**: decompose K across multiple CTAs (K-parallel)
- **split_k_factor**: number of K-splits (1, 2, 4, ...)
- **split_k_reduction**: how partial results combine (separate reduce kernel, in-place atomics, cooperative reduction via SMEM/DSMEM)

## 2. Tiling

- **block_m**: tile M dimension per CTA
- **block_n**: tile N dimension per CTA
- **block_k**: tile K dimension per pipeline stage
- **partial_tile_handling**: edge tiles where problem doesn't divide evenly (TMA OOB zero-fill, separate edge kernel, pad inputs, mask in epilogue)
- **tile_shape_selection**: how tile shapes are chosen (fixed, runtime heuristic, autotuned)

## 3. Launch Configuration

- **grid_dim**: number of thread blocks [x, y, z]
- **block_dim**: threads per block [x, y, z]
- **cluster_dim**: CTAs per cluster [x, y, z]
- **launch_bounds**: compiler hint [max_threads_per_block, min_blocks_per_sm]
- **dynamic_smem_size**: shared memory requested at launch
- **launch_api**: <<<>>> vs cudaLaunchKernelEx (required for clusters)

## 4. Grid & Scheduling

- **grid_strategy**: non-persistent, persistent (grid = SMs), multi-tile (grid < tiles)
- **tile_ordering**: linear, swizzled/grouped for L2 locality (GROUP_SIZE_M in Triton)
- **tile_assignment**: static (blockIdx mapping) vs dynamic (atomic counter, CLC work-stealing)
- **tiles_per_cta**: fixed 1, fixed N, or dynamic
- **clc_scheduling**: cluster launch control for dynamic work distribution
- **target_occupancy**: desired resident CTAs per SM (1, 2, ...)
- **wave_quantization**: grid size chosen to minimize partial waves

## 5. Pipeline

- **num_stages**: pipeline depth (SMEM buffer count for overlapping loads + compute)
- **split_ab_pipeline**: independent stage counts for A and B operands
- **stages_a / stages_b**: per-operand depth (if split)
- **pipeline_type**: simple circular, ping-pong, producer-consumer
- **pipeline_overlap**: what operations overlap (TMA↔MMA, MMA↔epilogue, TMA↔MMA↔epilogue)
- **pipeline_warmup**: how many stages filled before MMA starts
- **pipeline_drain**: how tail iterations are handled

## 6. K-loop Structure

- **k_unroll**: unroll factor (1, 2, 4, 8)
- **k_block_grouping**: K-blocks processed per barrier cycle
- **k_loop_structure**: for-loop, while-loop with predicate, fully unrolled (small K)
- **software_pipelining_depth**: independent MMA operations in flight (distinct from SMEM stages)
- **inner_loop_unroll**: unroll within a single K-block (MMA instructions per block_k)

## 7. Threads & Warp Specialization

- **num_threads**: total threads per CTA
- **num_warps**: derived from threads (threads/32)
- **warp_specialization**: whether warps have dedicated roles vs all-participate
- **warp_roles**: role assignment (TMA producer, MMA compute, epilogue consumer, idle/helper)
- **num_tma_warps**: warps dedicated to TMA loads
- **num_mma_warps**: warps dedicated to MMA compute
- **num_epilogue_warps**: warps dedicated to epilogue stores
- **role_overlap**: whether warps handle multiple roles (e.g., MMA warp also does epilogue)
- **thread_mapping**: how threads map to output elements (1D, 2D, warp-tiled)

## 8. MMA (Matrix Multiply-Accumulate)

- **mma_instruction**: tcgen05.mma, wgmma.mma_async, mma.sync, wmma
- **mma_shape**: per-instruction [M, N, K] (e.g., 128×256×16)
- **mma_mode**: normal (operands from SMEM), .ws (A from TMEM, B from SMEM)
- **cta_group**: 1 (single CTA) or 2 (cooperative 2-SM MMA)
- **accumulator_type**: FP32, FP16, BF16
- **mma_issue_thread**: which thread(s) issue MMA (lane 0, all lanes predicated, elected)
- **mma_commit**: how MMA completion is signaled to barriers

## 9. TMA (Tensor Memory Accelerator)

- **tma_enabled**: use TMA vs manual loads (cp.async, LDG, LDGSTS)
- **tma_dim_a**: descriptor dimensionality for A (1D, 2D, 3D, 4D, 5D)
- **tma_dim_b**: descriptor dimensionality for B
- **tma_box_size_a**: TMA load box dimensions for A
- **tma_box_size_b**: TMA load box dimensions for B
- **tma_oob_fill**: out-of-bounds fill (zero, NaN)
- **tma_cache_hint_a**: L2 cache hint for A (EVICT_NORMAL, EVICT_FIRST, EVICT_LAST, STREAMING)
- **tma_cache_hint_b**: L2 cache hint for B
- **tma_prefetch**: prefetch descriptor (prefetch.tensormap)
- **tma_store**: use TMA for output stores (vs thread-driven STG)
- **tma_load_order**: order of issuing A vs B loads per stage (A-first, B-first, interleaved)

## 10. Shared Memory (SMEM)

- **smem_total**: total shared memory per CTA
- **smem_layout_a**: how A is laid out in SMEM
- **smem_layout_b**: how B is laid out in SMEM
- **smem_swizzle_a**: swizzle mode for A (NONE, 32B, 64B, 128B)
- **smem_swizzle_b**: swizzle mode for B
- **smem_padding**: padding to avoid bank conflicts
- **smem_store_buffer**: SMEM staging for TMA store epilogue (if used)
- **smem_store_stages**: epilogue store buffer count (if TMA store)
- **smem_allocation**: static (compile-time) vs dynamic (launch-time)

## 11. Tensor Memory (TMEM)

- **tmem_cols**: columns allocated for accumulator
- **tmem_buffering**: single, double, triple buffered
- **tmem_load_width**: columns per epilogue load instruction (x8, x16, x32)
- **tmem_release_timing**: when TMEM is signaled empty (after epilogue, before epilogue during async store)
- **tmem_layout**: mapping of accumulator elements to TMEM rows/cols
- **tmem_a_cols**: columns for A operand (if .ws mode)
- **tmem_scratch**: TMEM used for non-accumulator purposes (descriptors, small tables)

## 12. Epilogue

- **epilogue_type**: thread-driven STG, TMA store via SMEM staging, warp-cooperative
- **epilogue_threads**: how many threads participate in epilogue
- **epilogue_warps**: dedicated epilogue warps vs reusing other warps
- **epilogue_vectorization**: store width (32b, 64b, 128b)
- **epilogue_coalescing**: how stores are ordered for memory coalescing
- **epilogue_conversion**: accumulator→output type conversion (FP32→BF16, FP32→FP8, etc.)
- **epilogue_fusion**: fused post-ops (bias add, ReLU, GELU, residual add, quantization)
- **epilogue_overlap**: overlap with next tile (epilogue of tile N runs during TMA of tile N+1)
- **epilogue_batching**: TMEM load grouping (load all then convert, or load-convert-store interleaved)

## 13. Cluster & Multicast

- **cluster_size**: total CTAs (1, 2, 4, 8, 16)
- **cluster_layout**: [M_cluster, N_cluster] arrangement
- **multicast_a**: TMA multicast for A operand
- **multicast_b**: TMA multicast for B operand
- **multicast_type**: manual (multicast::cluster PTX), cta_group::2 cooperative TMA
- **dsmem_enabled**: distributed shared memory for cross-CTA data sharing
- **dsmem_usage**: what data travels via DSMEM (partial results, operand tiles, barriers)
- **cluster_sync_points**: where cluster-wide sync occurs (init, per-tile, exit)

## 14. Barrier & Synchronization

- **num_barriers**: mbarrier objects used (max 16 per CTA)
- **barrier_roles**: which barriers for which purpose (full per stage, empty per stage, TMEM full, TMEM empty, cluster sync, ...)
- **barrier_scope**: CTA-scope (.shared::cta) vs cluster-scope (.shared::cluster)
- **barrier_protocol**: arrive_expect_tx, arrive_one, arrive_and_expect_tx
- **barrier_init_count**: arrival count per barrier
- **barrier_wait_type**: try_wait spin, try_wait with nanosleep, blocking wait
- **fence_type**: proxy fence (.async), memory fence (.cta/.gpu/.sys)
- **fence_placement**: before MMA, after TMA, after epilogue store
- **commit_strategy**: how MMA completion signals barrier (per K-block, per unroll group, per tile)
- **sync_granularity**: per-stage, per-K-block, per-tile, per-cluster

## 15. Descriptor Management

- **smem_descriptor_encoding**: how SMEM descriptors are built (SBO, LBO, swizzle, version)
- **descriptor_computation**: per-stage arithmetic, shfl broadcast, precomputed in registers
- **instruction_descriptor**: compile-time constant vs runtime dynamic
- **tma_descriptor_storage**: __grid_constant__, global memory, SMEM cached

## 16. Memory Policies

- **l2_policy_a**: L2 cache promotion for A (NONE, 128B, 256B)
- **l2_policy_b**: L2 cache promotion for B
- **l2_persistence**: pin data in L2 via stream attributes (cudaStreamSetAttribute)
- **l2_access_window**: L2 persistence window size
- **prefetch_strategy**: TMA descriptor prefetch, software prefetch ahead of pipeline
- **memory_ordering**: relaxed, acquire/release — scope of memory operations

## 17. Registers & Compiler

- **max_registers**: per-thread register limit (auto, 32, 64, 128, 255)
- **fast_math**: --use_fast_math (faster but less precise intrinsics)
- **ptx_optimization**: -Xptxas -O{level} (ptxas optimization aggressiveness)
- **expensive_optimizations**: -Xptxas --allow-expensive-optimizations
- **inline_ptx**: hand-written PTX for critical sections
- **pragma_unroll**: explicit unroll hints on specific loops
- **restrict_pointers**: __restrict__ for pointer aliasing hints
- **const_memory**: __constant__ for read-only data
- **volatile**: where volatile is used/avoided

## 18. Data Layout

- **a_layout**: row-major, column-major, custom tiled
- **b_layout**: row-major, column-major, custom tiled
- **c_layout**: output layout
- **a_alignment**: memory alignment of A pointer
- **b_alignment**: memory alignment of B pointer
- **data_type_in**: input precision (FP16, BF16, FP8, TF32, FP32, INT8)
- **data_type_out**: output precision
- **data_type_accum**: accumulator precision

## 19. Numeric

- **accumulation_order**: sequential, pairwise, Kahan compensated
- **rounding_mode**: RN (nearest), RZ (toward zero), RU, RD
- **denormal_handling**: flush-to-zero vs IEEE compliant
- **mixed_precision**: input vs accumulator vs output precision combinations

---

## Dimension Interactions

These dimensions are NOT independent. Key interactions:

| Change | Affects |
|---|---|
| Tiling (block_m/n/k) | SMEM per stage, pipeline depth, tile count, TMEM cols, TMA box size |
| Pipeline depth | SMEM total, barrier count, warmup/drain overhead |
| Threads | Warp count, registers per thread, epilogue parallelism |
| K-unroll | Register pressure, code size, barrier amortization, compiler behavior |
| Cluster | Grid size, barrier scope, multicast options, launch API |
| Occupancy | SMEM budget per CTA, register budget, pipeline depth |
| MMA mode (.ws) | TMEM layout (A in TMEM), SMEM budget (B only), pipeline balance |
| Split-K | Grid size, reduce kernel, workspace memory, synchronization |
| Swap-AB | ALL tiling, TMEM layout, pipeline balance, multicast roles |
| Epilogue type | SMEM budget (staging buffer), thread roles, overlap strategy |
| Data type | MMA instruction selection, TMEM layout, conversion cost |

---

## How to Use This

This is a REFERENCE of what dimensions exist and how they interact.
Each candidate specifies only the dimensions that DIFFER from the current
baseline — everything else inherits. A candidate is a coherent joint
configuration, not an isolated change to one dimension.
