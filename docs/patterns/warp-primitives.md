# Warp-Level Primitives

## Warp Shuffle in Production

### FlashAttention: 4-Thread Butterfly Reduction for Online Softmax

In SM90's WGMMA layout, each row of the attention score matrix is split across 4 threads. FlashAttention uses `Allreduce<4>` for row-wise max and sum:

From `flash-attention-fa4-v4.0.0.beta4/hopper/softmax.h`:

```cpp
// quad_allreduce_ reduces across 4 threads sharing an attention score row
template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst,
                                                 Tensor<Engine1, Layout1> &src, Operator &op) {
    #pragma unroll
    for (int i = 0; i < size(dst); i++) {
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}
```

Called in the online softmax with `MaxOp<float>` for row-max and `SumOp<float>` for row-sum. The `Allreduce<4>` expands to two `__shfl_xor_sync` calls (offsets 2, then 1).

### FlashInfer: Shuffle for stmatrix Emulation

When `stmatrix` hardware support is missing, FlashInfer uses `__shfl_sync` to reconstruct tile data across lanes:

From `flashinfer-0.6.7/include/flashinfer/mma.cuh`:

```cpp
const uint32_t tx = threadIdx.x;
uint4 word;
#pragma unroll
for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
    word.x = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4);
    word.y = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4 + 1);
    word.z = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4 + 2);
    word.w = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4 + 3);
    if (tx / 8 == reg_id) {
        *(uint4*)smem_ptr = word;
    }
}
```

Each thread shuffles register values to the appropriate lane to reconstruct the tile for a shared memory store.

## Warp Vote / Ballot in Production

### FlashAttention: Warp-Cooperative Binary Search

From `flash-attention-fa4-v4.0.0.beta4/hopper/tile_scheduler.hpp`:

```cpp
int batch_idx_in_group = __popc(__ballot_sync(0xffffffff,
    group_start_tile + num_m_blocks_cumulative * params.num_head <= next_tile_idx));
bidb += batch_idx_in_group;
num_m_blocks = __shfl_sync(0xffffffff, num_m_blocks, batch_idx_in_group);
```

Each lane holds data for a different batch element. `__ballot_sync` creates a bitmask of which lanes' batch elements fit before the target tile index. `__popc` gives the batch index. `__shfl_sync` broadcasts the chosen batch's data. This is a warp-cooperative binary search over batch boundaries.

### NCCL: Integer Division via Ballot

From `nccl-2.29.7/src/device/sendrecv.h`:

```cpp
// Fastest way to compute warp-uniform division x/y in [0,32):
// each lane guesses a quotient, ballot collects, popcount gives result
int nWarpPerWork = __popc(__ballot_sync(~0u, nWorks*(lane+1) <= nWarps));
int workIx = __popc(__ballot_sync(~0u, (lane+1)*nWarpPerWork <= wid));
```

3x faster than standard integer division — each lane tests a candidate, ballot aggregates, popcount gives the answer.

### ThunderKittens: NaN Detection

From `thunderkittens-main/include/ops/group/register/tile/tile.cuh`:

```cpp
return (__ballot_sync(0xffffffff, nan_detected) != 0);
```

Each lane checks its tile elements for NaN, `__ballot_sync` aggregates — any set bit means NaN was found.

### NCCL: Role Detection Mask

From `nccl-2.29.7/src/device/prims_simple.h`:

```cpp
uint32_t mask = __ballot_sync(~0u,
    ((flags & RoleWaitRecv) && (flags & NetDeviceUnpack)) ? 1 : 0);
if (tid == 0) {
    ncclShmem.groups[this->group].devicePlugin.unpack.unpackNetDeviceIndexMask = mask;
}
```

Identifies which threads have specific capabilities to create a dispatch mask.

## Elect One (SM90+)

From `cutlass-4.4.2/include/cute/arch/cluster_sm90.hpp`:

```cpp
CUTE_HOST_DEVICE uint32_t elect_one_sync() {
    uint32_t pred = 0;
    uint32_t laneid = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "     elect.sync %%rx|%%px, %2;\n"
        "@%%px mov.s32 %1, 1;\n"
        "     mov.s32 %0, %%rx;\n"
        "}\n"
        : "+r"(laneid), "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}
```

Used in CUTLASS FMHA for selecting the pipeline leader. From `cutlass-4.4.2/examples/88_hopper_fmha/`:

```cpp
int lane_predicate = cute::elect_one_sync();

// Only the elected lane prefetches TMA descriptors
if ((warp_idx == 0) && lane_predicate) {
    CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
}

// Only the elected lane is the pipeline leader
pipeline_params.is_leader = lane_predicate && (producer_warp_role == WarpRoleLoadQ);
```

## Cooperative Groups (Modern API)

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel() {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    float sum = cg::reduce(warp, my_val, cg::plus<float>());

    // Sub-warp tiles
    auto half_warp = cg::tiled_partition<16>(warp);
    float half_sum = cg::reduce(half_warp, my_val, cg::plus<float>());
}
```

## Performance Notes

- Warp shuffles: 1 cycle latency, no shared memory needed
- `__ballot_sync` + `__popc` is the fastest warp-wide count/search
- `elect_one_sync` (SM90+) is faster than `threadIdx.x % 32 == 0` for leader selection
- Prefer shuffle over shared memory for warp-local communication
- Cooperative groups is the modern API — prefer for new code

