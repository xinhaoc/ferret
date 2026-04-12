# Thread Block Clusters

## 1. What Is a Thread Block Cluster

A thread block cluster is an optional grouping in the CUDA hierarchy: threads -> warps -> thread blocks (CTAs) -> **clusters** -> grid. Introduced in Compute Capability 9.0 (Hopper), carried forward in 10.0 (Blackwell).

A cluster is a set of CTAs that are **guaranteed to be co-scheduled** onto SMs within the same GPU Processing Cluster (GPC). This co-scheduling guarantee enables:

1. **Distributed Shared Memory (DSMEM)**: Any thread in any CTA of the cluster can directly load, store, and atomically operate on the shared memory of any other CTA in the cluster.
2. **Hardware-accelerated cluster-wide barriers**: `barrier.cluster.arrive` / `barrier.cluster.wait` synchronize all threads across all CTAs in the cluster without going through global memory.
3. **TMA multicast**: A single TMA load can deliver data from global memory to the shared memory of multiple CTAs simultaneously.
4. **Dedicated SM-to-SM network**: Communication within a cluster uses a fast interconnect within the GPC, approximately 7x faster than global memory exchange.

### Hardware Model

Each GPC on Hopper contains multiple SMs (typically 16-18 per GPC on H100, which has 8 GPCs for 132 SMs). A cluster's CTAs are placed on adjacent SMs within the same GPC, connected by a dedicated SM-to-SM network for DSMEM access.

The cluster is NOT the same as "cooperative groups" in general. Cooperative groups is a software programming API that provides group abstractions at various levels (warp, block, grid). The `cluster_group` is one specific type within that API that exposes cluster functionality. The hardware guarantee of co-scheduling is what makes clusters special -- cooperative groups at the grid level do NOT guarantee co-scheduling.

---

## 2. How to Launch Clusters

### Method 1: Compile-Time Attribute

```cpp
__global__ void __cluster_dims__(2, 1, 1) my_kernel(float* data) {
    // cluster of 2 CTAs in x-dimension
}

// Launch: grid must be a multiple of cluster dims
dim3 grid(num_blocks, 1, 1);  // num_blocks must be even
dim3 block(256, 1, 1);
my_kernel<<<grid, block>>>(data);
```

### Method 2: Runtime via cudaLaunchKernelEx

```cpp
__global__ void my_kernel(float* data) {
    // No compile-time cluster attribute
}

cudaLaunchConfig_t config = {0};
config.gridDim = num_blocks;
config.blockDim = 256;

cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeClusterDimension;
attribute[0].val.clusterDim.x = 2;
attribute[0].val.clusterDim.y = 1;
attribute[0].val.clusterDim.z = 1;
config.attrs = attribute;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, my_kernel, data);
```

### Method 3: CUTLASS launch_kernel_on_cluster

```cpp
dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
cutlass::launch_kernel_on_cluster(launch_params, kernel, kernel_params);
```

### Method 4: __block_size__ (Blocks as Clusters)

```cpp
// Specify both block size and cluster size, then launch with cluster count
__block_size__((1024, 1, 1), (2, 2, 2)) __global__ void foo();
foo<<<dim3(8, 8, 8)>>>();  // 8x8x8 clusters
```

### Cluster Size Limits

| GPU | Portable Max | Non-Portable Max | Opt-In Required |
|-----|-------------|-----------------|-----------------|
| H100 (SM90) | 8 CTAs | 16 CTAs | `cudaFuncAttributeNonPortableClusterSizeAllowed` |
| B200 (SM100) | 8 CTAs | 16 CTAs | `cudaFuncAttributeNonPortableClusterSizeAllowed` |
| RTX 5090 (SM120) | 8 CTAs | 8 CTAs | N/A |

The cluster shape is a dim3 (x, y, z). The total number of CTAs = x * y * z <= max cluster size.

**Critical**: The grid dimension (in blocks) must be a multiple of the cluster dimension in each axis. If not, the launch will fail.

### Querying Cluster Occupancy

```cpp
// Query max cluster size for a kernel
int max_cluster_size = 0;
cudaOccupancyMaxPotentialClusterSize(&max_cluster_size, (void*)kernel, &config);

// Query max active clusters
int max_active_clusters = 0;
cudaOccupancyMaxActiveClusters(&max_active_clusters, (void*)kernel, &config);
```

---

## 3. Distributed Shared Memory (DSMEM)

### Concept

Every CTA in a cluster has its own shared memory. DSMEM is the union of all these shared memories, accessible by any thread in the cluster. The total DSMEM size = (shared memory per block) x (number of blocks in cluster).

Shared memory is declared per-block as usual. There is no separate DSMEM declaration. The difference is in how you address it.

### Address Spaces

PTX has two sub-qualifiers for `.shared`:
- `.shared::cta` -- addresses in the current CTA's shared memory (default)
- `.shared::cluster` -- addresses in any CTA's shared memory within the cluster

`.shared::cta` addresses are a subset of `.shared::cluster` addresses. If no sub-qualifier is specified, `.shared` defaults to `::cta`.

### Accessing Peer CTA's Shared Memory

**Method 1: cooperative_groups API (high-level)**

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel() {
    extern __shared__ int smem[];
    auto cluster = cg::this_cluster();

    // Ensure all blocks in cluster are running
    cluster.sync();

    // Get pointer to another block's shared memory
    int peer_rank = (cluster.block_rank() + 1) % cluster.num_blocks();
    int* peer_smem = cluster.map_shared_rank(smem, peer_rank);

    // Now read/write/atomic on peer_smem
    int val = peer_smem[threadIdx.x];
}
```

**Method 2: PTX mapa instruction (low-level)**

```
// Map local shared address to CTA rank b
mapa.shared::cluster.u32  d, local_smem_addr, target_cta_rank;

// Then load/store using shared::cluster
ld.shared::cluster.u32  result, [d];
st.shared::cluster.u32  [d], value;
```

The `mapa` instruction takes a shared memory address in the current CTA and a target CTA rank, and returns the corresponding `.shared::cluster` address that points to the same offset in the target CTA's shared memory.

**Method 3: getctarank -- reverse lookup**

```
// Given a shared::cluster address, find which CTA it belongs to
getctarank.shared::cluster.u32  cta_rank, cluster_addr;
```

### DSMEM Performance Rules

- **Coalesce accesses**: Align to 32-byte segments, same as global memory coalescing rules.
- **Avoid non-unit stride**: If you need scattered access patterns, copy data to local shared memory with padding first.
- **~7x faster** than global memory exchange (per NVIDIA documentation for Hopper).
- **Simultaneous with L2**: DSMEM accesses can occur in parallel with L2 cache accesses, providing combined bandwidth.
- **Must ensure existence**: The peer CTA must be alive (not exited) when you access its shared memory. Use `cluster.sync()` before first DSMEM access and before any CTA exits.

### DSMEM One-Way Barrier Pattern

A key asymmetry: threads can perform `mbarrier.arrive` on an mbarrier in `.shared::cluster` (remote CTA), but they CANNOT perform `mbarrier.try_wait` or `mbarrier.test_wait` on a remote mbarrier. Wait operations are only supported on `.shared::cta` (local) mbarriers.

This means the pattern for cross-CTA synchronization is:
1. CTA A arrives on CTA B's mbarrier (one-way signal)
2. CTA B waits on its own local mbarrier

This is exactly what TMA multicast uses -- the TMA hardware arrives on each destination CTA's local mbarrier.

---

## 4. Cluster-Scoped vs CTA-Scoped Barriers

### barrier.cluster (Hardware Cluster Barrier)

```
barrier.cluster.arrive.aligned;    // arrive (release semantics by default)
barrier.cluster.wait.aligned;      // wait for all CTAs in cluster (acquire semantics)
```

- Synchronizes ALL non-exited threads across ALL CTAs in the cluster.
- Automatically reinitialized after each completion.
- Has release/acquire memory ordering by default.
- `.relaxed` variant on arrive skips memory ordering (use with explicit fence).
- Equivalent to `cooperative_groups::this_cluster().sync()`.

### mbarrier Scope Differences

mbarrier objects live in shared memory and can have different scopes:

| Operation | `.shared::cta` mbarrier | `.shared::cluster` mbarrier |
|-----------|------------------------|---------------------------|
| `mbarrier.init` | Supported | NOT supported |
| `mbarrier.arrive` | Supported, returns token | Supported, CANNOT return token |
| `mbarrier.arrive_drop` | Supported | Supported, cannot return result |
| `mbarrier.expect_tx` | Supported | Supported |
| `mbarrier.complete_tx` | Supported | Supported |
| `mbarrier.try_wait` | Supported | NOT supported |
| `mbarrier.test_wait` | Supported | NOT supported |
| `mbarrier.pending_count` | Supported | NOT supported |

**Key rule**: You can signal (arrive on) a remote CTA's mbarrier, but you can only wait on your own CTA's mbarrier.

### When to Use Which

- **CTA-scoped mbarrier** (`mbarrier.init/arrive/wait` in `.shared::cta`): Standard producer-consumer within a single CTA. Used for TMA load completion, pipeline stage tracking.
- **Cluster-scoped mbarrier arrive** (`mbarrier.arrive` targeting `.shared::cluster`): Signal a remote CTA that data is ready. Used for cross-CTA producer-consumer patterns, DSMEM writes followed by remote signal.
- **barrier.cluster**: Full cluster synchronization (all threads, all CTAs). Used at kernel start (ensure all CTAs running), before first DSMEM access, and before any CTA exits.

### Fence Requirements

After `mbarrier.init`, you MUST call:
```
fence.mbarrier_init.release.cluster;
```
followed by `__syncthreads()` (or `barrier.cluster` if multi-CTA) before any thread uses the barrier. This is required even for CTA-scoped barriers when in a cluster context.

---

## 5. TMA Multicast

### Concept

TMA multicast loads data from global memory and delivers it to the shared memory of MULTIPLE CTAs in a cluster simultaneously, using a single TMA instruction. This reduces global memory bandwidth consumption proportionally to the number of participating CTAs.

### How It Works

1. One thread (typically the producer/leader thread in one CTA) issues a `cp.async.bulk.tensor` with the `.multicast::cluster` qualifier.
2. A 16-bit multicast mask specifies which CTAs receive the data: bit i = 1 means CTA with cluster rank i participates.
3. The TMA hardware reads the data once from global memory and writes it to the same shared memory offset in each participating CTA.
4. The TMA hardware signals each participating CTA's mbarrier upon completion.

### PTX Syntax

```
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster
    [smem_addr], [tensor_desc, {coord0, coord1}], [mbar_addr], ctamask;
```

- `smem_addr`: Destination in shared memory (same offset in each participating CTA)
- `tensor_desc`: TMA descriptor created on host
- `mbar_addr`: mbarrier address. For `cta_group::1`, this is the local CTA's barrier (multicast to same offset in all CTAs). For `cta_group::2`, the signal is multicast to odd or even CTAs specifically.
- `ctamask`: 16-bit mask of participating CTAs

### Multicast Mask Construction

For a 2x2 cluster with column-major CTA ordering:
```
CTA layout:  [0, 2]
             [1, 3]
```

For loading operand A (shared along rows):
- CTA 0 and CTA 2 share A rows -> mask for CTA 0: (1 << 0) | (1 << 2) = 0x0005
- CTA 1 and CTA 3 share A rows -> mask for CTA 1: (1 << 1) | (1 << 3) = 0x000a

For loading operand B (shared along columns):
- CTA 0 and CTA 1 share B columns -> mask for CTA 0: (1 << 0) | (1 << 1) = 0x0003
- CTA 2 and CTA 3 share B columns -> mask for CTA 2: (1 << 2) | (1 << 3) = 0x000c

### FlashAttention Multicast Example

From flash-attention-fa4 (Hopper):
```cpp
// Construct multicast mask for K/V loads (cluster along M dimension)
uint16_t mcast_mask_kv = 0;
for (int m = 0; m < size<0>(ClusterShape{}); ++m) {
    mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
}

// Issue TMA load with multicast
copy(params.tma_load_K.with(
    *pipeline_k.producer_get_barrier(smem_pipe_write),
    mcast_mask_kv,
    TMA::CacheHintSm90::EVICT_LAST),
    tKgK_TMA(_, n_block), tKsK_TMA(_, smem_pipe_write.index()));
```

FlashAttention uses ClusterM=2 (2 CTAs along M). Both CTAs in the cluster load the same K/V tile, so multicast halves the K/V global memory bandwidth.

### Multicast Barrier Crediting (SM100)

There are two approaches to multicast on SM100. They have different barrier setups.

#### Approach 1: `cta_group::2` cooperative TMA (DeepGemm pattern)

Both CTAs execute the **same** TMA instruction. Hardware delivers data to both CTAs'
SMEM and credits the leader CTA's barrier. Uses `ClusterTransactionBarrier` (cluster-scoped).

```cpp
// Both CTAs execute this — hardware handles distribution
// Peer bit masked to 0 so complete_tx credits CTA 0's barrier
const auto copy_func = cute::SM100_TMA_2SM_LOAD_2D::copy;
copy_func(desc_ptr, barrier_ptr, cache_hint, smem_ptr, crd0, crd1);
// See: cutlass-4.4.2/include/cute/arch/copy_sm100_tma.hpp line 78

// Barrier setup:
// Leader CTA (rank 0): set expected bytes = per_cta_bytes × num_multicast_ctas
if (is_leader_cta && elect_one_sync())
    full_barriers[s]->arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast);
// Non-leader CTA (rank 1): just arrive, no byte expectation
if (!is_leader_cta && elect_one_sync())
    full_barriers[s]->arrive(0u);
// See: deepgemm-2.1.1/deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_1d2d.cuh lines 243-246
```

Key details:
- `SM100_TMA_2SM_LOAD_2D::copy` uses `cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global`
- The `cta_group::2` qualifier means hardware knows this is a 2-CTA cooperative load
- `Sm100MmaPeerBitMask` (bit 20 of the mbar address) controls which CTA's barrier
  receives the `complete_tx` credit. Cleared to 0 → CTA 0's barrier only.
- Leader expects `bytes × kNumMulticast` because hardware credits `complete_tx`
  once per destination CTA, all on the leader's barrier
- Non-leader calls `arrive(0u)` — it contributes arrival count, not byte count,
  to the cluster-scoped barrier. When total `complete_tx` = total `expect_tx`,
  the barrier flips for ALL CTAs.

#### Approach 2: Manual `multicast::cluster` PTX

One CTA issues multicast, receiving CTAs get data + barrier credits individually.

```
// CTA 0 issues multicast
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster
    [smem_addr], [tensor_desc, {crd0, crd1}], [mbar_addr], ctamask;
```

Barrier setup for manual multicast:
- The `complete_tx::bytes` credits **each receiving CTA's local mbarrier** separately
- Both issuing and receiving CTAs' mbarriers get credited for the bytes landing
  in their respective SMEM
- Each CTA must call `mbarrier.arrive_expect_tx` with the total bytes landing in
  ITS OWN SMEM (including bytes from multicast AND bytes from its own TMA loads)
- See: `docs/ptx-isa-9.2/09-instruction-set/` for `cp.async.bulk.tensor` semantics

Common deadlock with manual multicast: reducing the receiving CTA's `expect_tx` to exclude
multicast bytes (reasoning: "CTA 1 didn't issue the load"). This is wrong — multicast
DOES credit the receiving CTA's barrier via `complete_tx::bytes`. If `expect_tx` is too low,
the barrier flips before data arrives → data corruption. If `expect_tx` is too high
(not accounting for multicast credits), the barrier never flips → deadlock.

#### Which approach to use

| | `cta_group::2` cooperative | Manual `multicast::cluster` |
|---|---|---|
| Barrier accounting | Leader only, simpler | Both CTAs, error-prone |
| Who issues TMA | Both CTAs (same instruction) | One CTA only |
| Hardware support | SM100 native (cta_group::2 TMA) | SM90+ (multicast) |
| Reference | DeepGemm `sm100_bf16_gemm.cuh` | FlashAttention4 |
| Complexity | Lower | Higher |

For SM100 2-CTA clusters, prefer approach 1 (`cta_group::2`) — it's what DeepGemm uses
and the barrier accounting is simpler (one-sided).

### DeepGemm Multicast Example

DeepGemm uses 2-way multicast with cluster size 2:
```cpp
// Only leader CTA (rank 0) issues the multicast TMA
if (cute::block_rank_in_cluster() == 0) {
    cute::SM90_TMA_LOAD_MULTICAST_2D::copy(
        desc_ptr, barrier_ptr,
        (1 << num_tma_multicast) - 1,  // mask: 0b11 for 2-way
        cache_hint, smem_ptr, crd_0, crd_1);
}
```

DeepGemm dynamically disables multicast when the problem shape doesn't evenly divide the cluster size (`is_tma_multicast_valid` check).

### ThunderKittens Multicast Example

```cpp
// CLUSTER_SIZE = 2, each CTA loads its own half
tma::cluster::load_async(
    input_tiles[stage].A, g.A,
    {row_block_idx*2 + cta_id, i},
    tiles_arrived[stage],
    (uint16_t)(1<<cta_id),  // multicast mask: each CTA to itself
    0);
```

ThunderKittens uses `cta_group::2` for 2-CTA MMA on Blackwell, where each CTA loads its own A tile (no multicast on A) but B tiles are shared via multicast.

### Performance Impact

- **Bandwidth reduction**: N-way multicast reduces global memory reads by N for the multicast operand.
- **Optimized for**: `sm_90a`, `sm_100f`, `sm_100a`. May have reduced performance on other targets.
- **When to use**: When multiple CTAs in the cluster need the same data (e.g., shared K/V in attention, shared A or B rows/columns in GEMM).

---

## 6. Cluster Launch Control (CLC) -- Blackwell SM100

### What It Is

CLC is a Blackwell-specific (CC 10.0+) hardware feature that enables **dynamic persistent scheduling** via work-stealing. It replaces the static tile schedulers used in Hopper persistent kernels.

### The Problem CLC Solves

Traditional persistent kernels pre-assign tiles to SMs statically. If some SMs are occupied by other kernels or have variable execution times, this leads to load imbalance and wasted resources.

Non-persistent kernels (one CTA per tile) have good load balancing but high launch overhead, and they cannot amortize prologue costs (e.g., computing convolution coefficients, allocating TMEM).

CLC combines the best of both:
- Launch as many CTAs as tiles (like non-persistent)
- Workers stay alive and steal work from pending CTAs (like persistent)
- Hardware supports preemption (higher-priority kernels can interrupt)

### How It Works

1. **Kernel launches with full grid**: As many CTAs as output tiles.
2. **Worker CTAs start**: The GPU scheduler assigns initial CTAs to SMs.
3. **Work loop**: After completing its initial tile, a CTA issues a CLC query to try to cancel (steal) a pending CTA.
4. **Success**: The CTA receives the stolen CTA's blockIdx and processes that tile.
5. **Failure**: No more work available, or higher-priority kernel needs the SM. The CTA exits.

### API (PTX via libcu++)

```cpp
// Declare shared variables
__shared__ uint4 result;   // CLC response
__shared__ uint64_t bar;   // Synchronization barrier
int phase = 0;

// Initialize
if (threadIdx.x == 0)
    ptx::mbarrier_init(&bar, 1);

// Work-stealing loop
int bx = blockIdx.x;
while (true) {
    __syncthreads();  // protect result from overwrite

    // Submit async cancellation request (one thread)
    if (threadIdx.x == 0) {
        ptx::fence_proxy_async_generic_sync_restrict(
            ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);
        cg::invoke_one(cg::coalesced_threads(),
            ptx::clusterlaunchcontrol_try_cancel, &result, &bar);
        ptx::mbarrier_arrive_expect_tx(
            ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared,
            &bar, sizeof(uint4));
    }

    // Do current tile's work
    compute(bx);

    // Wait for CLC response
    while (!ptx::mbarrier_try_wait_parity(&bar, phase)) {}
    phase ^= 1;

    // Check result
    bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
    if (!success) break;

    bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);
}
```

### CLC with Clusters

When using CLC with thread block clusters:
- The cancellation steals an entire cluster (all CTAs in the cluster are cancelled together).
- Use `clusterlaunchcontrol_try_cancel_multicast` to broadcast the result to all CTAs in the cluster.
- Each CTA's shared memory gets the same result (the cancelled cluster's first CTA blockIdx).
- Each CTA adds its local cluster offset: `bx += cg::cluster_group::block_index().x`.
- Use `cluster.sync()` instead of `__syncthreads()` for synchronization.
- Cancellation request should come from a single thread across the ENTIRE cluster (not per-CTA).

### CLC Constraints

- After observing a failed cancellation, you MUST NOT submit another request (undefined behavior).
- You CAN submit a second request before observing the first (valid pipelining).
- Retrieving blockIdx from a failed cancellation is undefined behavior.

### CLC in CUTLASS

CUTLASS implements CLC through `PersistentTileSchedulerSm100` with:
- `PipelineCLCFetchAsync`: 3-stage pipeline to hide CLC query latency.
- `advance_to_next_work()`: Issues CLC query (scheduler warp).
- `get_current_work()`: Loads response from shared memory.

---

## 7. Cluster Synchronization

### Full Cluster Sync

```cpp
// Cooperative groups
auto cluster = cooperative_groups::this_cluster();
cluster.sync();  // blocks until all threads in all CTAs arrive

// PTX equivalent
barrier.cluster.arrive.aligned;
barrier.cluster.wait.aligned;

// CuTe / CUTLASS
cute::cluster_arrive_relaxed();
cute::cluster_wait();
// or
cute::cluster_sync();
```

### Split Arrive/Wait

```cpp
auto token = cluster.barrier_arrive();  // non-blocking arrive
// ... do independent work ...
cluster.barrier_wait(std::move(token));  // blocking wait
```

This is useful to overlap local computation with waiting for remote CTAs.

### When Cluster Sync Is Required

1. **Before first DSMEM access**: All CTAs must be running before any CTA reads another's shared memory.
2. **Before any CTA exits**: A CTA must not exit while another CTA is still reading its shared memory.
3. **After mbarrier init** (in cluster context): Use `fence.mbarrier_init.release.cluster` + `barrier.cluster` to ensure all barriers are visible.
4. **At iteration boundaries** (in persistent kernels): Ensure all CTAs have finished with the current tile before stealing next work.

### Memory Ordering with Cluster Barriers

`barrier.cluster.arrive` has `.release` semantics by default: all prior memory operations (loads and stores to shared, global) are guaranteed visible to other CTAs after the barrier completes.

`barrier.cluster.wait` has `.acquire` semantics by default: all subsequent memory operations will see the effects of operations prior to other CTAs' arrive.

For performance, use `.relaxed` arrive with an explicit fence when you need finer control:
```
fence.cluster.acq_rel;
barrier.cluster.arrive.relaxed.aligned;
```

---

## 8. Special Registers for Clusters

```
%cluster_ctaid.{x,y,z}   -- Position of this CTA within its cluster (0-based)
%cluster_nctaid.{x,y,z}  -- Cluster shape (number of CTAs per dimension)
%cluster_ctarank          -- Linear rank of CTA within cluster (0 to nctarank-1)
%cluster_nctarank         -- Total number of CTAs in cluster
%clusterid.{x,y,z}       -- Position of this cluster within the grid of clusters
%nclusterid.{x,y,z}      -- Grid shape in clusters
%is_explicit_cluster      -- 1 if launched with explicit cluster dims, 0 otherwise
```

CUDA C++ equivalents:
```cpp
// cooperative_groups
auto cluster = cooperative_groups::this_cluster();
cluster.block_rank()     // %cluster_ctarank
cluster.num_blocks()     // %cluster_nctarank
cluster.block_index()    // dim3(%cluster_ctaid.x, .y, .z)
cluster.dim_blocks()     // dim3(%cluster_nctaid.x, .y, .z)

// CuTe
cute::block_rank_in_cluster()  // %cluster_ctarank
cute::cluster_shape()          // (%cluster_nctaid.x, .y, .z)
```

**Note**: `blockIdx` still refers to the global grid position (in blocks), NOT the position within the cluster. `gridDim` is still in blocks, not clusters. This is for backward compatibility.

---

## 9. Common Pitfalls

### Deadlocks

1. **Accessing DSMEM before cluster.sync()**: Peer CTA may not exist yet. Always sync before first DSMEM access.

2. **CTA exits while peer reads its SMEM**: Peer's DSMEM access races with the exiting CTA. Always `cluster.sync()` before exit to ensure all remote operations are done.

3. **Wrong transaction bytes on cluster-scoped mbarrier**: If the expected bytes don't match actual TMA transfer size, the barrier never flips -> deadlock. With multicast, each CTA's mbarrier must account for the bytes arriving at THAT CTA.

4. **Multiple CTAs doing arrive_and_expect_tx on the same barrier**: Only one thread should set expected bytes. Multiple arrivals double-count -> premature flip -> data corruption.

5. **Grid not a multiple of cluster size**: Launch fails silently or with an error. Always check.

6. **Cluster sync in divergent code**: `barrier.cluster` requires ALL threads in the warp to execute it (when `.aligned`). Divergent warps -> hang.

### Occupancy Impact

Larger clusters reduce the number of simultaneously active clusters across the GPU:
- H100 with 132 SMs: cluster size 16 leaves 132 % 16 = 4 SMs unused. Cluster size 8 leaves 132 % 8 = 4 SMs unused. But larger clusters mean fewer clusters can run simultaneously per GPC.
- Occupancy should be computed with `cudaOccupancyMaxActiveClusters`, NOT `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.
- Benchmark result: cluster sizes > 8 can cause >20% overhead from SM underutilization.

### Shared Memory Limits

Each CTA still has its own shared memory (up to 228 KB on Hopper/Blackwell). DSMEM does NOT increase the per-CTA limit. It increases the TOTAL accessible memory across the cluster but does not help if a single CTA needs more than 228 KB.

Shared memory carveout affects occupancy per-SM. With large shared memory AND large clusters, occupancy can drop severely.

### Non-Portable Cluster Size

Cluster size > 8 requires:
```cpp
cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
```
This makes the kernel non-portable across GPUs. On GPUs that don't support the requested size, the launch fails.

---

## 10. cta_group::2 MMA Data Flow (Blackwell)

`tcgen05.mma.cta_group::2` uses a 2-CTA cluster to double the M dimension (128→256). Understanding the data flow is critical — getting it wrong produces silent correctness failures.

### Which CTA Loads What

For a 256×N output tile with BLOCK_K along the K dimension:

```
                    CTA 0 SMEM              CTA 1 SMEM
                ┌──────────────┐        ┌──────────────┐
  A (M×K):      │ A[0:127, :]  │        │ A[128:255, :]│    ← each CTA loads its own 128 M-rows
                └──────────────┘        └──────────────┘
  B (N×K):      │ B[0:N/2, :]  │        │ B[N/2:N, :]  │    ← each CTA loads HALF of N
                └──────────────┘        └──────────────┘
  Scale A:      │ SFA[0:127]   │        │ SFA[128:255] │    ← each CTA loads its own M-rows' scales
                └──────────────┘        └──────────────┘
  Scale B:      │ SFB[0:N/2]   │        │ SFB[N/2:N]   │    ← each CTA loads its half of B scales
                └──────────────┘        └──────────────┘
```

**Key rules:**
- **A**: Each CTA loads its own 128 M-rows. No sharing needed.
- **B**: Each CTA loads **half** the N dimension. The MMA reads B from **both** CTAs via DSMEM.
- **Scales**: Follow the same split as their parent matrices.
- **Only CTA 0 (leader) issues `tcgen05.mma`**. CTA 1 does NOT call MMA.
- **Both CTAs execute `tcgen05.cp`** to copy their local scales to TMEM.
- **Accumulator D in TMEM**: Shared across the 2-CTA group. Each CTA's 128 TMEM lanes map to its own 128 rows (CTA 0 → rows 0-127, CTA 1 → rows 128-255). Both CTAs use the same TMEM address — the hardware maps CTA rank to physical TMEM rows.

### TMA Load Coordinates

Each CTA uses its `cta_rank` to offset into different tiles:

```cpp
int cta_rank = get_ctarank();  // 0 or 1

// A: split along M
tma_load(sA, tma_a, mbar, k_offset, m_tile*2 + cta_rank);  // different M tiles

// B: split along N
tma_load(sB, tma_b, mbar, k_offset, n_tile*2 + cta_rank);  // different N tiles
```

The TMA tensor map for B must be created with per-CTA tile dimensions (N/2 per CTA, not full N).

### SMEM Descriptors for MMA

Both A and B descriptors point to local SMEM (`shared::cta`). The hardware handles DSMEM access to the peer CTA's B automatically. The descriptor leading/stride dimensions must match the **per-CTA** tile layout (128 M-rows for A, N/2 columns for B).

### Commit and Synchronization

```cpp
// Only CTA 0 issues MMA
if (cta_rank == 0 && threadIdx.x == 0) {
    tcgen05.mma.cta_group::2 ...;
    // Multicast commit signals BOTH CTAs' mbarriers
    tcgen05.commit.cta_group::2.mbarrier::arrive::one
        .shared::cluster.multicast::cluster.b64 [mbar], mask;
    // mask = 0x3 (both CTAs)
}

// BOTH CTAs wait on their local mbarrier for MMA completion
mbar_wait(local_mbar, phase);
```

### Epilogue

Each CTA reads its own 128 rows from TMEM using the same TMEM address. The hardware maps lane 0-127 to different physical rows per CTA:

```cpp
// Both CTAs execute this — same code, different physical TMEM rows
tcgen05.ld ... [tmem_d + col_offset];  // row = threadIdx.x (0-127)

// Store to different HBM rows
gD[(m_tile*256 + cta_rank*128 + row) * N + col] = result;
```

### Common Mistakes

1. **Both CTAs loading full B** — Most common bug. Each CTA must load only N/2 columns of B. If both load the full N, the MMA sees duplicated data via DSMEM and produces wrong results.

2. **Wrong TMA B descriptor** — The tensor map for B must tile at N/2 per CTA, not the full N.

3. **B scales not split** — Scale factors for B must follow the same N/2 split as B data.

4. **CTA 1 calling MMA** — Only the leader CTA (rank 0) issues `tcgen05.mma.cta_group::2`. CTA 1 must NOT call MMA.

5. **Using `scale_vec::4X` when reference uses `scale_vec::2X`** — These change the scale TMEM layout. Match the reference implementation.

---

## 11. When Clusters Help vs Hurt

### Clusters Help When

1. **Multiple CTAs need the same data**: TMA multicast saves bandwidth. In GEMM, CTAs sharing a row of A or column of B benefit from multicast. In attention, CTAs sharing K/V benefit.

2. **DSMEM avoids global memory round-trips**: If CTAs produce data consumed by neighboring CTAs, DSMEM (~7x faster than global) eliminates L2 traffic.

3. **Problem is large enough**: Need enough tiles to fill the GPU with clusters. If you have fewer tiles than SMs, clusters waste resources.

4. **2-CTA MMA on Blackwell**: `tcgen05.mma.cta_group::2` requires a cluster of >= 2. See Section 10 for the full data flow — each CTA loads half of B, and the hardware combines via DSMEM.

### Clusters Hurt When

1. **Problem is too small**: If the grid has fewer CTAs than the cluster size, the launch fails (grid must be a multiple of cluster dims). Even if it fits, small grids waste SMs.

2. **Causal / variable-length attention**: FlashAttention disables clusters for causal, local, split, paged KV, and variable-length modes because the irregular tile shapes prevent even cluster utilization.

3. **Odd tile counts**: FlashAttention requires `num_q_tiles % 2 == 0` for cluster size 2. If the tile count is odd, one cluster would have an empty CTA -> waste.

4. **High shared memory usage**: Large per-CTA shared memory combined with clusters can drop occupancy below useful thresholds.

5. **Non-uniform workloads**: If some CTAs finish much faster than others in the cluster, the fast CTAs must wait at `cluster.sync()` points.

### Decision Heuristics from Real Libraries

**FlashAttention (Hopper)**:
- Cluster enabled: `Arch == 90 && headDim >= 128 && !causal && !local && !split && !paged_kv && !varlen`
- Cluster size: 2 (along M)
- Runtime check: `num_q_tiles % 2 == 0`

**DeepGemm (Hopper/Blackwell)**:
- Multicast (cluster size 2): Used for normal GEMM and some grouped GEMMs
- Dynamic disable: Checks `is_tma_multicast_valid()` at runtime for each tile
- Multicast axis: A or B, chosen by heuristic to minimize L2 footprint

**CUTLASS (Blackwell)**:
- Common shapes: `<2,4,1>`, `<2,2,1>`, `<4,4,1>` for GEMM
- Uses CLC for dynamic persistent scheduling
- Log-swizzle tile scheduler aligns group size with cluster dims

**ThunderKittens (Blackwell)**:
- CLUSTER_SIZE=2 for NVFP4 GEMM (needed for `cta_group::2`)
- CLUSTER_SIZE=1 for simpler kernels

---

## 12. Real Examples with Performance Data

### Modular Blackwell GEMM (BF16, M=N=K=8192 on B200)

Progressive optimization stages with clusters:

| Stage | Technique | TFLOPS | % of cuBLAS |
|-------|----------|--------|-------------|
| Kernel 5 | Multicast + 2xSM MMA (no pipelining) | 360 | ~20% |
| Kernel 6 | + Pipelining + Warp Specialization | 1429 | ~81% |
| Kernel 7 | + Double-buffered Output | 1493 | ~85% |

The jump from 360 to 1429 TFLOPS shows that clusters alone (multicast + 2-CTA MMA) provide the foundation, but pipelining and warp specialization are needed to extract the performance.

### CUTLASS Blackwell NVFP4 GEMM

Uses cluster shape `<2,4,1>` (8 CTAs) with:
- MMA tile `<256,256,256>`
- `tcgen05.mma.blockscaled` with `cta_group::2`
- CLC-based dynamic scheduler
- 2x throughput vs FP8 from NVFP4

### FlashAttention (Hopper, FP16, d=128)

Uses cluster size 2 along M for forward pass:
- K and V loaded via TMA multicast (2-way), halving K/V bandwidth
- Q loaded without multicast (each CTA has different Q tiles)
- Benefit: reduces memory-boundedness for long-sequence attention

### DeepGemm (Hopper, FP8)

Uses cluster size 2 with TMA multicast:
- Multicast on A or B depending on which axis minimizes L2 footprint
- Dynamic fallback to no-multicast for boundary tiles
- Scheduler aligns group size (8 or 16) with multicast direction

### DSMEM Microbenchmarks (from literature)

- DSMEM load latency: ~7x lower than global memory round-trip
- DSMEM bandwidth: limited by SM-to-SM network within GPC
- Applications using DSMEM for inter-CTA communication: up to 2.3x speedup vs global memory
- Applications extending working set with DSMEM: up to 2.1x speedup

---

## 13. PTX Instruction Quick Reference

### Cluster Barriers
```
barrier.cluster.arrive{.sem}{.aligned};     // .sem = {.release, .relaxed}
barrier.cluster.wait{.acquire}{.aligned};
```

### DSMEM Load/Store
```
ld.shared::cluster.u32  reg, [addr];
st.shared::cluster.u32  [addr], reg;
atom.add.shared::cluster.u32  reg, [addr], val;
```

### Address Translation
```
mapa.shared::cluster.u32  dst, src_addr, target_cta_rank;
getctarank.shared::cluster.u32  rank, cluster_addr;
```

### Fences
```
fence.mbarrier_init.release.cluster;          // after mbarrier init
fence.acquire.sync_restrict::shared::cluster.cluster;  // acquire DSMEM
fence.release.sync_restrict::shared::cta.cluster;      // release to cluster
fence.proxy.async.shared::cluster;            // async proxy for DSMEM
```

### mbarrier with Cluster Scope
```
// Arrive on remote CTA's mbarrier
mbarrier.arrive.relaxed.cluster.shared::cluster.b64  state, [remote_bar];

// Set expected tx on remote mbarrier
mbarrier.expect_tx.relaxed.cluster.shared::cluster.b64  [remote_bar], bytes;

// Async store with mbarrier signal (to remote CTA)
st.async.shared::cluster.mbarrier::complete_tx::bytes.u32  [addr], val, [mbar];

// Reduction with mbarrier signal (to remote CTA)
red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32 [addr], val, [mbar];
```

### TMA Multicast
```
cp.async.bulk.tensor.2d.shared::cluster.global.tile
    .mbarrier::complete_tx::bytes.multicast::cluster
    [smem], [tensor_desc, {c0, c1}], [mbar], ctamask;

// With cta_group::2 (Blackwell)
cp.async.bulk.tensor.2d.shared::cluster.global.tile
    .mbarrier::complete_tx::bytes.cta_group::2.multicast::cluster
    [smem], [tensor_desc, {c0, c1}], [mbar], ctamask;
```

### CLC (Blackwell)
```
clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes
    [result], [mbar];

clusterlaunchcontrol.try_cancel.async.shared::cluster.mbarrier::complete_tx::bytes.multicast::cluster
    [result], [mbar];

clusterlaunchcontrol.query_cancel.is_canceled  pred, [result];
clusterlaunchcontrol.query_cancel.get_first_ctaid.{x,y,z}  reg, [result];
```

### Special Registers
```
mov.u32  %r, %cluster_ctarank;     // linear rank in cluster
mov.u32  %r, %cluster_nctarank;    // total CTAs in cluster
mov.u32  %r, %cluster_ctaid.x;     // position in cluster (x)
mov.u32  %r, %cluster_nctaid.x;    // cluster shape (x)
mov.u32  %r, %is_explicit_cluster; // 1 if explicit cluster launch
```
