# Memory Access Patterns

## Global Memory Coalescing

A warp (32 threads) issues memory requests together. The hardware coalesces these into as few 128-byte transactions as possible.

### Coalesced (good) — consecutive threads access consecutive addresses
```cpp
// Each thread reads one float from consecutive addresses
// → 1 transaction per 32 floats (128 bytes)
float val = input[blockIdx.x * blockDim.x + threadIdx.x];
```

### Strided (bad) — threads access with stride > 1
```cpp
// Stride-2 access: threads skip every other element
// → 2x the transactions needed
float val = input[2 * (blockIdx.x * blockDim.x + threadIdx.x)];
```

### Random (terrible) — threads access scattered addresses
```cpp
// Each thread accesses a random location
// → up to 32 separate transactions per warp
float val = input[indices[threadIdx.x]];
```

### Fix: Structure of Arrays (SoA) vs Array of Structures (AoS)

```cpp
// AoS (bad for GPU — strided access to each field)
struct Particle { float x, y, z, w; };
Particle particles[N];
float x = particles[tid].x;  // stride-4 access

// SoA (good — coalesced access to each field)
float x_arr[N], y_arr[N], z_arr[N], w_arr[N];
float x = x_arr[tid];  // coalesced
```

### Coalescing in Production: Flash Attention SM80 Attention Layout

Flash Attention ensures coalesced access by using `Stride<int64_t, _1, ...>` where the
innermost dimension (head dim d) has stride 1, meaning consecutive elements within a
head are contiguous in memory. The thread layout for global memory loads is then
arranged so that consecutive threads in a warp map to consecutive elements along
that contiguous dimension:

```cpp
// Source: flash-attention-fa4-v4.0.0.beta8/hopper/mainloop_fwd_sm80.hpp

// Stride _1 on the d dimension ensures contiguous memory along head dim
using StrideQK = cute::Stride<int64_t, _1, int64_t, int64_t>;  // (seqlen, d, head, batch)

// Each thread loads 128 bits (uint128_t) = 8 half elements at once
static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);  // = 8 for fp16

// Thread layout: consecutive threads cover consecutive chunks along K (head dim)
// e.g. for 128 threads and 8 threads per row: 16 rows of 8 threads
using GmemLayoutAtom = Layout<Shape <Int<NumMmaThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                              Stride<Int<kGmemThreadsPerRow>, _1>>;

// Combined tiled copy: each thread issues one 128-bit load of 8 consecutive elements
using GmemTiledCopyQKV = decltype(
    make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                    GmemLayoutAtom{},
                    Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
```

The key coalescing guarantee: because consecutive threads map to consecutive chunks
along the stride-1 dimension, all threads in a warp access a contiguous 128-byte
region, resulting in a single memory transaction.

## Vectorized Loads

### Production: CUTLASS 16-byte Predicated Vectorized Load (uint4 / 128-bit)

CUTLASS implements predicated 128-bit vectorized loads from global memory with
inline PTX. The data is loaded as `uint4` (4x 32-bit = 128 bits = 16 bytes).
The predicate guard allows masking off out-of-bounds threads without branching:

```cpp
// Source: cutlass-4.4.2/include/cutlass/arch/memory.h

template <typename AccessType>
struct global_load<AccessType, 16, CacheOperation::Always> {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    uint4 &data = reinterpret_cast<uint4 &>(D);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %5, 0;\n"
        "  mov.b32 %0, %6;\n"           // initialize with current value (for OOB case)
        "  mov.b32 %1, %7;\n"
        "  mov.b32 %2, %8;\n"
        "  mov.b32 %3, %9;\n"
        "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"  // 128-bit vectorized load
        "}\n"
        : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
        : "l"(ptr), "r"((int)pred_guard),
          "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));
  }
};
```

### Production: CUTLASS 32-byte (256-bit) Double Vectorized Load

For even wider loads, CUTLASS issues two back-to-back `ld.global.v4.u32` instructions
loading 256 bits total, with an L2 prefetch hint on Ampere+:

```cpp
// Source: cutlass-4.4.2/include/cutlass/arch/memory.h

template <typename AccessType>
struct global_load<AccessType, 32, CacheOperation::Always> {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    uint4 *data = reinterpret_cast<uint4 *>(&D);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %9, 0;\n"
        "  mov.b32 %0, %10;\n"
        "  mov.b32 %1, %11;\n"
        "  mov.b32 %2, %12;\n"
        "  mov.b32 %3, %13;\n"
        "  mov.b32 %4, %14;\n"
        "  mov.b32 %5, %15;\n"
        "  mov.b32 %6, %16;\n"
        "  mov.b32 %7, %17;\n"
        "  @p ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%8];\n"   // first 128 bits
        "  @p ld.global.L2::128B.v4.u32 {%4, %5, %6, %7}, [%18];\n"  // next 128 bits (+16 bytes)
        "}\n"
        : "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z), "=r"(data[0].w),
          "=r"(data[1].x), "=r"(data[1].y), "=r"(data[1].z), "=r"(data[1].w)
        : "l"(ptr), "r"((int)pred_guard),
          "r"(data[0].x), "r"(data[0].y), "r"(data[0].z), "r"(data[0].w),
          "r"(data[1].x), "r"(data[1].y), "r"(data[1].z), "r"(data[1].w),
          "l"(((uint8_t *)ptr) + 16));
  }
};
```

Note the `L2::128B` hint on Ampere+ (`CUTLASS_ENABLE_L2_PREFETCH`), which tells
the L2 cache to prefetch a 128-byte cache line. The `mov.b32` instructions before
the guarded loads initialize outputs so OOB threads preserve their existing values.

### Production: Flash Attention SM80 cp.async with 128-bit Loads

Flash Attention FA2 on SM80 uses `SM80_CP_ASYNC_CACHEGLOBAL` to do 128-bit async
copies from global memory directly to shared memory, bypassing registers:

```cpp
// Source: flash-attention-fa4-v4.0.0.beta8/csrc/flash_attn/src/kernel_traits.h

// We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
// from the same address by the same threadblock. This is slightly faster.
using Gmem_copy_struct = std::conditional_t<
    Has_cp_async,
    SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,  // 128-bit async copy, bypass L1
    AutoVectorizingCopyWithAssumedAlignment<128>
>;
using GmemTiledCopyQKV = decltype(
    make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                    GmemLayoutAtom{},
                    Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read (128 bits for fp16)
```

## Shared Memory Bank Conflicts

Shared memory has 32 banks (one per warp lane). Bank index = `(address / 4) % 32`.

### No conflict — each thread accesses a different bank
```cpp
// Thread i reads shared[i] → each thread hits a different bank
float val = shared_mem[threadIdx.x];
```

### 2-way conflict — two threads hit the same bank
```cpp
// Stride-2: thread 0 and 16 both hit bank 0
float val = shared_mem[2 * threadIdx.x];
```

### 32-way conflict (broadcast) — all threads hit same bank
```cpp
// All threads read the same address → broadcast (actually free!)
float val = shared_mem[0];
```

### Fix: Padding to avoid conflicts
```cpp
// Without padding: column access has 32-way bank conflicts
__shared__ float tile[32][32];
float val = tile[threadIdx.x][col];  // threads in a warp access same column = same bank

// With padding: add 1 element per row to shift banks
__shared__ float tile[32][32 + 1];  // +1 padding
float val = tile[threadIdx.x][col];  // now each thread hits a different bank
```

### Fix: Swizzle pattern (used by CUTLASS/CuTe)

Instead of padding, XOR the row and column indices to scatter bank accesses:
```cpp
int swizzled_col = col ^ (row % 32);
float val = tile[row][swizzled_col];
```

### Production: Flash Attention SM80 Swizzled Shared Memory Layout

Flash Attention uses CuTe's `Swizzle` to avoid bank conflicts when loading
Q/K/V tiles into shared memory. The swizzle parameters are computed from
the tile dimensions:

```cpp
// Source: flash-attention-fa4-v4.0.0.beta8/hopper/mainloop_fwd_sm80.hpp

static constexpr int kBytePerRow = kHeadDim * sizeof(Element);
static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
static constexpr int kSwizzle = kBlockKGmem == 128 ? 4 : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1));
static constexpr int kSwizzleBase = sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);

using SmemLayoutAtomQKV = decltype(
    composition(Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{},
                Layout<Shape<_8, Int<kBlockKGmem>>,
                       Stride<Int<kBlockKGmem>, _1>>{}));
using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQKV{}, select<0, 2>(TileShape_MNK{})));
```

## L2 Cache Optimization

### Persistent L2 Cache (Ampere+)

Reserve L2 cache for frequently accessed data:
```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
size_t l2_size = prop.l2CacheSize;

// Reserve 50% of L2 for a specific data window
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr = d_data;
attr.accessPolicyWindow.num_bytes = min(data_size, l2_size / 2);
attr.accessPolicyWindow.hitRatio = 1.0f;
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

### Cache Hints in PTX

```cpp
// Streaming load — don't pollute cache (for data read once)
asm volatile("ld.global.cs.b32 %0, [%1];" : "=r"(val) : "l"(ptr));

// Cache at all levels (default)
asm volatile("ld.global.ca.b32 %0, [%1];" : "=r"(val) : "l"(ptr));

// Last use — evict after this access (CUTLASS CacheOperation::LastUse)
asm volatile("ld.global.lu.v4.u32 {%0, %1, %2, %3}, [%4];" ...);
```

### Production: CUTLASS L2 Prefetch Hints in Vectorized Loads

On Ampere+ (SM75+), CUTLASS automatically appends `L2::128B` hints to global
loads and cp.async instructions. This tells the L2 cache to prefetch a full
128-byte cache line, improving bandwidth utilization:

```cpp
// Source: cutlass-4.4.2/include/cutlass/arch/memory.h

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11)) && \
    defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
  #define CUTLASS_ENABLE_L2_PREFETCH 1
#endif

// Used in global loads:
"  @p ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];\n"  // with L2 hint

// Used in cp.async:
"  @p cp.async.ca.shared.global.L2::128B [%1], [%2], %3;\n"   // with L2 hint
"  @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"   // cache-global variant
```

### Production: TMA L2 Cache Eviction Hints (Hopper/Blackwell)

On SM90+, TMA instructions accept a 64-bit cache hint that controls L2 eviction
policy. CUTLASS defines these hints as an enum. Flash Attention uses them
strategically: K/V tiles (streamed repeatedly) get `EVICT_LAST` (keep in cache),
while Q tiles (read once per CTA) get `EVICT_FIRST` (evict quickly):

```cpp
// Source: cutlass-4.4.2/include/cute/arch/copy_sm90_desc.hpp

enum class CacheHintSm90 : uint64_t {
  EVICT_NORMAL = 0x1000000000000000,
  EVICT_FIRST  = 0x12F0000000000000,   // evict this data before other data
  EVICT_LAST   = 0x14F0000000000000,   // keep this data in cache longest
};
```

```cpp
// Source: flash-attention-fa4-v4.0.0.beta8/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp

// K/V are streamed over repeatedly — keep in L2 cache as long as possible
copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write),
     mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
     tKgK_TMA(_, n_block_idx, bidb_kv_idx), tKsK_TMA(_, smem_pipe_write.index()));

// Q is read once per CTA — evict early to free L2 space for K/V
copy(params.tma_load_Q.with(
     reinterpret_cast<typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(
         shared_storage.pipelines.barrier_Q),
     0 /*mcast_mask*/,
     !Split ? TMA::CacheHintSm90::EVICT_FIRST : TMA::CacheHintSm90::EVICT_LAST),
     tQgQ, tQsQ);
```

## Global to Shared Memory Async Copy

### cp.async (Ampere / SM80+)

Copy from global to shared memory without going through registers.

### Production: CUTLASS cp.async with Predication

CUTLASS implements `cp.async` as a template with compile-time size (4/8/16 bytes)
and cache operation. The predicate `pred_guard` skips the copy at instruction
level using `@p`, avoiding warp divergence:

```cpp
// Source: cutlass-4.4.2/include/cutlass/arch/memory_sm80.h

template <int SizeInBytes>
struct cp_async<SizeInBytes, CacheOperation::Always> {
  CUTLASS_DEVICE
  cp_async(void *smem_ptr, void const *global_ptr, bool pred_guard = true) {
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16),
              "Size is not supported");
    unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %0, 0;\n"
        "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int)pred_guard),
        "r"(smem_int_ptr), "l"(global_ptr), "n"(SizeInBytes));
  }
};
```

### Production: CUTLASS cp.async with Zero Fill (Boundary Handling)

For tiles at matrix boundaries where some elements are out of bounds, `cp_async_zfill`
writes zeros to shared memory for out-of-bounds elements instead of garbage. The
hardware handles this by passing `src_in_bytes=0` when the predicate is false:

```cpp
// Source: cutlass-4.4.2/include/cutlass/arch/memory_sm80.h

template <int SizeInBytes>
struct cp_async_zfill<SizeInBytes, CacheOperation::Always> {
  CUTLASS_DEVICE
  cp_async_zfill(void *smem_ptr, void const *global_ptr, bool pred_guard) {
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16),
              "Size is not supported");
    unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
    int src_in_bytes = (pred_guard ? SizeInBytes : 0);  // 0 bytes → hardware writes zeros
    asm volatile(
      "cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
      "l"(global_ptr), "n"(SizeInBytes), "r"(src_in_bytes));
  }
};
```

### Production: CUTLASS cp.async Commit and Wait Groups

The cp.async pipeline uses commit groups to batch transfers and wait on them.
`cp.async.commit_group` marks a group boundary; `cp.async.wait_group N` blocks
until all but the N most recent groups have completed:

```cpp
// Source: cutlass-4.4.2/include/cutlass/arch/memory_sm80.h

/// Establishes an ordering w.r.t previously issued cp.async instructions. Does not block.
CUTLASS_DEVICE
void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

/// Blocks until all but <N> previous cp.async.commit_group operations have committed.
template <int N>
CUTLASS_DEVICE void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

/// Blocks until all previous cp.async.commit_group operations have committed.
template <>
CUTLASS_DEVICE void cp_async_wait<0>() {
  asm volatile("cp.async.wait_all;\n" ::);
}
```

### Production: Flash Attention SM80 cp.async Pipeline

Flash Attention on SM80 uses cp.async with CACHEGLOBAL policy and multi-stage
pipelining to overlap loads with computation:

```cpp
// Source: flash-attention-fa4-v4.0.0.beta8/hopper/mainloop_fwd_sm80.hpp

// Each thread independently copies a 128-bit chunk with cp.async
using GmemCopyAtom = Copy_Atom<std::conditional_t<
    Has_cp_async,
    SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>,   // 128-bit cp.async with zero fill
    AutoVectorizingCopyWithAssumedAlignment<128>
>, Element>;

// Load K tile with boundary masking: copy only valid rows, zero-fill the rest
flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false,
            /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/true>(
    gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK_cur, t0KVcKV, tKVpKV, seqlenk_row_limit);

// Commit the group after Q is loaded, then overlap K/V loads with Q processing
cute::cp_async_fence();

// Later, wait for the earliest pipeline stage to drain before reusing its smem
flash::cp_async_wait<kStages * 2 - 1>();
```

## TMA --- Tensor Memory Accelerator (Hopper / SM90+)

Hardware unit that handles multi-dimensional async copies with no address
arithmetic in the kernel. A single TMA instruction can copy an entire 2D/3D/4D/5D
tile from global memory to shared memory.

### Production: CUTLASS TMA Load (PTX Level)

This is the actual PTX emitted by CUTLASS for a 2D TMA load on SM90. The TMA
descriptor (`gmem_int_desc`) encodes the tensor layout, the mbarrier tracks
completion, and the cache hint controls L2 eviction:

```cpp
// Source: cutlass-4.4.2/include/cute/arch/copy_sm90_tma.hpp

struct SM90_TMA_LOAD_2D {
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "l"(cache_hint)
      : "memory");
  }
};
```

### Production: DeepGemm TMA Copy Wrapper with Multicast

DeepGemm (used in DeepSeek inference) wraps TMA with multicast support.
When running with clusters, only CTA 0 issues the multicast TMA:

```cpp
// Source: deepgemm-2.1.1.post3/deep_gemm/include/deep_gemm/common/sm90_utils.cuh

__device__ __forceinline__ void
tma_copy(void const* desc_ptr, uint64_t* barrier_ptr, void* smem_ptr,
         const uint32_t& crd_0, const uint32_t& crd_1,
         const uint32_t& num_tma_multicast = 1) {
    constexpr auto cache_hint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
    if (num_tma_multicast == 1) {
        cute::SM90_TMA_LOAD_2D::copy(desc_ptr, barrier_ptr, cache_hint, smem_ptr, crd_0, crd_1);
    } else if (cute::block_rank_in_cluster() == 0) {
        cute::SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr, barrier_ptr,
            (1 << num_tma_multicast) - 1, cache_hint, smem_ptr, crd_0, crd_1);
    }
}
```

### Production: Flash Attention TMA Descriptor Setup and Load

Flash Attention FA4 on Hopper creates TMA descriptors on the host and uses them
in the kernel. The `tma_load_K` descriptor encodes the K tensor's memory layout
so the kernel needs only tile coordinates, not pointer arithmetic:

```cpp
// Source: flash-attention-fa4-v4.0.0.beta8/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp

// --- Host side: create TMA descriptor ---
Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
TMA_K tma_load_K = make_tma_copy_B_sm90(
    GmemTiledCopyKV{},
    mK,
    take<0, 2>(SmemLayoutK{}),
    TileShape_MNK{},
    ClusterShape{});  // mcast along M mode for this N load

// --- Device side: prefetch TMA descriptor then issue loads ---
cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());

// Load K tile: single instruction copies entire tile, with L2 eviction hint
copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write),
     mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
     tKgK_TMA(_, n_block_idx, bidb_kv_idx),
     tKsK_TMA(_, smem_pipe_write.index()));
```

### Production: Flash Attention TMA Store (Epilogue)

The epilogue uses TMA store to write the output O tile from shared memory back
to global memory. Only one elected thread per warp issues the TMA store:

```cpp
// Source: flash-attention-fa4-v4.0.0.beta8/hopper/epilogue_fwd.hpp

Tensor mO = params.tma_store_O.get_tma_tensor(params.shape_O)(_, _, bidh, bidb, split_idx);
Tensor gO = local_tile(mO, select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));
auto block_tma_O = params.tma_store_O.get_slice(_0{});
Tensor tOgO = block_tma_O.partition_D(gO);
Tensor tOsO = block_tma_O.partition_S(sO);

if (cute::elect_one_sync()) {
    cute::copy(params.tma_store_O, tOsO, tOgO);    // TMA store: smem → gmem
    tma_store_arrive();
    tma_store_wait<0>();                              // wait for store to complete
}
```

### Production: Bulk Async Reduce via TMA (Backward Pass)

Flash Attention's backward pass uses `cp.reduce.async.bulk` to atomically
accumulate gradient tiles from shared memory into global memory:

```cpp
// Source: flash-attention-fa4-v4.0.0.beta8/hopper/copy_sm90_bulk_reduce.hpp

struct SM90_BULK_REDUCE_ADD {
  CUTE_HOST_DEVICE static void
  copy(float const* smem_ptr,
       float      * gmem_ptr, int32_t store_bytes, uint64_t cache_hint)
  {
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
      "cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.add.f32 [%0], [%1], %2, %3;\n"
      :
      : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes), "l"(cache_hint)
      : "memory");
  }
};

// Usage in backward pass epilogue:
SM90_BULK_REDUCE_ADD::copy(
    raw_pointer_cast(sdQ(_, warpgroup_idx).data()),
    raw_pointer_cast(gdQaccum(_, warpgroup_idx, m_block).data()),
    dQ_TMA_num_bytes,
    static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_LAST));
```

TMA advantages:
- No address calculation in registers
- Hardware handles 2D/3D/4D/5D tiling
- Supports multicast to multiple CTAs in a cluster
- Reduces register pressure vs manual indexing
- L2 cache hints control eviction policy per-transfer
- Available on Hopper (H100) and Blackwell (B200). Enhanced on Blackwell with wider transfer support.

### Tensor Memory --- TMEM (Blackwell only)

Blackwell's 5th-gen Tensor Cores have a dedicated **Tensor Memory** (TMEM) --- a new on-chip memory space separate from shared memory. Tensor core operands can be read/written directly from TMEM, reducing data movement. Accessed via `tcgen05` PTX instructions. See `docs/architecture/blackwell-b200.md` for details.
