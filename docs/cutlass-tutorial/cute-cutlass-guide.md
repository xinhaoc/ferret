# CuTe and CUTLASS Guide

## What is CUTLASS / CuTe

CUTLASS is NVIDIA's C++ CUDA template library for high-performance GEMM. CuTe (CUDA Tensors) is its core abstraction layer since CUTLASS 3.0 — a header-only library for defining hierarchically multidimensional layouts of threads and data.

CuTe replaced all bespoke iterator/thread map types from CUTLASS 2.x with one vocabulary type: `cute::Layout`. Key benefit: "If the code compiles, it's probably correct."

---

## Part 1: CuTe Core Concepts

### Static vs Dynamic Integers

- Dynamic: `int`, `size_t`
- Static (compile-time): `cute::Int<N>`, aliases `_1`, `_2`, `_3`
- Static integers produce static results through math operators
- Any dynamic can be replaced with static and vice versa

### IntTuple

Recursive: an integer, or a tuple of IntTuples.

- `rank(x)`: number of top-level elements
- `get<I>(x)`: I-th element
- `depth(x)`: nesting depth
- `size(x)`: product of all elements

Shape and Stride are both IntTuple concepts.

### Layout — The Core Abstraction

`Layout = (Shape, Stride)` — maps coordinate space to index space.

```cpp
// Column-major 2x4
Layout s2xs4 = make_layout(make_shape(Int<2>{},Int<4>{}));          // (_2,_4):(_1,_2)
// Row-major 2x4
Layout s2x4_row = make_layout(make_shape(Int<2>{},4), LayoutRight{}); // (_2,4):(4,_1)
// Custom strides
Layout custom = make_layout(make_shape(Int<2>{},4), make_stride(Int<12>{},Int<1>{}));
```

Key operations:

- `rank(layout)`, `shape(layout)`, `stride(layout)`
- `size(layout)`: domain size (product of shape)
- `cosize(layout)`: codomain size (for memory allocation)
- `layout(m, n)`: maps 2D coordinate to 1D index
- `layout(i)`: 1D coordinate traverses in colexicographical order

Matrix convention:

- `(4,2):(1,4)` — 4x2 column-major
- `(4,2):(2,1)` — 4x2 row-major

Layout manipulation: `select<I,J>`, `group`, `flatten`, `coalesce`

### Layout Algebra

**Composition** `composition(A, B)`: `R(c) = A(B(c))`. Result is a Layout.

**Complement** `complement(A, M)`: elements NOT touched by A, up to size M.

**Logical Divide** `logical_divide(A, B)`: splits A into tile + rest.

Convenience flavors:

```
logical_divide : ((TileM,RestM), (TileN,RestN), ...)
zipped_divide  : ((TileM,TileN), (RestM,RestN,...))    <- most common
tiled_divide   : ((TileM,TileN), RestM, RestN, ...)
```

**Logical Product** `logical_product(A, B)`: reproduces A according to B.

### Tensor

`Tensor = Engine (iterator/array) + Layout`

```cpp
// Nonowning views
Tensor gmem = make_tensor(make_gmem_ptr(A), make_shape(Int<8>{}, 16));
Tensor smem = make_tensor(make_smem_ptr(smemA), smem_layout);

// Owning (register memory, static layouts only)
Tensor rmem = make_tensor<float>(Shape<_4,_8>{});
Tensor rmem_like = make_tensor_like(some_tensor);
```

Memory tags (`make_gmem_ptr`, `make_smem_ptr`) enable dispatch to optimized copy instructions.

Slicing: `A(2, _)` keeps columns, slices row 2. `A(_, 5)` keeps rows, slices column 5.

### Algorithms

`copy(src, dst)` or `copy(copy_atom, src, dst)` — dispatches to optimized instructions based on memory space.

`gemm(A, B, C)` or `gemm(mma, A, B, C)` — matrix multiply-accumulate.
Convention: K is always rightmost mode; V (vector) is leftmost.

### MMA Atoms and TiledMMA

MMA Atom = PTX instruction wrapper + traits (types, shapes, thread/value layouts).

TiledMMA combines atoms into larger patterns:

```cpp
TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                               Layout<Shape<_2,_2>>{},
                               Tile<_64,_64,_16>{});
```

Using TiledMMA for partitioning:

```cpp
ThrMMA thr_mma = mma.get_slice(threadIdx.x);
Tensor tCsA = thr_mma.partition_A(sA);        // (MMA,MMA_M,MMA_K)
Tensor tCsB = thr_mma.partition_B(sB);        // (MMA,MMA_N,MMA_K)
Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // (MMA,MMA_M,MMA_N)
cute::gemm(mma, tCsA, tCsB, tCrC);
```

### Copy Atoms and TiledCopy

Analogous to MMA atoms but for data movement:

```cpp
TiledCopy copyA = make_tiled_copy(
    Copy_Atom<UniversalCopy<uint128_t>, float>{},
    Layout<Shape<_32,_8>>{},    // thread layout
    Layout<Shape<_4,_1>>{});    // values per thread
```

---

## Part 2: Writing a GEMM with CuTe

### Mode Conventions

A is (M,K), B is (N,K), C is (M,N). "M-major" = stride-1 in M.

### Step-by-Step

**Step 1: Define full tensors**

```cpp
Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);  // (M,K)
Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);  // (N,K)
Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);  // (M,N)
```

**Step 2: CTA partitioning**

```cpp
auto cta_tiler = make_shape(bM, bN, bK);
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // (BLK_M,BLK_K,k)
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});  // (BLK_M,BLK_N)
```

**Step 3: Shared memory**

```cpp
__shared__ TA smemA[cosize_v<ASmemLayout>];
Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
```

**Step 4: Copy partitioning (threads for loading)**

```cpp
auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
Tensor tAgA = local_partition(gA, tA, threadIdx.x);
Tensor tAsA = local_partition(sA, tA, threadIdx.x);
```

**Step 5: Math partitioning (threads for compute)**

```cpp
auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));
Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{});
Tensor tCrC = make_tensor_like(tCgC);  // register accumulators
```

Naming convention: `tCsA` = "partitioning tC applied to tensor sA"

**Step 6: Mainloop**

```cpp
for (int k = 0; k < K_TILES; ++k) {
    copy(tAgA(_,_,k), tAsA);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();
    gemm(tCsA, tCsB, tCrC);
    __syncthreads();
}
```

---

## Part 3: CUTLASS 3.x Programming Model

### Architecture Hierarchy

| Level | Class | Purpose |
|---|---|---|
| Device | `GemmUniversalAdapter` | Host-side launch handle |
| Kernel | `GemmUniversal` | Grid scheduling, tile dispatch |
| Collective | `CollectiveMma`, `Epilogue` | K-tile mainloop, epilogue |
| Tiled MMA/Copy | `TiledMma`, `TiledCopy` | Warp/block operations |
| Atom | `Mma_Atom`, `Copy_Atom` | Single instruction wrappers |

### Assembling a Kernel

```cpp
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TilesShape, ClusterShape,
    StageCountAuto, KernelScheduleAuto
>::CollectiveOp;

using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<...>;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

### Key Terminology

| Term | Meaning |
|---|---|
| Fragment | Register array storing a thread's part of a tile |
| Tile | Partition with compile-time-known extents |
| Mainloop | K-tile loop over input matrix tiles |
| Epilogue | Post-MMA ops (scale, bias, activation, writeback) |
| Warp-specialized | Different warps have different roles (producer/consumer) |
| Persistent kernel | One thread block processes multiple tiles via scheduler |

---

## Quick Reference: Essential Types

| Type | Description |
|---|---|
| `Int<N>`, `_1`, `_2` | Compile-time integers |
| `cute::Layout<Shape, Stride>` | Core: coordinate to index mapping |
| `cute::Tensor<Engine, Layout>` | Multidimensional array |
| `cute::TiledMma` | Tiled MMA operation |
| `cute::TiledCopy` | Tiled copy operation |
| `Step<_1, X, _1>` | Mode projection |
| `_` (Underscore) | Slice: retain a mode |

## Quick Reference: Essential Functions

| Function | Description |
|---|---|
| `make_layout(shape, stride)` | Create Layout |
| `make_tensor(ptr, layout)` | Create nonowning Tensor |
| `make_tensor<T>(layout)` | Create owning Tensor (registers) |
| `make_gmem_ptr(p)` / `make_smem_ptr(p)` | Tag pointer with memory space |
| `local_tile(tensor, tiler, coord, step)` | CTA-level tiling |
| `local_partition(tensor, layout, idx)` | Thread-level partitioning |
| `size(x)` / `size<I>(x)` | Total/mode size |
| `cosize(layout)` | Codomain size (for allocation) |
| `composition(a, b)` | Layout composition |
| `zipped_divide(a, b)` | Divide into tile + rest |
| `copy(src, dst)` | Copy between tensors |
| `gemm(mma, a, b, c)` | Matrix multiply-accumulate |

---
