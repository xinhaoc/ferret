# Blockscaled MMA for FP4 (SM100)

This doc covers the `tcgen05.mma.blockscaled` instruction family for FP4 GEMM on Blackwell (SM100). Use this when writing NVFP4 or MXFP4 kernels from scratch in CUDA C++ with inline PTX.

## Overview

Blockscaled MMA multiplies FP4 (E2M1) matrices with per-block scale factors applied by the hardware. The scale factors live in TMEM alongside the accumulator. This is fundamentally different from regular `tcgen05.mma` (BF16/FP8) which has no scale operands.

**Operation:** `D = (A * scale_A) * (B * scale_B) + D`

**Two variants:**
- `kind::mxf4nvf4` — NVFP4: supports both UE4M3 (block-16) and UE8M0 (block-32) scales
- `kind::mxf4` — MXFP4: UE8M0 (block-32) scales only

NVFP4 is the more common format (used by cuBLAS, TensorRT-LLM, ThunderKittens).

## PTX Instruction Syntax

### NVFP4 with UE4M3 scales (block-16)

```
tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X
    [d-tmem], a-desc, b-desc, idesc, [sfa-tmem], [sfb-tmem], enable-input-d;

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X
    [d-tmem], a-desc, b-desc, idesc, [sfa-tmem], [sfb-tmem], enable-input-d;
```

`.block16` is an alias for `.scale_vec::4X` (PTX ISA 8.8+).

### MXFP4 with UE8M0 scales (block-32)

```
tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X
    [d-tmem], a-desc, b-desc, idesc, [sfa-tmem], [sfb-tmem], enable-input-d;
```

`.block32` is an alias for `.scale_vec::2X`.

### Inline PTX example (cta_group::1, NVFP4)

```cpp
__device__ __forceinline__ void tcgen05_mma_nvfp4(
    uint32_t d_tmem,      // TMEM address for accumulator D
    uint64_t a_desc,      // SMEM descriptor for A
    uint64_t b_desc,      // SMEM descriptor for B
    uint32_t idesc,       // Instruction descriptor (see below)
    uint32_t sfa_tmem,    // TMEM address for scale factors A
    uint32_t sfb_tmem,    // TMEM address for scale factors B
    uint32_t scale_d)     // 1 = accumulate (D += ...), 0 = overwrite
{
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
           "r"(sfa_tmem), "r"(sfb_tmem), "r"(scale_d));
}
```

### Key differences from regular tcgen05.mma

| Aspect | Regular (f16/f8f6f4) | Blockscaled (mxf4/mxf4nvf4) |
|--------|----------------------|------------------------------|
| Extra operands | `{disable-output-lane}` mask | `[sfa-tmem]`, `[sfb-tmem]` addresses |
| Scale input D | Optional power-of-2 scaling | Not available |
| K per instruction | 16 (f16), 32 (f8) | 64 (dense), 128 (sparse) |
| M for cta_group::1 | 64 or 128 | 128 only |
| .ws mode | Supported | NOT supported |
| Transpose A/B | Supported | NOT supported |
| Accumulator type | FP32 or FP16 | FP32 only |
| TMEM regions | D only | D + scale_A + scale_B |
| Throughput (B200) | 2.25 PFLOPS (FP16), 4.5 PFLOPS (FP8) | 9 PFLOPS (FP4) |

## Scale Factor Data Types

### UE4M3 (unsigned FP8 E4M3) — NVFP4 default

- 8 bits: 4 exponent + 3 mantissa, unsigned (always positive)
- CUTLASS type: `cutlass::float_ue4m3_t`
- CUDA type: `__nv_fp8_e4m3` (same storage, different semantics — unsigned)
- Max value: 448
- One scale per 16 elements along K
- Used with `.scale_vec::4X` / `.block16`

### UE8M0 (unsigned FP8 E8M0) — MXFP4 / OCP MX standard

- 8 bits: 8 exponent + 0 mantissa, power-of-2 only
- CUTLASS type: `cutlass::float_ue8m0_t`
- One scale per 32 elements along K
- Used with `.scale_vec::2X` / `.block32`

### FP4 E2M1 — The data element type

- 4 bits: 1 sign + 2 exponent + 1 mantissa
- Only 15 distinct values (7 positive, 7 negative, zero)
- Max value: 6.0
- Two elements packed per byte (`__nv_fp4x2_e2m1` / `fp4e2m1_2`)
- CUTLASS type: `cutlass::float_e2m1_t`

### Two-level scaling (NVFP4)

NVFP4 uses two levels of scaling:
1. **Per-block FP8 E4M3 scale**: one per 16 elements along K, applied by hardware via TMEM
2. **Per-tensor FP32 global scale**: one per entire tensor, applied manually in epilogue

Dequantization: `x_float = x_fp4 * scale_block * scale_global`

In the kernel, the hardware applies `scale_block` automatically during MMA. You apply `scale_global` in the epilogue:
```cpp
float global_scale = A_global_scale * B_global_scale;
// After loading accumulator from TMEM:
result *= global_scale;
```

## Instruction Descriptor Encoding

The blockscaled instruction descriptor is 32 bits with a DIFFERENT layout from the regular descriptor:

```
Bit(s)   Field              Values for NVFP4
-------  -----------------  -----------------------------------------
[0,2)    sparse_id2         0 (dense)
[2]      sparse_flag        0 (dense), 1 (sparse)
[3]      reserved           0
[4,6)    b_sf_id            SFB data ID (0 for scale_vec::4X)
[6]      reserved           0
[7,10)   a_format           0b001 = E2M1
[10,12)  b_format           0b01 = E2M1 (only 2 bits, not 3!)
[12]     reserved           0
[13]     a_negate            0
[14]     b_negate            0
[15]     a_major             0 (transpose NOT supported)
[16]     b_major             0 (transpose NOT supported)
[17,23)  n_dim              N >> 3
[23]     scale_format        0 = UE4M3, 1 = UE8M0
[24,27)  reserved           0
[27,29)  m_dim              M >> 7 (NOT M >> 4 like regular!)
[29,31)  a_sf_id            SFA data ID (0 for scale_vec::4X)
[31]     k_size             0 = K64 (dense), 1 = K96 (sm_103a only)
```

### Building the descriptor in C++

```cpp
uint32_t build_nvfp4_descriptor(int M, int N) {
    uint32_t desc = 0;
    desc |= 0b001 << 7;     // a_format = E2M1
    desc |= 0b01  << 10;    // b_format = E2M1
    desc |= (N >> 3) << 17; // n_dim
    desc |= 0 << 23;        // scale_format = UE4M3
    desc |= (M >> 7) << 27; // m_dim (M=128 -> 1, M=256 -> 2)
    // a_sf_id = 0, b_sf_id = 0 for scale_vec::4X
    return desc;
}
```

### vs. Regular descriptor (BF16/FP8)

Key differences:
- M encoding: `M >> 7` in 2 bits (blockscaled) vs `M >> 4` in 5 bits (regular)
- Bits [4,5]: `b_sf_id` (blockscaled) vs part of c_format (regular)
- Bit [23]: `scale_format` (blockscaled) vs reserved (regular)
- Bits [29,30]: `a_sf_id` (blockscaled) vs reserved (regular)
- `a_format`/`b_format` encoding: E2M1 = 0b001 (blockscaled) vs different codes (regular)

## Tile Size Constraints

### cta_group::1 (single CTA)

| Dimension | Allowed values |
|-----------|---------------|
| M | 128 only |
| N | 8, 16, 24, ..., 256 (multiples of 8) |
| K | 64 (dense), 128 (sparse) |

### cta_group::2 (2-CTA cluster)

| Dimension | Allowed values |
|-----------|---------------|
| M | 128 or 256 |
| N | 16, 32, 48, ..., 256 (multiples of 16) |
| K | 64 (dense), 128 (sparse), 96 (sm_103a only) |

### Typical tile configurations

| Source | CTA tile | K per MMA | Cluster | Notes |
|--------|----------|-----------|---------|-------|
| ThunderKittens | 256x256x256 | K=64, 4 MMAs | 2-CTA | Best for 4K+ square |
| CUTLASS 72a | 256x256x256 | K=64 | 2x4 cluster | Large cluster, big problems |
| CUTLASS 72b | 128x128x256 | K=64 | 1x1 | Simple, no cluster |
| FlashInfer | 128x128x256 or 128x256x256 | K=64 | 1x1 or 2x1 | Multiple configs |

With K=64 per MMA instruction and tile K=256, each tile requires 4 MMA instructions along K.

## TMEM Layout

TMEM is 128 lanes x 512 columns, 32 bits per cell (256 KB per SM).

### Accumulator region

For M=128, N=256: uses 128 lanes x 256 columns (128 KB).

### Scale factor regions

Scale factors MUST be in TMEM. They occupy separate columns from the accumulator.

**For NVFP4 (scale_vec::4X, UE4M3, block-16):**
- Scale A: 128 lanes x 4 columns (4 scale factors per row for K=64)
- Scale B: stored transposed, 4 columns
- Each scale is 8 bits (UE4M3), but occupies a full 32-bit TMEM cell
- 4 scales are packed into 4 bytes of one 32-bit word per lane

**For K=256 tile (4 MMAs):** Each MMA uses K=64, so you need 4 scale values per row. With block-16 scaling, that's 64/16 = 4 scale factors per K-block, fitting in one TMEM column of 4 bytes.

### Loading scales from SMEM to TMEM

Use `tcgen05.cp` to copy scale factor tiles from shared memory to TMEM:

```cpp
// Copy 32x128b (32 rows x 16 bytes = 32x16 FP8 scales) from SMEM to TMEM
asm volatile(
    "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
    :: "r"(tmem_scale_addr), "l"(smem_desc));
```

For cta_group::2, use `tcgen05.cp.cta_group::2.32x128b.warpx4` to load into both CTAs' TMEM.

### TMEM allocation

```cpp
// Allocate TMEM columns for D + scales
// For M=128, N=256, NVFP4: need 256 (D) + 16 (SFA) + 16 (SFB) = 288 columns
using Allocator = cute::TMEM::Allocator1Sm;  // or Allocator2Sm for cta_group::2
Allocator().allocate(288, tmem_ptr_in_smem);
```

For cta_group::2, use `Allocator2Sm` which coordinates allocation across both CTAs in the cluster.

## Scale Factor Memory Layout in Global/Shared Memory

### The swizzled layout

Scale factors are NOT stored in simple row-major order. They use an interleaved layout optimized for TMEM loading.

For NVFP4 (16-element blocks, UE4M3 scales), the layout atom covers 128 rows x 4 K-blocks:

```
// For a 128-row group at K-block positions [k0, k1, k2, k3]:
// Scale index = row_in_32 * 16 + tile_in_block * 4 + kb_in_block
//
// Where:
//   row_in_32 = row % 32           (0-31)
//   tile_in_block = (row / 32) % 4 (0-3, selecting which 32-row chunk)
//   kb_in_block = k_block % 4      (0-3, which K-block in the group)
//
// Total: 32 * 16 = 512 bytes per 128-row x 4-K-block atom
```

In CUTLASS, this is encoded as:
```cpp
using SfAtom = Layout<
    Shape<Shape<_32, _4>, Shape<_16, _4>>,
    Stride<Stride<_16, _4>, Stride<_0, _1>>
>;
```

### Quantization kernel scale output

When quantizing FP32/BF16 data to NVFP4, the quantization kernel must write scales in this swizzled layout:

```cpp
// For row `r` and K-block `kb` (each block covers 16 K elements):
int M_block = r / 128;
int K_block_group = kb / 4;
int row_in_32 = r % 32;
int tile_in_block = (r / 32) % 4;
int kb_in_block = kb % 4;

int scale_offset = (M_block * K_block_groups + K_block_group) * 512
                 + row_in_32 * 16 + tile_in_block * 4 + kb_in_block;

scale_buffer[scale_offset] = computed_scale;
```

## TMA Setup for FP4 Data + Scales

FP4 data and scale factors require separate TMA descriptors.

### FP4 data TMA

Two FP4 elements are packed per byte. TMA sees the packed type:

```cpp
// A is (M, K) but each byte holds 2 elements, so TMA shape is (M, K/2)
// Element type: fp4e2m1_2 (1 byte = 2 FP4 values)
// Box size: (BLOCK_M, BLOCK_K/2) in bytes
// Alignment: 32 elements = 16 bytes

CUtensorMap tma_a;
uint64_t size[2] = {K/2, M};           // inner=K/2 bytes, outer=M rows
uint64_t stride[1] = {K/2};            // row stride in bytes
uint32_t box[2] = {BLOCK_K/2, BLOCK_M};
cuTensorMapEncodeTiled(&tma_a, CU_TENSOR_MAP_DATA_TYPE_UINT8,
    2, data_ptr, size, stride, box, elem_stride,
    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, ...);
```

### Scale factor TMA

```cpp
// Scales have the swizzled layout described above
// Load as half (FP16) type even though actual data is FP8 —
// this matches the 2-byte granularity of the swizzled layout
CUtensorMap tma_sfa;
// Shape depends on the interleaved atom structure
// ThunderKittens uses: gl<half, 1, -1, -1, 256, st_hf<4,256>>
```

### Coordinating data + scale loads

Both data and scale loads must complete before the MMA instruction. Use the same mbarrier or separate barriers:

```cpp
// Option 1: Same barrier for data + scales (simpler)
tma_load_A(barrier, smem_A, coords...);
tma_load_SFA(barrier, smem_SFA, coords...);
tma_load_B(barrier, smem_B, coords...);
tma_load_SFB(barrier, smem_SFB, coords...);
barrier.arrive_and_expect_tx(bytes_A + bytes_SFA + bytes_B + bytes_SFB);

// Wait for all loads
barrier.wait(phase);

// Copy scales from SMEM to TMEM
tcgen05_cp_scales(tmem_sfa, smem_sfa_desc);
tcgen05_cp_scales(tmem_sfb, smem_sfb_desc);

// Issue MMA
tcgen05_mma_nvfp4(tmem_d, a_desc, b_desc, idesc, tmem_sfa, tmem_sfb, accumulate);
```

## 2-CTA Cluster Pattern

For cta_group::2, two CTAs cooperate. The leader CTA issues MMA; the peer provides B data via DSMEM.

### Roles

```
CTA 0 (leader):
  - Loads A tile to local SMEM
  - Loads half of B to local SMEM
  - Loads A scales to local SMEM, copies to TMEM
  - Loads B scales (multicast to both CTAs)
  - Issues tcgen05.mma.cta_group::2

CTA 1 (peer):
  - Loads A tile to local SMEM (different M rows)
  - Loads other half of B to local SMEM
  - Loads A scales to local SMEM, copies to TMEM
  - Does NOT call MMA — hardware reads B from peer's SMEM via DSMEM

Both CTAs:
  - Run epilogue: load from TMEM -> apply global scale -> store output
  - Coordinate TMEM allocation/deallocation via semaphores
```

### TMA load coordinates

Each CTA offsets its tile coordinates by `cta_rank` to load different data:

```cpp
int cta_rank = get_ctarank();  // 0 or 1

// A: each CTA loads its own 128 M-rows
tma_load(sA, tma_a, mbar, k_offset, m_tile*2 + cta_rank);

// B: each CTA loads HALF the N dimension (N/2 columns each)
tma_load(sB, tma_b, mbar, k_offset, n_tile*2 + cta_rank);

// IMPORTANT: The TMA tensor map for B must be created with per-CTA
// tile dimensions (N/2 per CTA), NOT the full N.
```

### Scale multicast

For B scales, use TMA multicast to deliver to both CTAs:
```cpp
// Multicast mask 0b11 = both CTA 0 and CTA 1
tma::cluster::load_async(smem_sfb, global_sfb, coords, barrier, 0b11, 0);
```

### TMEM allocation for cta_group::2

```cpp
// Use Allocator2Sm — coordinates across both CTAs
using Allocator = cute::TMEM::Allocator2Sm;

// For M=256 with cta_group::2:
// D uses all 128 lanes x N columns per CTA (hardware interleaves across 2 CTAs)
// Scales need separate TMEM columns
Allocator().allocate(total_cols, tmem_ptr_in_smem);
```

### Commit (umma_arrive) for cta_group::2

```cpp
// cta_group::2 commit with multicast — signals BOTH CTAs' mbarriers
// The .shared::cluster.multicast::cluster qualifier is required so the
// peer CTA's mbarrier also gets signaled.
// dst_cta_mask is uint16_t: 0x3 = both CTA 0 and CTA 1
asm volatile(
    "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
    ".shared::cluster.multicast::cluster.b64 [%0], %1;"
    :: "r"(smem_barrier_addr), "h"((uint16_t)0x3));
```

## SMEM Descriptor for Blockscaled MMA

The SMEM descriptor format is the same as regular tcgen05.mma (version 1, 64-bit), but the swizzle and layout must match the FP4 packed format:

```cpp
// For FP4 data with 128B swizzle:
// Two FP4 elements per byte, so address calculations use half the K dimension
cute::UMMA::SmemDescriptor desc;
desc.version_ = 1;
desc.layout_type_ = SWIZZLE_128B;  // 128-byte swizzle for K-major
desc.start_address_ = smem_addr >> 4;
desc.stride_byte_offset_ = ...;    // Stride between non-contiguous rows
desc.leading_byte_offset_ = 0;
```

## Complete Kernel Skeleton (cta_group::1, NVFP4)

```cpp
// Simplified — shows the structure, not a complete kernel

__global__ void __launch_bounds__(256, 1)
nvfp4_gemm_kernel(
    const __nv_fp4x2_e2m1* A,    // (M, K/2) packed
    const __nv_fp8_e4m3* A_scale, // swizzled scale layout
    float A_global_scale,
    const __nv_fp4x2_e2m1* B,    // (N, K/2) packed
    const __nv_fp8_e4m3* B_scale,
    float B_global_scale,
    __nv_bfloat16* D,             // (M, N) output
    int M, int N, int K)
{
    // 1. Allocate TMEM: D (128 x N_TILE) + SFA (128 x 4) + SFB (128 x 4)
    Allocator1Sm().allocate(N_TILE + 8, tmem_ptr);

    // 2. Initialize barriers for TMA pipeline

    // 3. Main loop over K tiles (K_TILE = 256, 4 MMAs per tile)
    for (int k = 0; k < K; k += K_TILE) {
        // TMA load A, B, SFA, SFB to SMEM
        // Wait for loads
        // Copy SFA, SFB from SMEM to TMEM via tcgen05.cp

        // 4 MMA instructions (K=64 each)
        for (int ki = 0; ki < 4; ki++) {
            uint32_t idesc = build_nvfp4_descriptor(128, N_TILE);
            uint32_t accumulate = (k > 0 || ki > 0) ? 1 : 0;
            tcgen05_mma_nvfp4(tmem_d, a_desc[ki], b_desc[ki],
                              idesc, tmem_sfa, tmem_sfb, accumulate);
        }
        umma_arrive(empty_barrier);  // Signal SMEM can be reused
    }

    // 4. Epilogue: load D from TMEM, multiply by global scale, store to global
    float global_scale = A_global_scale * B_global_scale;
    for (int col = 0; col < N_TILE; col += 8) {
        // tcgen05.ld.sync.aligned.32x32b.x8 to load 8 FP32 values from TMEM
        // Convert to BF16, apply global_scale
        // Store to global memory
    }

    // 5. Deallocate TMEM
    Allocator1Sm().free(0, N_TILE + 8);
}
```

## Compilation

```bash
nvcc -gencode arch=compute_100a,code=sm_100a \
     -O3 --expt-relaxed-constexpr \
     -I path/to/cutlass/include \  # For TMEM allocator, barrier types, TMA descriptors
     kernel.cu -o kernel -lcuda
```

The `--expt-relaxed-constexpr` flag is required for CUTLASS headers. You only need CUTLASS headers for TMEM allocation (`cute/arch/tmem_allocator_sm100.hpp`), barrier types (`cutlass/arch/barrier.h`), and TMA descriptor helpers — not for the MMA instruction itself.

## Reference Implementations

| Source | File | Approach |
|--------|------|----------|
| ThunderKittens | `resources/thunderkittens-main/kernels/gemm/nvfp4_b200/nvfp4_b200_gemm.cu` | Full custom kernel with kittens.cuh, 2-CTA cluster |
| CUTLASS 72a | `resources/cutlass-4.4.2/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu` | CUTLASS builder, BF16 output |
| CUTLASS 72b | Same directory, `72b_blackwell_nvfp4_nvfp4_gemm.cu` | CUTLASS builder, FP4 output with output scales |
| FlashInfer | `resources/flashinfer-0.6.7/include/flashinfer/gemm/fp4_gemm_template_sm100.h` | CUTLASS-based template |
| CUTLASS blockscaled MMA | `resources/cutlass-4.4.2/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp` | Internal MMA implementation |
