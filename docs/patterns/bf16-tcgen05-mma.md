# BF16 tcgen05.mma for SM100 (Blackwell)

This doc covers the `tcgen05.mma.kind::f16` instruction for BF16 GEMM on Blackwell (SM100). Use this when writing BF16 attention or GEMM kernels from scratch in CUDA C++ with inline PTX.

## Overview

`tcgen05.mma.kind::f16` multiplies BF16/FP16 matrices using the 5th-generation tensor cores on SM100. Accumulator D lives in TMEM (Tensor Memory), operands A and B come from SMEM (with swizzled layout).

**Operation:** `D[tmem] = A[smem] * B[smem] + D[tmem]` (FP32 accumulator)

## PTX Instruction Syntax

### No `.ws` (both operands from SMEM)

```
tcgen05.mma.cta_group::1.kind::f16 [d-tmem], a-desc, b-desc, idesc, scale-d;
tcgen05.mma.cta_group::1.kind::f16 [d-tmem], a-desc, b-desc, idesc, scale-d, {disable-output-lane};
```

### `.ws` mode (A from TMEM, B from SMEM)

```
tcgen05.mma.cta_group::1.kind::f16.ws [d-tmem], a-tmem, b-desc, idesc, scale-d;
```

In `.ws` mode, operand A is already in TMEM (loaded once via `tcgen05.cp`). Useful for attention: load Q once to TMEM, stream KV tiles from SMEM.

### Inline PTX wrapper (cta_group::1, no .ws)

```cpp
__device__ __forceinline__ void tcgen05_mma_bf16(
    uint32_t d_tmem,   // TMEM address for accumulator D
    uint64_t a_desc,   // SMEM descriptor for A matrix
    uint64_t b_desc,   // SMEM descriptor for B matrix
    uint32_t idesc,    // Instruction descriptor
    uint32_t scale_d)  // 1 = accumulate (D += A*B), 0 = overwrite (D = A*B)
{
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(scale_d));
}
```

### Inline PTX wrapper (.ws mode — A from TMEM)

```cpp
__device__ __forceinline__ void tcgen05_mma_bf16_ws(
    uint32_t d_tmem,   // TMEM address for accumulator D
    uint32_t a_tmem,   // TMEM address for A matrix (pre-loaded via tcgen05.cp)
    uint64_t b_desc,   // SMEM descriptor for B matrix
    uint32_t idesc,    // Instruction descriptor
    uint32_t scale_d)  // 1 = accumulate, 0 = overwrite
{
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16.ws [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(d_tmem), "r"(a_tmem), "l"(b_desc), "r"(idesc), "r"(scale_d));
}
```

## Instruction Descriptor Encoding

The instruction descriptor is 32 bits with this layout for `kind::f16`:

```
Bit(s)   Field              Values for BF16 GEMM
-------  -----------------  -----------------------------------------
[0,2)    sparsity_sel       0 (dense)
[2]      sparse_flag        0 (dense)
[3]      saturate           0
[4,6)    d_type             1 = FP32 (accumulator)
[6]      reserved           0
[7,10)   a_type             1 = BF16  (0 = FP16)
[10,13)  b_type             1 = BF16  (0 = FP16)
[13]     negate_a           0
[14]     negate_b           0
[15]     transpose_a        0 or 1
[16]     transpose_b        0 or 1
[17,23)  n_dim              N >> 3
[23]     reserved           0
[24,29)  m_dim              M >> 4
[29]     reserved           0
[30,32)  ws_b_shift         0 (no shift) — only for .ws mode
```

### Building the descriptor in C++

```cpp
uint32_t build_bf16_mma_descriptor(int M, int N, bool transpose_a = false, bool transpose_b = false) {
    uint32_t desc = 0;
    desc |= 1u << 4;              // d_type = FP32
    desc |= 1u << 7;              // a_type = BF16
    desc |= 1u << 10;             // b_type = BF16
    desc |= (transpose_a ? 1u : 0u) << 15;
    desc |= (transpose_b ? 1u : 0u) << 16;
    desc |= ((uint32_t)(N >> 3)) << 17;  // n_dim
    desc |= ((uint32_t)(M >> 4)) << 24;  // m_dim
    return desc;
}

// Example: M=64, N=64, no transpose
// desc = (1<<4) | (1<<7) | (1<<10) | (8<<17) | (4<<24) = 0x04100490
```

## Tile Size Constraints

### cta_group::1

| Mode | M | N | K |
|------|---|---|---|
| No .ws | 64, 128 | 8, 16, 24, ..., 256 (multiples of 8) | 16 |
| .ws | 32, 64, 128 | 64, 128, 256 | 16 |

### cta_group::2

| Mode | M | N | K |
|------|---|---|---|
| No .ws | 128, 256 | 16, 32, ..., 256 (multiples of 16) | 16 |
| .ws | NOT supported | | |

### For MLA decode attention (Q@KV^T → softmax → P@V)

- **QK GEMM**: M=64 (heads), N=CTA_KV (e.g. 64), K=16, iterate K 576/16=36 times
  - Use `.ws` mode: Q in TMEM (loaded once), KV in SMEM (streamed)
- **PV GEMM**: M=64 (heads), N=256 (half of DV=512), K=16, iterate K CTA_KV/16 times
  - Both P and V from SMEM (no `.ws`)
  - Two passes for full DV=512

## SMEM Layout (128B Swizzling)

tcgen05 requires swizzled SMEM layouts. Use 128-byte swizzling for BF16:

### Swizzle pattern for BF16

The swizzle XORs address bits to eliminate bank conflicts:

```cpp
// For 128B swizzling with BF16 (2 bytes per element):
// 64 elements per 128-byte row
// Swizzle: addr[4:6] ^= addr[7:9]  (for 128B mode)

__device__ __forceinline__ uint32_t swizzle_128b(uint32_t smem_addr) {
    // XOR bits [4:6] with bits [7:9] of the shared memory address
    return smem_addr ^ ((smem_addr >> 3) & 0x70);
}
```

### K-major layout for BF16 (used by MMA B operand)

For a tile of [rows, K] BF16 values in K-major (K is the fast-changing dimension):

```cpp
// Store BF16 value at (row, k) into swizzled SMEM:
// Base layout: row-major with rows of K BF16 elements (K*2 bytes per row)
// Swizzle is applied by hardware via the SMEM descriptor swizzle mode

// Simple approach: just store contiguously and let descriptor handle swizzle
// The descriptor's swizzle mode field tells the MMA unit how to deswizzle
__device__ void store_bf16_smem_swizzled(
    __nv_bfloat16* smem_base,
    int row, int k, int stride_k,
    __nv_bfloat16 val)
{
    // Physical address with 128B XOR swizzle
    int byte_offset = (row * stride_k + k) * 2;
    int swizzled = byte_offset ^ ((byte_offset >> 3) & 0x70);
    *reinterpret_cast<__nv_bfloat16*>((char*)smem_base + swizzled) = val;
}
```

## SMEM Descriptor (64-bit)

The SMEM descriptor tells the MMA unit where the matrix is in shared memory and how it's laid out:

```
Bit(s)   Field                          Encoding
-------  ---------------------------    ---------------------------
[0,14)   Start address                  (smem_addr & 0x3FFFF) >> 4
[14,16)  Reserved                       0
[16,30)  Leading dimension byte offset  (stride_bytes & 0x3FFFF) >> 4
[30,32)  Reserved                       0
[32,46)  Stride dimension byte offset   (stride_dim_bytes & 0x3FFFF) >> 4
[46,49)  Fixed                          0b001
[49,52)  Base offset                    0 (for standard swizzle start)
[52]     Leading dim mode               0 = relative, 1 = absolute
[53,61)  Fixed                          0
[61,64)  Swizzle mode                   2 = 128-byte swizzle
```

### Building the descriptor in C++

```cpp
__device__ uint64_t make_smem_desc(
    uint32_t smem_addr,        // shared memory address (from __cvta_generic_to_shared)
    uint32_t leading_dim_bytes,// stride between rows in bytes
    uint32_t stride_dim_bytes) // stride in the K dimension in bytes (usually 0 or leading_dim * num_rows)
{
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_addr & 0x3FFFF) >> 4);                  // start addr
    desc |= (uint64_t)((leading_dim_bytes & 0x3FFFF) >> 4) << 16;    // leading dim
    desc |= (uint64_t)((stride_dim_bytes & 0x3FFFF) >> 4) << 32;     // stride dim
    desc |= (uint64_t)0x1 << 46;                                     // fixed 0b001
    desc |= (uint64_t)0x0 << 49;                                     // base offset
    desc |= (uint64_t)0x0 << 52;                                     // relative mode
    desc |= (uint64_t)0x2 << 61;                                     // 128B swizzle
    return desc;
}

// Example for BF16 tile [64, 576] in SMEM, K-major:
// leading_dim = 576 * 2 = 1152 bytes (stride between rows)
// stride_dim = 0 (for simple K-major)
// uint64_t desc = make_smem_desc(smem_ptr, 1152, 0);
```

## TMEM Allocation and Usage

TMEM is 128 lanes × 512 columns, 32 bits per cell (256 KB per SM). Only one thread per warp (`elect_one_sync()`) issues TMEM operations.

### Allocating TMEM

```cpp
// Allocate N columns of TMEM. Returns base address in smem_ptr.
__device__ __forceinline__ void tmem_alloc(uint32_t* smem_ptr, uint32_t num_cols) {
    if (threadIdx.x % 128 == 0) {  // elect one per warpgroup
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(__cvta_generic_to_shared(smem_ptr)), "r"(num_cols));
    }
    __syncwarp();
}
```

### Deallocating TMEM

```cpp
__device__ __forceinline__ void tmem_dealloc(uint32_t tmem_addr, uint32_t num_cols) {
    if (threadIdx.x % 128 == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(tmem_addr), "r"(num_cols));
    }
    __syncwarp();
}
```

### Loading data from SMEM to TMEM (tcgen05.cp)

Used to load Q into TMEM for `.ws` mode:

```cpp
// Copy 32 rows × 128 bits from SMEM to TMEM
__device__ __forceinline__ void tmem_copy_32x128b(uint32_t tmem_addr, uint64_t smem_desc) {
    if (threadIdx.x % 128 == 0) {
        asm volatile(
            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
            :: "r"(tmem_addr), "l"(smem_desc));
    }
}

// Copy 64 rows × 128 bits from SMEM to TMEM
__device__ __forceinline__ void tmem_copy_64x128b(uint32_t tmem_addr, uint64_t smem_desc) {
    if (threadIdx.x % 128 == 0) {
        asm volatile(
            "tcgen05.cp.cta_group::1.warpx4 [%0], %1;"
            :: "r"(tmem_addr), "l"(smem_desc));
    }
}
```

### Reading accumulator from TMEM to registers

After MMA completes, read results from TMEM for softmax or output:

```cpp
// Read 32 consecutive 32-bit values from TMEM at (lane, col..col+31)
__device__ __forceinline__ void tmem_load_32(float* dst, uint32_t tmem_col) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0, %1, %2, %3, %4, %5, %6, %7,"
        " %8, %9, %10, %11, %12, %13, %14, %15,"
        " %16, %17, %18, %19, %20, %21, %22, %23,"
        " %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15]),
          "=f"(dst[16]), "=f"(dst[17]), "=f"(dst[18]), "=f"(dst[19]),
          "=f"(dst[20]), "=f"(dst[21]), "=f"(dst[22]), "=f"(dst[23]),
          "=f"(dst[24]), "=f"(dst[25]), "=f"(dst[26]), "=f"(dst[27]),
          "=f"(dst[28]), "=f"(dst[29]), "=f"(dst[30]), "=f"(dst[31])
        : "r"(tmem_col));
}
```

### Storing to TMEM from registers

```cpp
// Store 32 consecutive 32-bit values to TMEM
__device__ __forceinline__ void tmem_store_32(uint32_t tmem_col, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x1.b32 [%0],"
        " {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16,"
        "  %17, %18, %19, %20, %21, %22, %23, %24,"
        "  %25, %26, %27, %28, %29, %30, %31, %32};"
        :: "r"(tmem_col),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]),
           "f"(src[16]), "f"(src[17]), "f"(src[18]), "f"(src[19]),
           "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]),
           "f"(src[24]), "f"(src[25]), "f"(src[26]), "f"(src[27]),
           "f"(src[28]), "f"(src[29]), "f"(src[30]), "f"(src[31]));
}
```

## Fences

tcgen05 requires explicit fences around TMEM operations:

```cpp
// Before __syncthreads() that separates TMEM write from TMEM read
__device__ __forceinline__ void tcgen05_fence_before_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

// After __syncthreads()
__device__ __forceinline__ void tcgen05_fence_after_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}
```

### Fence usage pattern

```cpp
// After MMA writes to TMEM, before reading D:
tcgen05_fence_before_sync();
__syncthreads();
tcgen05_fence_after_sync();
// Now safe to read D from TMEM via tcgen05.ld
```

## TMEM Layout for MLA Decode Attention

For MLA with 64 heads per CTA, DK=576, DV=512:

```
TMEM columns (512 total):
  [0,   255]  = O accumulator (64 heads × 512/128 × 32 cols = 256 cols for FP32 output)
  [256, 399]  = Q matrix (64 heads × 576 dim, loaded once via tcgen05.cp)
  [400, 463]  = P scores (64 heads × 64 KV tokens = 64 cols for QK^T output)
  [464, 511]  = spare
```

This layout comes from FlashMLA SM100. Q stays in TMEM for the entire tile, P is written by QK GEMM and read for softmax, O accumulates PV results.

## Complete Minimal Example: BF16 tcgen05 GEMM tile

```cpp
// Minimal: compute C[M,N] += A[M,K] * B[K,N] for one tile
// A, B in SMEM (128B swizzled), C in TMEM
// M=64, N=64, K=16 (one MMA instruction)

__device__ void gemm_tile_tcgen05(
    uint32_t c_tmem,             // TMEM address for output
    __nv_bfloat16* a_smem,       // A in SMEM [M, K] swizzled
    __nv_bfloat16* b_smem,       // B in SMEM [K, N] or [N, K] swizzled
    int a_stride_bytes,          // leading dim of A in bytes
    int b_stride_bytes,          // leading dim of B in bytes
    bool accumulate)             // true = C += A*B, false = C = A*B
{
    if (threadIdx.x % 128 == 0) {  // elect one thread per warpgroup
        uint32_t a_smem_addr = __cvta_generic_to_shared(a_smem);
        uint32_t b_smem_addr = __cvta_generic_to_shared(b_smem);

        uint64_t a_desc = make_smem_desc(a_smem_addr, a_stride_bytes, 0);
        uint64_t b_desc = make_smem_desc(b_smem_addr, b_stride_bytes, 0);

        uint32_t idesc = build_bf16_mma_descriptor(64, 64);  // M=64, N=64

        tcgen05_mma_bf16(c_tmem, a_desc, b_desc, idesc, accumulate ? 1 : 0);
    }
}

// For K-loop (K=576, 36 iterations of K=16 each):
for (int ki = 0; ki < 36; ki++) {
    __nv_bfloat16* a_ptr = a_smem + ki * 16;  // advance K by 16
    __nv_bfloat16* b_ptr = b_smem + ki * 16;  // advance K by 16
    gemm_tile_tcgen05(c_tmem, a_ptr, b_ptr,
                      576 * 2, 576 * 2,        // strides in bytes
                      ki > 0);                  // accumulate after first
}
```

## Key Differences from mma.sync (SM80)

| Aspect | mma.sync (SM80) | tcgen05 (SM100) |
|--------|-----------------|-----------------|
| Operand source | Registers | SMEM + TMEM |
| Accumulator | Registers | TMEM (256 KB) |
| Issuing threads | All warps | 1 elected thread per warpgroup |
| Tile size | 16×8×16 | Up to 128×256×16 |
| Register pressure | High (accum in regs) | Low (accum in TMEM) |
| Throughput | ~300 TFLOPS (B200) | ~2250 TFLOPS (B200) |
| Bank conflicts | Manual padding | Hardware swizzle |

**The throughput difference is 7.5×.** No amount of mma.sync optimization can close this gap. On SM100, always use tcgen05.
