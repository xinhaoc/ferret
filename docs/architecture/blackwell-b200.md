# Blackwell B200/GB200 Architecture Deep-Dive

## GPU Variants

| Variant | CC | SMs | Target | Notes |
|---|---|---|---|---|
| B200 | 10.0 (`sm_100`) | 148 (74/die) | Data center | Dual-die, 208B transistors |
| GB200 | 10.0 (`sm_100`) | 2× 148 | Data center | Grace CPU + 2× B200 superchip |
| B300 / GB300 | 10.3 (`sm_103`) | 160 | Data center | Blackwell Ultra, 288 GB HBM3e |
| RTX 5090 | 12.0 (`sm_120`) | 170 | Consumer | GB202 die, different SM design |
| RTX 5080 | 12.0 (`sm_120`) | 84 | Consumer | GB203 die |

**Critical**: CC 10.0 and CC 12.0 are **fundamentally different SM designs** despite both being "Blackwell":
- SM100 has TMEM, tcgen05, 64 FP64 cores, 256 KB shared+L1, 64 warps/SM
- SM120 has **NO TMEM, NO tcgen05**, 2 FP64 cores, 100 KB shared+L1, 48 warps/SM, uses `mma.sync` (Ampere-style)
- **No kernel compatibility between sm_100 and sm_120**

### Compilation Targets

```
compute_100    — baseline, forward-compatible
compute_100f   — family-specific (compatible with sm_100 + sm_103)
compute_100a   — architecture-specific (sm_100 only, NOT forward-compatible)
sm_100         — B200/GB200 SASS
sm_103         — B300/GB300 SASS
sm_120         — RTX 5090/5080 SASS
```

Use `compute_100a` / `sm_100a` for architecture-specific features like tcgen05 instructions.

## SM Architecture (CC 10.0 — B200)

- **128 FP32 cores** per SM (same as Hopper)
- **64 FP64 cores** per SM (improved double-precision)
- **64 INT32 cores** per SM
- **4 fifth-generation Tensor Cores** per SM
- **16 special function units**
- **4 warp schedulers** per SM
- Max 64 warps/SM, 2048 threads/SM, 32 blocks/SM
- Register file: 256 KB (64K × 32-bit registers), max 255 regs/thread
- **256 KB Tensor Memory (TMEM)** per SM — new dedicated on-chip memory for TC accumulators, separate from shared memory

## SM Architecture (CC 12.0 — RTX 5090)

- **128 FP32 cores** per SM
- **2 FP64 cores** per SM (minimal double-precision)
- **64 INT32 cores** per SM
- **5th-gen Tensor Core(s)** per SM (FP8, FP16, BF16, TF32, INT8 with sparsity)
- **16 special function units**
- **4 warp schedulers**
- Max 48 warps/SM (24 blocks/SM), 1536 threads/SM
- **NO Tensor Memory (TMEM)** — uses `mma.sync` (warp-synchronous, like Ampere) instead of tcgen05
- **NO tcgen05 instructions** — completely different tensor core programming model from SM100

## Memory Hierarchy

### B200 (CC 10.0)
- **Shared memory + L1**: 256 KB combined per SM (same as Hopper)
- **Configurable shared memory**: up to 228 KB. Carveout options: 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 KB
- **L2 cache**: 96 MB (nearly 2x Hopper's 50 MB)
- **HBM3e**: 192 GB, ~8 TB/s bandwidth (2.4x Hopper's 3.35 TB/s)
- 32 shared memory banks

### RTX 5090 (CC 12.0)
- **Shared memory + L1**: 100 KB combined per SM
- **Configurable shared memory**: up to 99 KB. Carveout options: 0, 8, 16, 32, 64, 100 KB
- **L2 cache**: 96 MB
- **GDDR7**: 32 GB, ~1792 GB/s

## Tensor Cores (5th Generation)

### Supported Data Types
FP64, TF32, FP16, BF16, FP8 (E4M3 and E5M2), INT8 — all with structured sparsity support.

**NEW in Blackwell**:
- **FP4 (NVFP4, E2M1)** — 4-bit floating point for inference, doubles throughput vs FP8
- **FP6** — 6-bit floating point, intermediate precision between FP4 and FP8
- **Micro-tensor scaling (MX formats)** — fine-grained block scaling for FP4/FP6/FP8

### Peak Performance (B200 HGX, dense | with 2:4 sparsity)

Source: NVIDIA Blackwell Datasheet 3384703, DEC24. Datasheet reports with-sparsity numbers; dense = datasheet ÷ 2.

| Data Type | Dense | Sparse (datasheet) |
|---|---|---|
| FP4 | 9 PFLOPS | 18 PFLOPS |
| FP6 | 4.5 PFLOPS | 9 PFLOPS |
| FP8 | 4.5 PFLOPS | 9 PFLOPS |
| FP16 | 2.25 PFLOPS | 4.5 PFLOPS |
| BF16 | 2.25 PFLOPS | 4.5 PFLOPS |
| TF32 | 1.1 PFLOPS | 2.2 PFLOPS |
| FP32 (CUDA cores) | 75 TFLOPS | — |
| FP64 TC | 37 TFLOPS | — |
| INT8 | 4.5 POPS | 9 POPS |

Note: GB200 NVL72 variant has ~10-15% higher numbers (higher clocks, 1200W TDP).

**vs Hopper**: ~2.3x dense FP16, ~2.3x dense FP8, new FP4/FP6 support.

### Tensor Memory (TMEM) — New in Blackwell (SM100 only)

5th-gen TensorCores on SM100 have **dedicated 256 KB on-chip Tensor Memory per SM**, completely separate from shared memory and L1:
- 2D structure: 512 columns × 128 rows (lanes) per CTA
- Each cell is 32 bits → 256 KB total
- Dynamically allocated by a single warp in a CTA
- Accessed via `tcgen05` PTX instructions
- Addresses are 32-bit: upper 16 bits = lane index, lower 16 bits = column index
- MMA results go directly to TMEM (not registers), freeing the register file
- Data flow: Global → (TMA) → SMEM → (tcgen05.mma) → TMEM → (tcgen05.ld) → Registers

**SM120 (RTX 5090) does NOT have TMEM.** It uses traditional register-based MMA accumulators via `mma.sync`.

This is a major architectural change — tensor operations read/write from dedicated memory, eliminating register pressure from large accumulator tiles.

## New Features vs Hopper

### tcgen05 Instruction Family (5th-Gen TC, SM100 only)

New PTX instruction set replacing Hopper's WGMMA. Introduced in PTX ISA 8.7.

**Key architectural difference from Hopper**: Single-thread semantics (one thread initiates MMA) vs Hopper's warpgroup model (4 warps). Results go to TMEM, not registers.

| Instruction | Purpose |
|---|---|
| `tcgen05.alloc` | Allocate TMEM columns (power-of-2, min 32) |
| `tcgen05.dealloc` | Deallocate TMEM |
| `tcgen05.relinquish_alloc_permit` | Release allocation permission |
| `tcgen05.mma` | Matrix multiply-accumulate (single SM) |
| `tcgen05.mma.sp` | MMA with structured sparsity |
| `tcgen05.mma.ws` | MMA with warp specialization |
| `tcgen05.mma.cta_group::2` | Cooperative 2-SM MMA (doubles M dimension to 256) |
| `tcgen05.ld` | Load from TMEM to registers |
| `tcgen05.st` | Store from registers to TMEM |
| `tcgen05.cp` | Copy from shared memory to TMEM |
| `tcgen05.shift` | Shift data within TMEM |
| `tcgen05.commit` | Track async completion on mbarrier |
| `tcgen05.wait` | Wait for TMEM operations |
| `tcgen05.fence` | Memory fence for TMEM |

MMA dimensions: single SM up to 128×256×K, cooperative 2-SM up to 256×256×K.

Available only with `sm_100a` or `sm_100f` compilation targets. **NOT available on SM120 (RTX 5090).**

### FP4 Support
- New E2M1 format (2 exponent bits, 1 mantissa bit, 1 sign bit)
- Doubles throughput vs FP8 for inference workloads
- Quantization-aware training needed for good accuracy

### Enhanced Cluster Support
- Thread block clusters carry over from Hopper
- Improved distributed shared memory performance
- Better multi-SM coordination

### NVLink 5th Generation
- 1.8 TB/s total GPU-to-GPU bandwidth (2x Hopper's 900 GB/s)
- Up to 576 GPUs connected via NVLink Switch

### NVLink Chip-2-Chip (C2C)
- GB200: connects Grace CPU to Blackwell GPU at 900 GB/s
- Enables unified memory between CPU and GPU

### Second-Generation Transformer Engine
- FP4 support for inference
- Improved mixed-precision management

## Comparison: Blackwell vs Hopper vs Ampere

| Feature | A100 (Ampere) | H100 (Hopper) | B200 (Blackwell) | B300 (Ultra) |
|---|---|---|---|---|
| Process | TSMC 7nm | TSMC 4N | TSMC 4NP | TSMC 4NP |
| Transistors | 54.2B | 80B | 208B (2-die) | Enhanced |
| SMs | 108 | 132 | 148 | 160 |
| FP32 cores/SM | 64 | 128 | 128 | 128 |
| FP64 cores/SM | 32 | 64 | 64 | 64 |
| Tensor Core gen | 3rd | 4th | 5th | 5th |
| FP4/FP6 support | No | No | **Yes** | **Yes** |
| FP8 support | No | Yes | Yes | Yes |
| Peak FP16 TC (dense) | 312 TFLOPS | 989 TFLOPS | 2.25 PFLOPS | ~3.5 PFLOPS |
| Peak FP8 TC (dense) | — | 1979 TFLOPS | 4.5 PFLOPS | ~7 PFLOPS |
| Peak FP4 TC (dense) | — | — | 4.5 PFLOPS | ~14 PFLOPS |
| Shared mem + L1 | 192 KB | 256 KB | 256 KB | 256 KB |
| Max shared mem | 164 KB | 228 KB | 228 KB | 228 KB |
| TMEM per SM | — | — | 256 KB | 256 KB |
| L2 cache | 40 MB | 50 MB | 96-126 MB | ~126 MB |
| HBM | 80 GB, 2 TB/s | 80 GB, 3.35 TB/s | 192 GB, 8 TB/s | 288 GB, ~12 TB/s |
| NVLink | 600 GB/s | 900 GB/s | 1800 GB/s | 1800 GB/s |
| TMA | No | Yes | Yes (enhanced) | Yes |
| Tensor Memory (TMEM) | No | No | **Yes** | **Yes** |
| Thread Block Clusters | No | Yes | Yes (up to 16) | Yes |
| DSMEM | No | Yes | Yes | Yes |
| TDP | ~400W | 700W | 1000W | 1100-1400W |

## Kernel Development Notes for Blackwell

### What carries over from Hopper
- TMA programming model (tensor descriptors, async copy)
- Thread block clusters and distributed shared memory
- Warp specialization patterns
- cp.async and async barriers
- Same shared memory configuration (228 KB on CC 10.0)

### What's new to learn
1. **Tensor Memory (TMEM)**: New memory space for 5th-gen tensor core operands. Requires explicit allocation/deallocation via tcgen05 instructions.
2. **tcgen05 MMA**: New instruction format for matrix operations using TMEM. Different from Hopper's wgmma.
3. **FP4 data path**: New precision level for inference. Requires quantization-aware kernel design.
4. **Larger L2 cache (96 MB)**: More data can stay in L2. Tune tile sizes and access patterns to exploit this.
5. **Higher HBM bandwidth (8 TB/s)**: Ridge point shifts — kernels that were memory-bound on Hopper may become compute-bound on Blackwell.

### Practical impact on roofline

| GPU | Peak FP16 (TFLOPS) | Peak BW (TB/s) | Ridge Point (FP16 FLOP/byte) |
|---|---|---|---|
| A100 | 312 | 2.0 | 156 |
| H100 | 989 | 3.35 | 295 |
| **B200** | **2250** | **8.0** | **281** |
| **B300** | **~3500** | **~12.0** | **~292** |

Blackwell's ridge point is similar to Hopper's despite 2.3x more compute, because bandwidth also scaled 2.4x. **Many memory-bound kernels on Hopper will remain memory-bound on Blackwell.** The win for Blackwell comes from:
1. FP4/FP6: halves/thirds the weight memory traffic for inference
2. Larger L2 (96-126 MB): more working set fits in cache
3. TMEM: frees registers from accumulator duty, enabling larger tiles

### Compilation

```bash
# Target Blackwell data center
nvcc -arch=sm_100 -O3 -lineinfo kernel.cu

# Use architecture-specific features (tcgen05, TMEM)
nvcc -arch=sm_100a -O3 -lineinfo kernel.cu

# Family-compatible (sm_100 + sm_103)
nvcc -gencode arch=compute_100f,code=sm_100 kernel.cu

# Multi-arch: Hopper + Blackwell
nvcc -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_100,code=sm_100 \
     -gencode arch=compute_100,code=compute_100 \
     kernel.cu

# Consumer Blackwell (RTX 5090)
nvcc -arch=sm_120 -O3 -lineinfo kernel.cu
```

### Use CUTLASS for tcgen05
The tcgen05 instruction family is complex. NVIDIA strongly recommends using CUTLASS rather than writing raw PTX:

> "It is strongly recommended that device kernels utilize this complex feature set through CUTLASS."
> — CUDA C++ Programming Guide

CUTLASS 4.x includes Blackwell-optimized collective mainloops that use tcgen05 + TMEM internally.
