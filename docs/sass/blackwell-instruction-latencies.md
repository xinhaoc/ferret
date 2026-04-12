# Blackwell SM100 Instruction Latencies & Microarchitecture

Data sourced from microbenchmark papers and NVIDIA documentation.
Use this when analyzing SASS output and optimizing instruction-level performance.

## Compute Instruction Latencies

| Instruction | True Latency (cycles) | Notes |
|---|---|---|
| FP32 ALU (FADD, FMUL, FFMA) | 4 | Same for pure and mixed workloads |
| INT32 ALU (IADD, IMUL) | 4 | Completion latency ~17 cycles |
| FP64 | ~64 | Only 2 FP64 units per SM; may be emulated via FP32 |
| tcgen05.mma (tensor core) | 11 | Down from Hopper's 32-128 cycles |
| FP4 tensor core | 12.6 | Slightly higher than FP8/BF16 |

## Memory Latencies

| Access | Latency (cycles) | Notes |
|---|---|---|
| Register file | 0 | Direct access |
| TMEM read | 420 | Tensor Memory, on-chip |
| TMEM bandwidth | 16 TB/s read, 8 TB/s write | Per SM |
| Shared memory (SMEM) | ~20-30 | Bank conflict adds latency |
| SMEM bandwidth | 128 bytes/cycle | Per SM, can be bottleneck for SMEM-bound MMA |
| L1 cache hit | 30-40 | 128 KB per SM |
| L2 cache hit | 358 | 96 MB unified (B200) |
| Global memory (HBM) | 877 | Via L2 miss → HBM3E |

## Tensor Core (tcgen05.mma) Details

### Throughput by shape
- M=128 or M=256: near 100% peak for N≥128
- **M=64: max 50% theoretical peak** — fundamental hardware limitation
- Larger instruction shapes = higher throughput (always prefer max N)
- Both operands from SMEM: SMEM-bandwidth-bound below N=128

### Instruction Level Parallelism (ILP)
- Max useful ILP: 6 independent MMA instructions
- Saturates at ~25 active warps
- Per-thread scheduling (Blackwell change): single thread issues MMA on behalf of CTA
- No 4-warp synchronization needed (unlike Hopper wgmma)

### SASS instruction names
| PTX | SASS | Precision |
|---|---|---|
| tcgen05.mma kind::f16 | UTCHMMA | BF16/FP16 |
| tcgen05.mma kind::mxf8f6f4 | QMMA | FP8/FP6/FP4 |

## Barrier / Synchronization Costs

| Operation | Estimated cost | Notes |
|---|---|---|
| mbarrier.arrive | ~4-8 cycles | Per-warp |
| mbarrier.try_wait (hit) | ~4-8 cycles | Immediate if phase matched |
| mbarrier.try_wait (miss) | 100+ cycles | Spins until phase flips |
| bar.sync | ~20-30 cycles | Block-wide sync |
| cluster.sync | ~50-100 cycles | Cross-CTA, depends on cluster size |
| __syncthreads | ~20-30 cycles | Same as bar.sync |
| fence.proxy.async | ~4 cycles | Lightweight |

## Register File

- 64K 32-bit registers per SM (65536 total)
- Max 255 registers per thread
- At 128 threads: 255 × 128 = 32640 registers used (49.8% of budget)
- At 256 threads: 255 × 256 = 65280 registers (99.7% — near full)
- **Low register usage (e.g., 48 regs) leaves massive headroom**
- More registers = more values cached = fewer reload instructions
- cuBLAS uses 255 registers for their BF16 GEMM kernel

## Warp Scheduling

- Blackwell uses **per-thread scheduling** for tensor ops (not warp-synchronous)
- Eliminates Hopper's 4-warp wgmma synchronization overhead (18-23% reduction)
- Smoother throughput ramp-up for 1-9 dependent instruction chains
- Better performance at low warp counts (1-4) compared to Hopper

## Optimization Implications

### For memory-bound kernels (like skinny GEMM with M=64):
1. **Instruction count matters for latency hiding** — more instructions per pipeline
   stage = longer compute time = less effective overlap with TMA loads
2. **Register usage is critical** — 48 regs leaves 207 regs unused per thread.
   The compiler may be spilling/recomputing values that could be cached.
   Try `--maxrregcount=255` or `--maxrregcount=128` to see effect.
3. **Barrier overhead is significant** — each mbarrier.arrive + try_wait is 8-16
   cycles. With 4 barriers per K-block × 112 K-blocks = ~1792 barrier cycles
   per tile. At 2 GHz = ~0.9 µs of pure barrier overhead.
4. **tcgen05.mma at M=64 = 50% peak** — tensor cores complete quickly, leaving
   more idle time. Reducing non-MMA instructions fills that idle time more
   effectively.

### Compiler flags to try:
- `--maxrregcount=255` — allow maximum register usage
- `--maxrregcount=128` — compromise between registers and occupancy
- `-Xptxas -O3` — aggressive PTX optimization
- `-Xptxas --allow-expensive-optimizations=true` — enable costly opts
- `--use_fast_math` — faster math intrinsics
- `-lineinfo` vs no `-lineinfo` — debug info can affect register allocation
