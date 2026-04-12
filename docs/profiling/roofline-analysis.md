# Roofline Analysis Guide

## What Is It

The roofline model classifies a kernel as **compute-bound** or **memory-bound** by comparing its arithmetic intensity against the hardware's compute-to-bandwidth ratio.

```
Performance = min(Peak_Compute, Arithmetic_Intensity × Peak_Bandwidth)
```

## Key Concepts

### Arithmetic Intensity (AI)

```
AI = Total FLOPs / Total DRAM Bytes Transferred  (FLOPs/byte)
```

- Low AI → memory-bound (kernel moves lots of data per FLOP)
- High AI → compute-bound (kernel does lots of math per byte)

### Ridge Point

Where the memory ceiling meets the compute ceiling:

```
Ridge Point = Peak Compute (FLOP/s) / Peak Bandwidth (byte/s)  (FLOPs/byte)
```

- AI < Ridge Point → **memory-bound**
- AI > Ridge Point → **compute-bound**

### Ridge Points by GPU

| GPU | Peak FP16 TFLOPS | Peak BW (GB/s) | Ridge Point (FP16 FLOP/byte) |
|---|---|---|---|
| V100-SXM2 | 125 | 900 | 139 |
| A100-SXM | 312 | 2039 | 153 |
| H100-SXM | 989 | 3352 | 295 |
| H200 | 989 | 4800 | 206 |
| **B200 (HGX)** | **2250** | **7700** | **292** |
| RTX 5090 | 419 | 1792 | 234 |

Note: B200's ridge point (281) is similar to H100's (295) despite 2.3x more compute, because bandwidth also scaled 2.4x. Many memory-bound kernels on Hopper remain memory-bound on Blackwell.

## How to Collect Data with ncu

### Method 1: Roofline section set
```bash
ncu --section SpeedOfLight_RooflineChart -o roofline_report ./my_kernel
```

### Method 2: Collect specific metrics
```bash
ncu --csv --metrics \
  gpu__time_duration.avg,\
  dram__bytes.sum,\
  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_hmul_pred_on.sum \
  -k my_kernel ./my_app
```

### Method 3: Hierarchical roofline (L1, L2, DRAM)
```bash
ncu --csv --metrics \
  gpu__time_duration.avg,\
  dram__bytes.sum,\
  lts__t_bytes.sum,\
  l1tex__t_bytes.sum,\
  smsp__sass_thread_inst_executed_op_hfma_pred_on.sum \
  -k my_kernel ./my_app
```

## Computing the Numbers

### Step 1: Count FLOPs

```
FP32_FLOPs = fadd + fmul + 2 × ffma
FP16_FLOPs = 2 × hadd + 2 × hmul + 4 × hfma  (packed half2: each inst = 2 elements)
FP64_FLOPs = dadd + dmul + 2 × dfma
Total_FLOPs = FP32_FLOPs + FP16_FLOPs + FP64_FLOPs + Tensor_FLOPs
```

### Step 2: Get memory traffic

```
DRAM_Bytes = dram__bytes.sum  (or dram__bytes_read.sum + dram__bytes_write.sum)
```

### Step 3: Compute arithmetic intensity

```
AI = Total_FLOPs / DRAM_Bytes
```

### Step 4: Compute achieved performance

```
Duration_s = gpu__time_duration.avg / 1e9
Achieved_TFLOPS = Total_FLOPs / Duration_s / 1e12
Achieved_BW_GB_s = DRAM_Bytes / Duration_s / 1e9
```

### Step 5: Compare against roofline

```
Roofline_TFLOPS = min(Peak_TFLOPS, AI × Peak_BW_GB_s / 1000)
Efficiency = Achieved_TFLOPS / Roofline_TFLOPS
```

## Hierarchical Roofline

Uses multiple memory bandwidth lines:

| Level | Traffic Metric | Bandwidth | Description |
|---|---|---|---|
| DRAM | `dram__bytes.sum` | HBM peak | Main memory bandwidth ceiling |
| L2 | `lts__t_bytes.sum` | L2 peak (~4-6× DRAM) | L2 cache bandwidth ceiling |
| L1 | `l1tex__t_bytes.sum` | L1 peak (~10-20× DRAM) | L1 cache bandwidth ceiling |

Each level has its own ridge point. A kernel might be:
- Compute-bound relative to DRAM
- Memory-bound relative to L1

This reveals opportunities: if L2 traffic >> DRAM traffic, the kernel benefits from caching. If they're similar, the data is streaming through without reuse.

## Interpreting Results

### Memory-Bound (AI < Ridge Point)

**Key question**: How close to peak bandwidth?

| BW Utilization | Interpretation | Action |
|---|---|---|
| > 80% | Near peak, efficient | Reduce total bytes (fusion, recomputation, compression) |
| 50-80% | Moderate | Check coalescing, vectorized loads, L2 locality |
| < 50% | Poor | Likely coalescing issues, strided access, or L2 thrashing |

**Common fixes:**
- Coalesced memory access (consecutive threads access consecutive addresses)
- Vectorized loads (`float4`, `int4`)
- Shared memory tiling to reduce DRAM traffic
- Operator fusion to avoid writing intermediates to DRAM
- TMA (Tensor Memory Accelerator) on Hopper+ for async bulk copies

### Compute-Bound (AI > Ridge Point)

**Key question**: How close to peak FLOPS?

| Compute Utilization | Interpretation | Action |
|---|---|---|
| > 70% | Near peak | At hardware limit, algorithmic changes needed |
| 40-70% | Moderate | Check ILP, instruction mix, tensor core usage |
| < 40% | Poor | Likely low occupancy, warp stalls, or wrong instruction types |

**Common fixes:**
- Use tensor cores (wmma/mma PTX instructions, CUTLASS, cuBLAS)
- Increase ILP (more independent ops per thread)
- Reduce register pressure to improve occupancy
- Ensure matrix dimensions align with tile sizes

### Latency-Bound (Both SOL% Low)

Neither compute nor memory is well-utilized. Typically caused by:
- Excessive `__syncthreads()` barriers
- Divergent branches
- Grid too small (not enough blocks to fill all SMs)
- Serial dependencies between instructions

**Diagnosis:** Check warp stall reasons:
- `stall_barrier` → too many syncs
- `stall_no_instructions` → instruction cache misses or branch resolution
- `stall_not_selected` → warps are eligible but idle (scheduler has choices, could reduce occupancy)

## Practical Example

```
Kernel: flash_attn_fwd
  batch=4, heads=16, seq_len=2048, head_dim=128
Duration: 0.95 ms (950 μs)
DRAM read: 128 MB, DRAM write: 32 MB → Total = 160 MB
FP16 FLOPs: 4 * 4 * 16 * 2048 * 2048 * 128 = 1.374e12
GPU: H100-SXM (peak FP16: 989 TFLOPS, peak BW: 3352 GB/s)

AI = 1.374e12 / 160e6 = 8588 FLOPs/byte
Ridge point = 989e12 / 3352e9 = 295 FLOPs/byte

AI (8588) >> Ridge (295) → COMPUTE-BOUND

Achieved TFLOPS = 1.374e12 / 0.95e-3 / 1e12 = 1446 TFLOPS
Roofline = min(989, 8588 × 3352 / 1000) = 989 TFLOPS
Efficiency = 1446 / 989 = 146%?? → Achieved > peak means the FLOP formula
  overcounts (counts both QK and PV, but causal masking skips ~half of QK).
  Adjust: effective FLOPs ≈ 0.75 × 1.374e12 = 1.03e12
  Adjusted TFLOPS = 1.03e12 / 0.95e-3 / 1e12 = 1084 TFLOPS
  Adjusted efficiency = 1084 / 989 = 110% → still above peak, indicating
  the kernel benefits from L2 cache (effective bandwidth > DRAM bandwidth)

Takeaway: for attention kernels, the standard "4*B*H*N*N*D" FLOP formula
is approximate. Causal masking and L2 caching make simple roofline
classification unreliable. Use ncu's Speed Of Light section instead.
```

## Known Limitations

- ncu does not have correct metrics for INT64/INT32/INT16/INT8 rooflines
- FP16 FLOP counting via SASS instructions may miss tensor core FLOPs (use `sm__ops_path_tensor_*` for tensor workloads)
- Roofline assumes all FLOPs and all bytes are "useful" — it doesn't account for redundant computation or wasted traffic
- Multi-pass profiling (replay mode) can skew cache behavior metrics
