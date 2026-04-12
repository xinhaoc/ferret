# Nsight Compute Metric Reference

## Metric Naming Convention

```
unit__(subunit?)_(pipestage?)_quantity.(rollup).(submetric)
```

- **Unit**: hardware unit (`sm`, `smsp`, `l1tex`, `lts`, `dram`, `gpu`)
- **Rollup**: `.sum`, `.avg`, `.min`, `.max` (aggregation across instances)
- **Submetric**: `.per_second`, `.per_cycle_active`, `.pct_of_peak_sustained_active`, `.pct_of_peak_sustained_elapsed`

### Hardware Units

| Prefix | Unit |
|---|---|
| `sm__` | Streaming Multiprocessor |
| `smsp__` | SM Sub-Partition |
| `l1tex__` | L1 / Texture Cache |
| `lts__` / `ltc__` | L2 Cache |
| `dram__` | Device Memory (HBM/GDDR) |
| `fbpa__` | Framebuffer Partition (between L2 and DRAM) |
| `gpu__` | GPU-level aggregate |

## Tier 1: Essential Metrics (always collect)

### Timing
| Metric | Unit | Description |
|---|---|---|
| `gpu__time_duration.avg` | ns | Kernel wall-clock execution time |
| `sm__cycles_elapsed.avg` | cycles | Total elapsed SM cycles |
| `sm__cycles_active.avg` | cycles | SM active cycles |

### DRAM Traffic
| Metric | Unit | Description |
|---|---|---|
| `dram__bytes.sum` | bytes | Total DRAM bytes (read + write) |
| `dram__bytes_read.sum` | bytes | DRAM bytes read |
| `dram__bytes_write.sum` | bytes | DRAM bytes written |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | % | DRAM throughput vs peak |

### Compute / FLOP Counts

**FP32:**
| Metric | Description |
|---|---|
| `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum` | FP32 ADD instructions (1 FLOP each) |
| `smsp__sass_thread_inst_executed_op_fmul_pred_on.sum` | FP32 MUL instructions (1 FLOP each) |
| `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` | FP32 FMA instructions (**2 FLOPs each**) |

**FP16 (packed half2 — each instruction operates on 2 FP16 values):**
| Metric | Description |
|---|---|
| `smsp__sass_thread_inst_executed_op_hadd_pred_on.sum` | FP16 ADD (HADD2: packed, **2 FLOPs each**) |
| `smsp__sass_thread_inst_executed_op_hmul_pred_on.sum` | FP16 MUL (HMUL2: packed, **2 FLOPs each**) |
| `smsp__sass_thread_inst_executed_op_hfma_pred_on.sum` | FP16 FMA (HFMA2: packed, **4 FLOPs each**: 2 mul + 2 add) |

**FP64:**
| Metric | Description |
|---|---|
| `smsp__sass_thread_inst_executed_op_dadd_pred_on.sum` | FP64 ADD |
| `smsp__sass_thread_inst_executed_op_dmul_pred_on.sum` | FP64 MUL |
| `smsp__sass_thread_inst_executed_op_dfma_pred_on.sum` | FP64 FMA (**2 FLOPs each**) |

**Tensor Core:**
| Metric | Description |
|---|---|
| `sm__inst_executed_pipe_tensor.sum` | Tensor core instructions executed |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | Tensor pipe utilization % |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | HMMA (half-precision MMA) utilization % |

### Occupancy
| Metric | Unit | Description |
|---|---|---|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | % | Achieved occupancy |
| `sm__maximum_warps_per_active_cycle_pct` | % | Theoretical occupancy |

### Launch Configuration
| Metric | Description |
|---|---|
| `launch__grid_size` | Total blocks in grid |
| `launch__block_size` | Threads per block |
| `launch__registers_per_thread` | Registers per thread |
| `launch__shared_mem_per_block_static` | Static shared memory (bytes) |
| `launch__shared_mem_per_block_dynamic` | Dynamic shared memory (bytes) |

## Tier 2: Bottleneck Diagnosis

### SM Throughput
| Metric | Description |
|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Overall SM throughput vs peak |
| `sm__inst_executed.avg.per_cycle_active` | IPC (instructions per active cycle) |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | Combined compute+memory throughput % |

### Warp Stall Reasons

These indicate WHY warps are stalled. Focus on stalls only if schedulers fail to issue every cycle.

| Metric | Stall Reason | What It Means |
|---|---|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | Memory dependency | Waiting for global/local memory load. **Primary memory-bound indicator.** |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | Execution dependency | Waiting for shared memory, MMA, or special function unit result |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | Fixed-latency wait | Waiting for coupled math ops (FMA, ALU, tensor) |
| `smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio` | Pipe busy | Math pipeline input FIFO full. **Indicates compute saturation.** |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | Barrier | Waiting at `__syncthreads()` |
| `smsp__average_warps_issue_stalled_membar_per_issue_active.ratio` | Memory barrier | Waiting for `__threadfence()` |
| `smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio` | LG throttle | Local/global memory pipe FIFO full |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | MIO throttle | Shared memory / special ops pipe FIFO full |
| `smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio` | TEX throttle | Texture pipe FIFO full |
| `smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio` | Not selected | Warp was eligible but not picked. High value = may be able to reduce occupancy. |
| `smsp__average_warps_issue_stalled_no_instructions_per_issue_active.ratio` | No instructions | Instruction cache miss or branch resolving |
| `smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio` | Sleeping | All threads blocked/sleeping |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | Drain | Exited warp waiting to drain outstanding memory writes |

### PC Sampling Stall Metrics (alternate, more precise)

Same stall reasons but via hardware PC sampling (`pcsamp`):

```
smsp__pcsamp_warps_issue_stalled_long_scoreboard
smsp__pcsamp_warps_issue_stalled_short_scoreboard
smsp__pcsamp_warps_issue_stalled_wait
smsp__pcsamp_warps_issue_stalled_math_pipe_throttle
smsp__pcsamp_warps_issue_stalled_barrier
smsp__pcsamp_warps_issue_stalled_selected  (not a stall — instruction was issued)
...
```

### Cache Metrics
| Metric | Description |
|---|---|
| `lts__t_bytes.sum` | Total L2 cache bytes |
| `lts__t_sectors_hit.sum` | L2 hit sectors |
| `lts__t_sectors_miss.sum` | L2 miss sectors |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | L2 throughput vs peak |
| `l1tex__t_bytes.sum` | Total L1 cache bytes |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | L1 global load sectors (for coalescing analysis) |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | L1 global load requests |
| `l1tex__data_bank_conflicts_pipe_lsu.sum` | L1 bank conflicts |
| `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | L1 throughput vs peak |

## Tier 3: Detailed Optimization

### Pipeline Utilization
| Metric | Description |
|---|---|
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | FMA pipeline utilization |
| `sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active` | ALU pipeline utilization |
| `sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active` | FP64 pipeline utilization |
| `sm__pipe_lsu_cycles_active.avg.pct_of_peak_sustained_active` | Load/Store Unit utilization |
| `sm__pipe_xu_cycles_active.avg.pct_of_peak_sustained_active` | Transcendental unit utilization |
| `sm__pipe_tex_cycles_active.avg.pct_of_peak_sustained_active` | Texture pipeline utilization |

### Scheduler
| Metric | Description |
|---|---|
| `smsp__warps_active.avg.per_cycle_active` | Active warps per sub-partition per cycle |
| `smsp__warps_eligible.avg.per_cycle_active` | Eligible (ready to issue) warps |
| `smsp__issue_active.avg.per_cycle_active` | Warps issuing instructions per cycle |
| `smsp__average_warp_latency` | Average warp latency (cycles/instruction) |

### Occupancy Limiters
| Metric | Description |
|---|---|
| `launch__occupancy_limit_registers` | Occupancy limited by register usage |
| `launch__occupancy_limit_shared_mem` | Occupancy limited by shared memory |
| `launch__occupancy_limit_warps` | Occupancy limited by block size (max warps/block) |
| `launch__occupancy_limit_blocks` | Occupancy limited by max blocks per SM |
| `launch__waves_per_multiprocessor` | Waves per SM (fractional = tail effect) |

### Memory Transaction Sizes
| Metric | Description |
|---|---|
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | Global load sectors through L1 |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | Global load requests |

Sectors/request ratio indicates coalescing efficiency:
- **Ideal**: 1 sector/request (32B per request, fully coalesced)
- **Worst**: 32 sectors/request (1 sector per thread, completely uncoalesced)

### Shared Memory
| Metric | Description |
|---|---|
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | Shared memory bank conflicts (total) |
| `launch__shared_mem_per_block` | Total shared memory per block |

## FLOP Counting Formulas

```
FP32_FLOPs = fadd + fmul + 2 * ffma
FP16_FLOPs = 2 * hadd + 2 * hmul + 4 * hfma    (packed half2: each instruction = 2 elements)
FP64_FLOPs = dadd + dmul + 2 * dfma
```

FP32/FP64: FMA = **2 FLOPs** (one multiply + one add).
FP16: Instructions are packed HADD2/HMUL2/HFMA2, operating on 2 FP16 values per instruction. So hadd/hmul = **2 FLOPs**, hfma = **4 FLOPs** (2 mul + 2 add).

## Speed of Light (SOL) Interpretation

SOL% = percentage of theoretical peak throughput achieved.

| SM SOL% vs Memory SOL% | Classification |
|---|---|
| Memory SOL% >> Compute SOL% | **Memory-bound** |
| Compute SOL% >> Memory SOL% | **Compute-bound** |
| Both high and similar | **Balanced** (multiple bottlenecks) |
| Both low | **Latency-bound** (check stalls, occupancy) |

## Discovery Commands

```bash
ncu --list-sets                              # Available metric sets
ncu --list-sections                          # Available sections
ncu --query-metrics                          # All metrics for this GPU
ncu --query-metrics-mode suffix --metrics sm__inst_executed  # Suffixes for a metric
ncu --list-chips                             # Supported GPU architectures
```
