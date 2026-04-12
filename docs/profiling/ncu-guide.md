# Nsight Compute (ncu) CLI Reference

## Quick Start

```bash
# Basic: profile all kernels, default metric set
ncu ./my_kernel

# Full metrics, save report, filter kernel by name
ncu --set full -o report -k my_kernel_name ./my_kernel

# CSV output for parsing (--page raw gives one row per metric)
ncu --csv --page raw --set full -k my_kernel ./my_kernel

# Specific metrics only (faster than --set full)
ncu --csv --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.avg ./my_kernel

# Skip warmup launches, profile 1
ncu -s 10 -c 1 --set full -o report ./my_kernel

# Profile a Python script
ncu --target-processes all --set full -k regex:flash.* python my_script.py
```

## Key Flags

### Kernel Filtering

| Flag | Description | Example |
|---|---|---|
| `-k, --kernel-name` | Filter kernels by name (exact or regex) | `-k my_kernel` or `-k regex:flash.*` |
| `--kernel-name-base` | What name to match against | `function` (default), `demangled`, `mangled` |
| `--kernel-id` | Filter by context:stream:name:invocation | `--kernel-id ::foo:2` |
| `-s, --launch-skip` | Skip N kernel launches before profiling | `-s 10` (skip warmup) |
| `-c, --launch-count` | Profile only N kernel launches | `-c 1` |
| `--launch-skip-before-match` | Skip N launches before filters even apply | `--launch-skip-before-match 5` |
| `--filter-mode` | How skip/count are applied | `global` (default), `per-gpu`, `per-launch-config` |

### Metric Collection

| Flag | Description | Example |
|---|---|---|
| `--set` | Predefined section set | `basic` (default), `full` |
| `--section` | Specific sections to collect | `--section SpeedOfLight,Occupancy` |
| `--metrics` | Specific metric names (overrides --set) | `--metrics dram__bytes_read.sum,sm__warps_active.avg.pct_of_peak_sustained_active` |
| `--list-sets` | Show available sets | |
| `--list-sections` | Show available sections | |
| `--query-metrics` | Show all available metrics for the GPU | |
| `--query-metrics-mode` | Level of detail for --query-metrics | `base`, `suffix`, `all` |

### Output

| Flag | Description | Example |
|---|---|---|
| `--csv` | Comma-separated output (implies `--print-units base`) | |
| `--page` | Output page type | `details` (default), `raw`, `source`, `session` |
| `-o, --export` | Save .ncu-rep report file | `-o my_report` |
| `-i, --import` | Load existing .ncu-rep file | `-i my_report.ncu-rep` |
| `-f, --force-overwrite` | Overwrite existing output files | |
| `--print-summary` | Aggregate summary | `none`, `per-gpu`, `per-kernel`, `per-nvtx` |
| `--print-metric-name` | Show metric name style | `label`, `name`, `label-name` |

### Replay & Profiling Control

| Flag | Description | Example |
|---|---|---|
| `--replay-mode` | How metrics are collected across passes | `kernel` (default), `application`, `range` |
| `--clock-control` | Lock GPU clocks for reproducibility | `base`, `boost` (default), `none` |
| `--cache-control` | Flush caches between replay passes | `all` (default), `none` |
| `--profile-from-start` | Start profiling at app launch | `on` (default), `off` |
| `--target-processes` | Which processes to profile | `application-only`, `all` (default) |

### NVTX Filtering

| Flag | Description | Example |
|---|---|---|
| `--nvtx` | Enable NVTX support | |
| `--nvtx-include` | Only profile kernels inside NVTX range | `--nvtx-include "training/"` |
| `--nvtx-exclude` | Exclude kernels inside NVTX range | `--nvtx-exclude "warmup/"` |

## Available Sections

| Section ID | What it shows |
|---|---|
| `SpeedOfLight` | Compute SOL% and Memory SOL% (high-level throughput overview) |
| `SpeedOfLight_RooflineChart` | Roofline chart data |
| `SpeedOfLight_HierarchicalSingleRooflineChart` | Hierarchical roofline (L1, L2, DRAM) |
| `LaunchStats` | Grid dims, block size, registers, shared memory |
| `Occupancy` | Theoretical vs achieved occupancy |
| `SchedulerStats` | Warp scheduler activity: active, eligible, issued warps |
| `WarpStateStats` | Warp stall reasons and cycles per instruction |
| `MemoryWorkloadAnalysis` | L1, L2, shared memory, DRAM usage breakdown |
| `ComputeWorkloadAnalysis` | SM pipeline utilization, IPC |
| `InstructionStats` | SASS instruction mix and frequencies |
| `SourceCounters` | Source-level branch efficiency and stall sampling |

## CSV Output Format

With `--csv --page raw`, ncu outputs one row per metric per kernel:

```
"ID","Kernel Name","Metric Name","Metric Value"
"0","my_kernel","gpu__time_duration.avg","145000"
"0","my_kernel","dram__bytes_read.sum","134217728"
...
```

- All values are in **base units** (bytes, nanoseconds, counts)
- Kernel launches are identified by `ID` column
- Use `--print-metric-name name` to get raw metric names (not human labels)

With `--csv --page details`, output is organized by sections with columns: metric label, unit, value.

## Common Recipes

### Profile a single kernel with full metrics
```bash
ncu --set full -s 10 -c 1 -k my_kernel -o report ./app
```

### Get just timing and memory bandwidth
```bash
ncu --csv --metrics gpu__time_duration.avg,dram__bytes_read.sum,dram__bytes_write.sum -k my_kernel ./app
```

### Get roofline data
```bash
ncu --section SpeedOfLight_RooflineChart -o roofline_report ./app
```

### Get occupancy limiters
```bash
ncu --csv --metrics launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps,launch__occupancy_limit_blocks,sm__warps_active.avg.pct_of_peak_sustained_active -k my_kernel ./app
```

### Get warp stall breakdown
```bash
ncu --csv --section WarpStateStats -k my_kernel ./app
```

### Re-analyze a saved report
```bash
ncu -i report.ncu-rep --csv --page raw
ncu -i report.ncu-rep --page details
```

### Profile Python CUDA kernel
```bash
ncu --target-processes all -k regex:my_kernel.* -s 5 -c 1 --set full python run.py
```

### Multi-GPU / MPI
```bash
mpirun -np 4 ncu -o report_%q{OMPI_COMM_WORLD_RANK} --set full ./app
```

### Deferred start (controlled by cudaProfilerStart/Stop in code)
```bash
ncu --profile-from-start off --set full -o report ./app
```

## Tips

- `--set full` is **very slow** (100-1000x kernel runtime). Use `--metrics` with specific metrics for faster iteration.
- Always use `-s` (launch-skip) to skip warmup launches.
- `-c 1` profiles just one launch — good for deterministic analysis.
- `--clock-control base` locks clocks for reproducible measurements.
- `--cache-control all` (default) flushes caches between replay passes for consistent results.
- For kernels using `cudaProfilerStart()`/`cudaProfilerStop()`, use `--replay-mode range`.
- Filename macros: `%h` (hostname), `%p` (PID), `%i` (auto-increment), `%q{ENV}` (env var).
