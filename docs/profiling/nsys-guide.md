# Nsight Systems (nsys) CLI Reference

## Quick Start

```bash
# Basic CUDA profiling
nsys profile ./my_app

# CUDA + NVTX markers, save with specific name
nsys profile --trace=cuda,nvtx -o my_profile ./my_app

# Profile Python CUDA app
nsys profile --trace=cuda,nvtx --target-processes=all python train.py

# Profile, then generate stats summary
nsys profile --stats=true -o results ./my_app

# Export existing report to SQLite for analysis
nsys export --type=sqlite results.nsys-rep

# Generate kernel timing summary from report
nsys stats --report cuda_gpu_kern_sum results.nsys-rep
```

## Subcommands

| Command | Purpose |
|---|---|
| `nsys profile` | Profile an application |
| `nsys export` | Convert .nsys-rep to SQLite/Arrow/JSON/etc |
| `nsys stats` | Generate statistical reports from results |
| `nsys analyze` | Run expert-system rules on results |
| `nsys launch` | Interactive: launch app in profiling env |
| `nsys start` / `stop` | Interactive: begin/end data collection |
| `nsys shutdown` | Interactive: disconnect and force exit |

## `nsys profile` Key Flags

### What to Trace (`--trace`)

Comma-separated, no spaces. Default: `cuda,opengl,nvtx,osrt`

**Most useful for kernel development:**
| Trace | Description |
|---|---|
| `cuda` | CUDA API calls and kernel/memcpy/memset activities |
| `nvtx` | NVIDIA Tools Extension markers/ranges |
| `osrt` | OS runtime (pthreads, file I/O) |
| `cublas` | cuBLAS calls |
| `cudnn` | cuDNN calls (Linux) |
| `mpi` | MPI operations |
| `none` | Disable all tracing |

**Full list:**
`cuda`, `cuda-sw`, `nvtx`, `osrt`, `opengl`, `vulkan`, `cublas`, `cublas-verbose`, `cudnn`, `cudla`, `cusolver`, `cusparse`, `openacc`, `openmp`, `mpi`, `ucx`, `oshmem`, `python-gil`, `gds`, `none`

### Output Control
| Flag | Description |
|---|---|
| `-o, --output <name>` | Output file path (auto-adds .nsys-rep). Macros: `%h` hostname, `%p` PID, `%q{ENV}` |
| `-f, --force-overwrite` | Overwrite existing files |
| `--stats=true` | Auto-generate summary stats after profiling |
| `--export <format>` | Auto-export to sqlite/arrow/jsonlines alongside .nsys-rep |
| `-w, --show-output` | Show app stdout/stderr (default: true) |

### Collection Timing
| Flag | Description |
|---|---|
| `-d, --duration <sec>` | Max collection duration |
| `-y, --delay <sec>` | Delay before collection starts |
| `--start-later` | Wait for `nsys start` command |

### CUDA-Specific
| Flag | Description |
|---|---|
| `--cuda-memory-usage=true` | Track GPU memory allocations per kernel (**high overhead**) |
| `--cuda-trace-all-apis=true` | Trace all CUDA APIs (not just performance-critical ones) |
| `--cudabacktrace=kernel` | Collect CPU backtrace for kernel launches |
| `--cuda-graph-trace=graph\|node` | Graph: unified units. Node: individual activities (higher overhead) |
| `--cuda-flush-interval <ms>` | Buffer flush interval. Set ~10000 for collections >30s |

### Capture Control (Programmatic Start/Stop)
| Flag | Description |
|---|---|
| `--capture-range=cudaProfilerApi` | Only collect between `cudaProfilerStart()` / `cudaProfilerStop()` |
| `--capture-range=nvtx` | Only collect inside specific NVTX range |
| `--nvtx-capture="range@domain"` | Which NVTX range triggers collection |
| `--capture-range-end=repeat:N` | Repeat capture N times |

### Process Control
| Flag | Description |
|---|---|
| `--target-processes=all` | Profile all child processes (important for Python) |
| `--kill=sigterm` | Signal to send on collection end |
| `--wait=all` | Wait for all processes to terminate |
| `-e, --env-var "A=B,C=D"` | Set env vars for target app |

### CPU / Sampling
| Flag | Description |
|---|---|
| `-s, --sample=process-tree` | CPU sampling scope |
| `--sampling-frequency=1000` | Sampling rate in Hz (100-8000) |
| `-b, --backtrace=dwarf\|fp\|lbr` | CPU backtrace method |

### GPU Metrics
| Flag | Description |
|---|---|
| `--gpu-metrics-devices=all` | Collect GPU-level metrics |
| `--gpu-metrics-frequency=10000` | GPU metric sampling rate in Hz |

### Framework Integration
| Flag | Description |
|---|---|
| `--pytorch=autograd-nvtx` | Auto-annotate PyTorch operations with NVTX |
| `--python-sampling=true` | Sample Python call stacks |

## `nsys export` — Convert Reports

```bash
# Export to SQLite (most common)
nsys export --type=sqlite results.nsys-rep

# Export with custom output name
nsys export --type=sqlite --output=results.sqlite results.nsys-rep

# Force overwrite
nsys export --type=sqlite -f true results.nsys-rep

# Only export specific tables
nsys export --type=sqlite --tables="CUPTI_ACTIVITY_KIND_KERNEL,StringIds" results.nsys-rep
```

### Output Formats
`sqlite`, `arrow`, `arrowdir`, `parquetdir`, `hdf`, `jsonlines`, `text`, `info`

### SQLite Schema — Key Tables

| Table | Contents |
|---|---|
| `CUPTI_ACTIVITY_KIND_KERNEL` | Kernel executions: name, start, end, grid, block, stream, device |
| `CUPTI_ACTIVITY_KIND_MEMCPY` | Memory copies: kind, bytes, start, end, stream |
| `CUPTI_ACTIVITY_KIND_MEMSET` | Memory sets |
| `CUDA_API_TRACE` | CUDA API calls with timestamps |
| `NVTX_EVENTS` | NVTX markers and ranges |
| `StringIds` | Maps numeric IDs to human-readable strings (kernel names, etc.) |
| `GPU_METRICS_*` | GPU utilization, memory, thermal data |
| `TARGET_INFO_*` | System hardware/OS info |

### Querying the SQLite Database

```sql
-- List all kernels with timing
SELECT s.value AS kernel_name,
       k.start, k.end, (k.end - k.start) AS duration_ns,
       k.gridX, k.gridY, k.gridZ,
       k.blockX, k.blockY, k.blockZ,
       k.streamId
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
ORDER BY k.start;

-- Kernel timing summary
SELECT s.value AS kernel_name,
       COUNT(*) AS count,
       AVG(k.end - k.start) AS avg_ns,
       MIN(k.end - k.start) AS min_ns,
       MAX(k.end - k.start) AS max_ns,
       SUM(k.end - k.start) AS total_ns
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
GROUP BY s.value
ORDER BY total_ns DESC;

-- Memory copy summary
SELECT copyKind, COUNT(*) AS count,
       SUM(bytes) AS total_bytes,
       SUM(end - start) AS total_ns
FROM CUPTI_ACTIVITY_KIND_MEMCPY
GROUP BY copyKind;
```

## `nsys stats` — Statistical Reports

```bash
# Default summary
nsys stats results.nsys-rep

# Specific report
nsys stats --report cuda_gpu_kern_sum results.nsys-rep

# Multiple reports in CSV
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum --format csv results.nsys-rep

# Save to file
nsys stats --report cuda_gpu_kern_sum --format csv --output kernel_stats results.nsys-rep
```

### Available Reports

**CUDA:**
| Report | Description |
|---|---|
| `cuda_api_sum` | CUDA API call statistics |
| `cuda_api_trace` | Full CUDA API call trace |
| `cuda_gpu_kern_sum` | Kernel execution summary (count, avg/min/max time) |
| `cuda_gpu_kern_gb_sum` | Kernel summary with grid/block dimensions |
| `cuda_gpu_mem_size_sum` | Memory ops by size |
| `cuda_gpu_mem_time_sum` | Memory ops by duration |
| `cuda_gpu_sum` | Combined kernel + memory ops |
| `cuda_gpu_trace` | Full GPU execution log |
| `cuda_kern_exec_sum` | Launch vs execution time comparison |

**NVTX:**
| Report | Description |
|---|---|
| `nvtx_sum` | NVTX range statistics |
| `nvtx_pushpop_sum` | Push/pop range stats |
| `nvtx_gpu_proj_sum` | NVTX ranges projected onto GPU timeline |
| `nvtx_kern_sum` | Kernels annotated by NVTX range |

**Other:**
`osrt_sum`, `mpi_event_sum`, `openmp_sum`, `um_sum`, `um_total_sum`

## `nsys analyze` — Expert Rules

```bash
nsys analyze --rule=gpu_gaps results.nsys-rep
nsys analyze --rule=cuda_api_sync results.nsys-rep
nsys analyze --rule=all results.nsys-rep
```

| Rule | What it checks |
|---|---|
| `cuda_memcpy_async` | Async memcpy best practices |
| `cuda_memcpy_sync` | Flags synchronous memory copies |
| `cuda_memset_sync` | Flags synchronous memsets |
| `cuda_api_sync` | Identifies synchronizing CUDA API calls |
| `gpu_gaps` | Detects idle gaps on GPU timeline |
| `gpu_time_util` | GPU time utilization analysis |

## Common Recipes

### Profile and get kernel summary in one command
```bash
nsys profile --stats=true --trace=cuda -o results ./my_app
```

### PyTorch training with NVTX auto-annotation
```bash
nsys profile --trace=cuda,nvtx --pytorch=autograd-nvtx -o pytorch_profile python train.py
```

### Only profile a specific section of code (cudaProfilerApi)
```bash
nsys profile --trace=cuda --capture-range=cudaProfilerApi -o results ./my_app
```
In code:
```c
cudaProfilerStart();
my_kernel<<<grid, block>>>(...);
cudaDeviceSynchronize();
cudaProfilerStop();
```

### MPI application
```bash
nsys profile --trace=mpi,cuda,nvtx --mpi-impl=openmpi -o mpi_%q{OMPI_COMM_WORLD_RANK} mpirun -np 4 ./app
```

### Full pipeline: profile -> export -> query
```bash
nsys profile --trace=cuda -o results ./my_app
nsys export --type=sqlite results.nsys-rep
sqlite3 results.sqlite "SELECT s.value, COUNT(*), AVG(k.end-k.start)/1000.0 AS avg_us FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id GROUP BY s.value ORDER BY avg_us DESC;"
```

## Tips

- Default traces include `opengl` and `osrt` which add noise — use `--trace=cuda,nvtx` for focused CUDA profiling.
- `--cuda-memory-usage=true` has significant overhead — only use when investigating memory leaks.
- For Python apps, always use `--target-processes=all`.
- `--stats=true` gives you a quick summary without needing separate `nsys stats` call.
- For long runs (>30s), set `--cuda-flush-interval=10000` to prevent buffer overflow.
- `nsys` gives you the timeline (what ran when); `ncu` gives you per-kernel metrics (why it's slow). Use both: nsys to find the hot kernel, ncu to analyze it.
