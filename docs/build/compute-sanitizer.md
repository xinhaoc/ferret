# compute-sanitizer Reference

NVIDIA's runtime error checker for CUDA. Catches memory errors, race conditions, uninitialized memory, and synchronization bugs without recompiling (but `-lineinfo` helps with source locations).

## Quick Start

```bash
# Memory error checking (default)
compute-sanitizer ./my_kernel

# Race condition detection
compute-sanitizer --tool racecheck ./my_kernel

# Uninitialized memory
compute-sanitizer --tool initcheck ./my_kernel

# Synchronization errors
compute-sanitizer --tool synccheck ./my_kernel

# With source line info (compile with -lineinfo)
nvcc -lineinfo -arch=sm_90 -O3 kernel.cu -o kernel
compute-sanitizer --tool memcheck ./kernel
```

## Tools

### memcheck (default) — Memory Errors

Catches:
- **Out-of-bounds access**: global, shared, local memory
- **Misaligned access**: address not aligned to access size
- **Invalid address**: accessing freed or unmapped memory
- **Stack overflow**: exceeding device stack limit
- **Illegal instruction**: executing undefined opcodes
- **Hardware exceptions**: ECC errors, bus errors

```bash
# Basic
compute-sanitizer ./kernel

# Also check host-side CUDA API memory errors
compute-sanitizer --check-api-memory-access yes ./kernel

# Track device memory leaks
compute-sanitizer --leak-check full ./kernel

# Report error details
compute-sanitizer --print-level info ./kernel
```

Output example:
```
========= Invalid __global__ write of size 4 bytes
=========     at 0x170 in my_kernel(float*, int)
=========     by thread (256,0,0) in block (0,0,0)
=========     Address 0x7f1234560400 is out of bounds
=========     Saved host backtrace up to driver entry point
```

### racecheck — Shared Memory Race Conditions

Catches:
- **Write-after-write (WAW)** hazards on shared memory
- **Write-after-read (WAR)** hazards on shared memory
- **Read-after-write (RAW)** hazards on shared memory
- Missing `__syncthreads()` between dependent accesses

```bash
compute-sanitizer --tool racecheck ./kernel

# Only report specific hazard types
compute-sanitizer --tool racecheck --racecheck-report hazard ./kernel
compute-sanitizer --tool racecheck --racecheck-report analysis ./kernel
```

Output example:
```
========= WARN: Potential WAR hazard detected at __shared__ 0x0 in block (0,0,0):
=========     Write Thread (1,0,0) at 0x250 in kernel.cu:42
=========     Read Thread (0,0,0) at 0x230 in kernel.cu:40
=========     Current Value : 0x41200000
```

**Note**: racecheck only checks shared memory, not global memory races. For global memory atomicity, use correct atomic operations.

### initcheck — Uninitialized Device Memory

Catches:
- Reading device global memory that was allocated but never written
- Reading shared memory before initialization
- Using values from `cudaMalloc` without `cudaMemset` or kernel write

```bash
compute-sanitizer --tool initcheck ./kernel

# Track specific memory types
compute-sanitizer --tool initcheck --track-unused-memory yes ./kernel
```

### synccheck — Synchronization Errors

Catches:
- Calling `__syncthreads()` in divergent code (not all threads in block reach the barrier)
- Invalid use of cooperative group synchronization
- Barrier misuse in warp-level synchronization

```bash
compute-sanitizer --tool synccheck ./kernel
```

Output example:
```
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x380 in my_kernel(float*, int)
=========     by thread (31,0,0) in block (2,0,0)
```

## Key Flags

| Flag | Description |
|---|---|
| `--tool <name>` | Select tool: `memcheck` (default), `racecheck`, `initcheck`, `synccheck` |
| `--print-level <level>` | Output detail: `info`, `warn`, `error`, `fatal` |
| `--log-file <path>` | Save output to file |
| `--save <path>` | Save results to binary file (for later analysis) |
| `--error-exitcode <N>` | Exit with code N if errors found (for CI) |
| `--check-api-memory-access yes` | Also check host CUDA API memory errors |
| `--leak-check full` | Report device memory leaks |
| `--report-api-errors all` | Report all CUDA API errors |
| `--destroy-on-device-error kernel` | Terminate kernel on error (vs continue) |
| `--target-processes all` | Check child processes too |
| `--launch-timeout <sec>` | Timeout for kernel launches |
| `--kernel-regex <pattern>` | Only check kernels matching pattern |
| `--kernel-regex-exclude <pattern>` | Exclude kernels matching pattern |

## Common Recipes

### CI integration (exit non-zero on errors)
```bash
compute-sanitizer --error-exitcode 1 ./kernel
echo $?  # 0 = clean, 1 = errors found
```

### Check a specific kernel only
```bash
compute-sanitizer --kernel-regex "my_kernel" ./app
```

### Full check: all tools
```bash
compute-sanitizer --tool memcheck ./kernel && \
compute-sanitizer --tool racecheck ./kernel && \
compute-sanitizer --tool initcheck ./kernel && \
compute-sanitizer --tool synccheck ./kernel
```

### Save and replay results
```bash
compute-sanitizer --save results.dat ./kernel
compute-sanitizer --read results.dat  # analyze offline
```

### With Python
```bash
compute-sanitizer python my_cuda_script.py
compute-sanitizer --tool racecheck python my_cuda_script.py
```

## Performance Impact

| Tool | Slowdown | Notes |
|---|---|---|
| memcheck | 5-20× | Checks every memory access |
| racecheck | 20-100× | Tracks all shared memory accesses per-thread |
| initcheck | 5-20× | Shadows all allocations |
| synccheck | 2-5× | Lightweight barrier checking |

All tools work on optimized code (no `-G` needed). Use `-lineinfo` for source locations.

## Tips

- Always compile with `-lineinfo` for source file:line references in errors.
- `memcheck` is the most common tool — run it first.
- `racecheck` only checks shared memory. Global memory races need careful atomic usage.
- `--error-exitcode 1` makes it CI-friendly.
- For Python/PyTorch: `compute-sanitizer python script.py` works directly.
- Very slow? Use `--kernel-regex` to limit checking to your kernel only.
- False positives in racecheck are possible — verify with careful code review.
