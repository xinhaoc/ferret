# cuda-gdb Cheat Sheet

## Setup

```bash
# Compile with debug info (device debug disables optimizations)
nvcc -g -G -O0 -arch=sm_90 kernel.cu -o kernel_debug

# Compile with limited debug (some optimization preserved)
nvcc -g -G -dopt=on -arch=sm_90 kernel.cu -o kernel_debug

# Launch cuda-gdb
cuda-gdb ./kernel_debug
```

## Essential Commands

### Breakpoints

```
# Break at a CUDA kernel
(cuda-gdb) break my_kernel
(cuda-gdb) break kernel.cu:42

# Conditional breakpoint
(cuda-gdb) break my_kernel if threadIdx.x == 0 && blockIdx.x == 0

# Break on CUDA API error
(cuda-gdb) set cuda api_failures stop

# List breakpoints
(cuda-gdb) info breakpoints
```

### Running

```
(cuda-gdb) run
(cuda-gdb) run arg1 arg2
(cuda-gdb) continue          # resume after breakpoint
(cuda-gdb) next              # step over (host or device)
(cuda-gdb) step              # step into
(cuda-gdb) finish            # run until current function returns
```

### Inspecting GPU State

```
# Show current CUDA focus (which thread/block/warp you're looking at)
(cuda-gdb) cuda thread
(cuda-gdb) cuda block
(cuda-gdb) cuda warp
(cuda-gdb) cuda kernel

# Switch focus to specific thread
(cuda-gdb) cuda thread (0,0,0)                    # threadIdx
(cuda-gdb) cuda block (3,0,0)                     # blockIdx
(cuda-gdb) cuda block (3,0,0) thread (32,0,0)     # both
(cuda-gdb) cuda kernel 1 block (0,0,0) thread (0,0,0)

# Show all active kernels
(cuda-gdb) info cuda kernels

# Show all active blocks
(cuda-gdb) info cuda blocks

# Show all active threads
(cuda-gdb) info cuda threads

# Show active warps in current block
(cuda-gdb) info cuda warps

# Show lanes in current warp
(cuda-gdb) info cuda lanes
```

### Printing Variables

```
# Print variable (current thread)
(cuda-gdb) print myVar
(cuda-gdb) print shared_array[threadIdx.x]
(cuda-gdb) print *d_ptr@10          # print 10 elements from pointer

# Print for specific thread
(cuda-gdb) cuda thread (5,0,0)
(cuda-gdb) print threadIdx.x        # prints 5
(cuda-gdb) print localVar

# Print register
(cuda-gdb) print $R0

# Print across all threads in a warp
(cuda-gdb) print/a *(float*)&shared_mem[0]@32
```

### Memory Inspection

```
# Examine device memory
(cuda-gdb) x/10f d_output           # 10 floats from device pointer
(cuda-gdb) x/32xw 0x7f1234560000    # 32 hex words from address

# Print shared memory
(cuda-gdb) print shared_data[0]@16

# Check pointer type
(cuda-gdb) print (void*)d_ptr
```

### Watchpoints

```
# Watch shared memory location (slow but powerful)
(cuda-gdb) watch shared_array[0]

# Watch for specific thread only
(cuda-gdb) cuda thread (0,0,0)
(cuda-gdb) watch myVar
```

### Backtrace

```
(cuda-gdb) backtrace              # call stack
(cuda-gdb) frame 2                # switch to frame 2
(cuda-gdb) info locals            # local variables in current frame
```

## Common Debugging Scenarios

### Find which thread caused an error
```
(cuda-gdb) set cuda api_failures stop
(cuda-gdb) run
# When it stops at error:
(cuda-gdb) cuda thread            # shows offending thread
(cuda-gdb) backtrace
(cuda-gdb) info locals
```

### Debug out-of-bounds access
```bash
# Better: use compute-sanitizer instead
compute-sanitizer --tool memcheck ./kernel_debug
```

### Debug race condition
```
# Set breakpoint where race is suspected
(cuda-gdb) break kernel.cu:55
(cuda-gdb) run
# Inspect values across threads
(cuda-gdb) cuda thread (0,0,0)
(cuda-gdb) print shared_val
(cuda-gdb) cuda thread (1,0,0)
(cuda-gdb) print shared_val
```

### Attach to running process
```
(cuda-gdb) attach <pid>
(cuda-gdb) info cuda kernels
```

## Tips

- `-G` makes kernels 10-100x slower. Only use for debugging, never for profiling.
- `cuda-gdb` can only break inside device code compiled with `-G`.
- Use `-dopt=on` with `-G` for a compromise between debuggability and performance.
- `info cuda kernels` shows all active GPU kernels — useful for multi-kernel debugging.
- Prefer `compute-sanitizer` for memory errors and race conditions — it's faster and more thorough.
- cuda-gdb supports Python scripting for automation: `(cuda-gdb) python print("hello")`
- Environment: `CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1` enables preemption-based debugging (slower but more reliable).
