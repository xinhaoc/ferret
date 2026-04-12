# CUDA Build Toolchain Reference (nvcc)

## Quick Start

```bash
# Compile a .cu file to executable
nvcc -o my_kernel my_kernel.cu

# Compile with GPU arch, optimization, and debug info
nvcc -arch=sm_90 -O3 -lineinfo -o my_kernel my_kernel.cu

# Compile to shared library (loadable from Python)
nvcc --shared -Xcompiler -fPIC -arch=sm_90 -O3 -o kernel.so kernel.cu

# Compile to PTX (for inspection)
nvcc --ptx -arch=sm_90 -O3 kernel.cu

# Compile to cubin (for SASS inspection)
nvcc --cubin -arch=sm_90 -O3 kernel.cu
cuobjdump --dump-sass kernel.cubin

# Print register/shared memory usage
nvcc -arch=sm_90 -O3 -Xptxas -v kernel.cu 2>&1 | grep "ptxas info"
```

## Compilation Model

NVCC compiles CUDA in two stages:

1. **Virtual**: CUDA C++ → PTX (targeting `compute_XX`)
2. **Real**: PTX → SASS/cubin (targeting `sm_XX`)

A **fat binary** bundles multiple cubins + optional PTX for JIT fallback.

## Architecture Flags

### Virtual vs Real

- **`compute_XX`** (virtual): Defines feature set. Produces PTX. Forward-compatible via JIT.
- **`sm_XX`** (real): Produces optimized SASS binary for specific GPU. NOT forward-compatible across major versions.

### `-arch` / `-gencode`

```bash
# Target specific GPU (shorthand: generates SASS + embeds PTX)
nvcc -arch=sm_90 kernel.cu
# equivalent to: -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90

# Multi-arch fat binary (production)
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_90,code=compute_90 \
     kernel.cu
# Last line embeds PTX for forward compatibility with future GPUs

# Special values
nvcc -arch=native kernel.cu    # Auto-detect current GPU
nvcc -arch=all kernel.cu       # All supported architectures
nvcc -arch=all-major kernel.cu # All major architectures
```

### Architecture-Specific Variants

`sm_XXa` / `compute_XXa` variants enable arch-specific features but lose forward compatibility:
- `sm_90a` — Hopper-specific (wgmma, TMA features)
- `sm_100a` — Blackwell-specific (tcgen05, Tensor Memory)
- `sm_100f` / `compute_100f` — Blackwell **family-specific** (features common to sm_100 + sm_103, more portable than `sm_100a`)

### Architecture Reference Table

| CC | Architecture | sm/compute | Representative GPUs |
|---|---|---|---|
| 7.0 | Volta | sm_70 | V100, TITAN V |
| 7.5 | Turing | sm_75 | T4, RTX 2080 |
| 8.0 | Ampere | sm_80 | A100, A30 |
| 8.6 | Ampere | sm_86 | RTX 3090, A40 |
| 8.9 | Ada Lovelace | sm_89 | RTX 4090, L40S, L4 |
| 9.0 | Hopper | sm_90 | H100, H200 |
| 10.0 | Blackwell | sm_100 | B200, GB200 |
| 10.3 | Blackwell | sm_103 | B300, GB300 |
| 12.0 | Blackwell (consumer) | sm_120 | RTX 5090/5080/5070 |

### CUDA Toolkit → Architecture Support

| CUDA Toolkit | Min SM | Max SM | Dropped |
|---|---|---|---|
| 11.0 - 11.7 | sm_35 | sm_86 | — |
| 11.8 | sm_35 | sm_90 | — |
| 12.0 - 12.5 | sm_50 | sm_90a | sm_35, sm_37 |
| 12.8+ | sm_50 | sm_100a | — |
| 13.0+ | sm_75 | sm_121 | sm_50-53, sm_60-62, sm_70-72 |

### `__CUDA_ARCH__` Macro

Defined only in device code. Value = `XY0` for compute capability X.Y:

```cpp
#if __CUDA_ARCH__ >= 900
    // Hopper+ code (wgmma, TMA, etc.)
#elif __CUDA_ARCH__ >= 800
    // Ampere/Ada code
#elif __CUDA_ARCH__ >= 700
    // Volta/Turing code
#endif
```

## Optimization Flags

| Flag | Description |
|---|---|
| `-O0` | No optimization |
| `-O2` | Standard optimization |
| `-O3` | Full optimization |
| `--use_fast_math` | Fast math intrinsics (see below) |
| `--maxrregcount=N` | Limit registers per thread (increases occupancy, may spill) |
| `--restrict` | Assert all kernel pointer params are non-aliasing |
| `--extra-device-vectorization` | More aggressive vectorization |
| `-Xptxas -O3` | Max PTX assembler optimization |
| `-Ofc max` | Fastest compile, least optimized device code |
| `-dlto` | Device link-time optimization |

### `--use_fast_math` Details

Equivalent to: `--ftz=true --prec-div=false --prec-sqrt=false --fmad=true`

| Flag | Default | Effect when changed |
|---|---|---|
| `--ftz=true` | false | Flush denormals to zero (faster) |
| `--prec-div=false` | true | Less precise division (faster) |
| `--prec-sqrt=false` | true | Less precise sqrt (faster) |
| `--fmad=true` | true | Contract multiply-add into FMA |

**Use for**: compute-bound kernels where ~1e-6 accuracy loss is OK.
**Avoid for**: numerical algorithms requiring IEEE compliance.

## Debug Flags

| Flag | Description |
|---|---|
| `-g` | Host debug info |
| `-G` | Device debug info (**disables all device optimizations**, 10-100x slower) |
| `-lineinfo` | Source line info for profiling (**no performance impact, always use this**) |
| `-Xptxas -v` | Print register/shared memory usage per kernel |
| `-dopt=on` | Enable partial optimization even with `-G` |
| `-res-usage` | Display per-kernel resource usage |

**Never use `-G` for profiling** — it disables optimizations. Use `-lineinfo` instead.

## Passing Flags to Sub-compilers

| Flag | Passes to | Example |
|---|---|---|
| `-Xcompiler` | Host compiler (gcc/clang) | `-Xcompiler "-fPIC,-Wall,-O3"` |
| `-Xptxas` | PTX assembler | `-Xptxas -v` (print register usage) |
| `-Xlinker` | Host linker | `-Xlinker "-rpath,/usr/local/lib"` |
| `-Xnvlink` | Device linker | `-Xnvlink --verbose` |

Syntax: single option `-Xcompiler -fPIC`, or comma-separated `-Xcompiler -fPIC,-Wall,-O3`.

## Output Phase Control

| Flag | Output | Description |
|---|---|---|
| (default) | executable | Compile + link |
| `-c` | `.o` | Compile to object file |
| `--ptx` | `.ptx` | PTX assembly only |
| `--cubin` | `.cubin` | SASS binary only |
| `--fatbin` | `.fatbin` | Fat binary |
| `--shared` | `.so` / `.dll` | Shared library |
| `-dc` | `.o` | Object with relocatable device code (separate compilation) |
| `-dlink` | `_dlink.o` | Device-link relocatable objects |

## Separate Compilation (multi-file projects)

Default is whole-program compilation. For multi-file device code:

```bash
# Compile each .cu to relocatable object
nvcc -dc -arch=sm_90 -O3 file1.cu -o file1.o
nvcc -dc -arch=sm_90 -O3 file2.cu -o file2.o

# Link everything
nvcc -arch=sm_90 file1.o file2.o -o program
```

Use when `__device__` functions or `extern __device__` variables cross file boundaries.

## C++ Standard & Language

| Flag | Description |
|---|---|
| `--std=c++17` | C++ standard (also: c++11, c++14, c++20) |
| `--extended-lambda` | Allow `__host__`/`__device__` on lambdas |
| `--expt-relaxed-constexpr` | constexpr callable from both host and device |

## Common Recipes

### Production multi-arch build
```bash
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_90,code=compute_90 \
     -O3 -lineinfo kernel.cu -o program
```

### Shared library for Python ctypes
```bash
nvcc --shared -Xcompiler -fPIC -arch=sm_90 -O3 -lineinfo -o kernel.so kernel.cu
```

In the .cu file:
```cpp
extern "C" void launch_kernel(float* in, float* out, int n) {
    my_kernel<<<(n+255)/256, 256>>>(in, out, n);
}
```

### Using torch cpp_extension (recommended for Python)
```python
from torch.utils.cpp_extension import load_inline

cuda_source = """
__global__ void my_kernel(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;
}

torch::Tensor my_op(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    my_kernel<<<(n+255)/256, 256>>>(
        output.data_ptr<float>(), input.data_ptr<float>(), n);
    return output;
}
"""

module = load_inline(
    name="my_kernel",
    cpp_sources="torch::Tensor my_op(torch::Tensor input);",
    cuda_sources=cuda_source,
    functions=["my_op"],
    extra_cuda_cflags=["-O3", "-lineinfo"],
)
output = module.my_op(input_tensor)
```

### LTO (Link-Time Optimization)
```bash
nvcc -dc -arch=sm_90 -dlto file1.cu -o file1.o
nvcc -dc -arch=sm_90 -dlto file2.cu -o file2.o
nvcc -arch=sm_90 -dlto file1.o file2.o -o program
```

### Inspect PTX with source lines
```bash
nvcc --ptx -arch=compute_90 -lineinfo -src-in-ptx kernel.cu
```

### Inspect SASS
```bash
nvcc --cubin -arch=sm_90 -O3 kernel.cu
cuobjdump --dump-sass kernel.cubin
```

### Maximum performance
```bash
nvcc -O3 --use_fast_math -Xptxas -O3 -maxrregcount=128 \
     -arch=sm_90 -lineinfo kernel.cu -o program
```

### Per-thread default stream
```bash
nvcc -arch=sm_90 --default-stream per-thread kernel.cu -o program
```

## CMake Integration

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_kernel CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES "80;90")

add_executable(my_kernel kernel.cu)

target_compile_options(my_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3 -lineinfo --use_fast_math -Xptxas -v
    >
)

set_target_properties(my_kernel PROPERTIES
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
)
```

## Predefined Macros

| Macro | When Defined |
|---|---|
| `__NVCC__` | Always (any nvcc compilation) |
| `__CUDACC__` | When compiling .cu files |
| `__CUDA_ARCH__` | Device code only. Value = `XY0` for CC X.Y |
| `__CUDACC_VER_MAJOR__` | CUDA compiler major version |
| `__CUDACC_VER_MINOR__` | CUDA compiler minor version |
| `__CUDACC_RDC__` | When `-rdc=true` (separate compilation) |
| `__CUDACC_DEBUG__` | When `-G` is active |

## Tips

- Always use `-lineinfo` — zero runtime cost, essential for profiling.
- Never use `-G` for performance measurement — it disables all device optimizations.
- `--use_fast_math` can give 10-30% speedup on compute-bound kernels.
- `-Xptxas -v` shows register/smem per kernel — critical for occupancy tuning.
- `--maxrregcount=N` trades register spilling for higher occupancy.
- Include `code=compute_XX` in your highest `-gencode` for forward compatibility.
- Use `-arch=native` during development for fastest compilation.
- Use `CUDA_FORCE_PTX_JIT=1` env var to test PTX forward compatibility.
