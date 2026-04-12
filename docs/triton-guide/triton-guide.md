# Triton Kernel Writing Guide

## Programming Model

Triton programs at the **block level**, not thread level. The compiler handles threads, warps, shared memory.

- Kernel = grid of **program instances** (like CUDA thread blocks, but threads are abstracted away)
- Each instance operates on a **block** (tile) of data
- `tl.program_id(axis)` identifies which program (axis: 0, 1, or 2)
- `tl.arange(start, end)` creates offset vectors (end-start must be power of 2)
- Memory via `tl.load`/`tl.store` with explicit masks for bounds

Launch:
```python
grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
```

## Decorators

### @triton.jit
Compiles Python function to GPU kernel. Parameters with `tl.constexpr` become compile-time constants (required for tensor shapes). Torch tensors auto-convert to pointers.

### @triton.autotune
Auto-searches over configurations:
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
```
- `configs`: list of Config objects with meta-params + compilation options
- `key`: argument names that trigger re-tuning when changed
- `num_warps`: parallelism within a block
- `num_stages`: software pipeline depth

## Core API Reference

### Program ID
| Function | Description |
|---|---|
| `tl.program_id(axis)` | Current program's ID (0, 1, or 2) |
| `tl.num_programs(axis)` | Total programs along axis |

### Block Init
| Function | Description |
|---|---|
| `tl.arange(start, end)` | Contiguous values [start, end). Size must be power of 2 |
| `tl.zeros(shape, dtype)` | Zero-filled tensor |
| `tl.full(shape, value, dtype)` | Filled tensor |

### Memory
**`tl.load(pointer, mask=None, other=None)`** — Three modes:
1. Scalar pointer → scalar
2. Tensor of pointers → block load. `mask=False` positions return `other`
3. Block pointer (from `make_block_ptr`) → structured load

**`tl.store(pointer, value, mask=None)`** — Same three modes.

**`tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)`** — Structured 2D access.

**`tl.advance(base, offsets)`** — Advance block pointer. **Does not mutate — returns new pointer.**

### Compute
**`tl.dot(a, b, acc=None, out_dtype=float32)`** — Matrix multiply of 2D/3D blocks.
- Inputs: int8, float8, float16, bfloat16, float32
- `acc` for accumulation: `acc = tl.dot(a, b, acc)` — critical for GEMM K-loop
- `input_precision`: "tf32" (default), "ieee"

### Elementwise
`+`, `-`, `*`, `/`, `//`, `%` work on tensors. Also:
- `tl.where(cond, x, y)` — **both x and y always evaluated**
- `tl.maximum(x, y)`, `tl.minimum(x, y)`, `tl.clamp(x, min, max)`
- `tl.exp`, `tl.exp2`, `tl.log`, `tl.log2`, `tl.sqrt`, `tl.rsqrt`
- `tl.abs`, `tl.cos`, `tl.sin`, `tl.erf`, `tl.sigmoid`
- `.to(dtype)` for type cast

### Reductions
| Function | Description |
|---|---|
| `tl.sum(x, axis)` | Sum (upcasts to float32) |
| `tl.max(x, axis)` | Maximum |
| `tl.min(x, axis)` | Minimum |
| `tl.argmax(x, axis)` | Index of max |
| `tl.softmax(x, dim)` | Fused softmax |
| `tl.cumsum(x, axis)` | Cumulative sum |
| `tl.associative_scan(x, axis, fn)` | Generic parallel scan |

### Atomics
`tl.atomic_add`, `tl.atomic_cas`, `tl.atomic_xchg`, `tl.atomic_max`, `tl.atomic_min`, `tl.atomic_and`, `tl.atomic_or`

### Compiler Hints
- `tl.multiple_of(input, values)` — values are multiples
- `tl.max_contiguous(input, values)` — contiguous in groups
- `tl.assume(cond)` — hint (e.g., `tl.assume(stride > 0)`)
- `tl.static_assert(cond)` — compile-time check
- `tl.device_print(...)` — GPU printf

### Random Numbers
`tl.rand(seed, offsets)` — uniform [0,1). `tl.randn(seed, offsets)` — normal.

## Common Patterns

### 1D Tiling
```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
data = tl.load(ptr + offsets, mask=mask)
tl.store(out_ptr + offsets, result, mask=mask)
```

### 2D Pointer Arithmetic
```python
offs_m = pid_m * BM + tl.arange(0, BM)
offs_n = pid_n * BN + tl.arange(0, BN)
ptrs = base + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
```

### GEMM Accumulation Loop
```python
acc = tl.zeros((BM, BN), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BK)):
    a = tl.load(a_ptrs, mask=..., other=0.0)
    b = tl.load(b_ptrs, mask=..., other=0.0)
    acc = tl.dot(a, b, acc)
    a_ptrs += BK * stride_ak
    b_ptrs += BK * stride_bk
result = acc.to(tl.float16)
```

### L2 Cache Optimization (Super-Grouping)
```python
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```
Improves GEMM performance by 10%+ via L2 cache reuse.

### Persistent Kernel
```python
pid = tl.program_id(0)
for tile in tl.range(pid, num_tiles, tl.num_programs(0), num_stages=N):
    # process tile
```
Fewer programs than tiles, each looping. Better for small problems.

### Online Reduction (Softmax/Attention)
```python
m_i = tl.full([BM], -float('inf'), dtype=tl.float32)
l_i = tl.full([BM], 1.0, dtype=tl.float32)
acc = tl.zeros([BM, HEAD_DIM], dtype=tl.float32)
for block in K_blocks:
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_ij)
    acc = acc * alpha[:, None]
    acc += tl.dot(p, v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_ij
acc = acc / l_i[:, None]
```

### Seeded Dropout (No Mask Storage)
```python
random = tl.rand(seed, offsets)
x_keep = random > p
output = tl.where(x_keep, x / (1 - p), 0.0)
```
Same seed + offsets = same mask. Recompute during backward.

## Important Pitfalls

1. **Block sizes must be powers of two**
2. **Always mask memory operations** — unmasked OOB = undefined behavior
3. **`other` matters for reductions**: `-float('inf')` for max, `0.0` for sum
4. **`tl.where` evaluates both branches** — use mask on load/store to avoid bad accesses
5. **`tl.advance` does not mutate** — must assign: `ptr = tl.advance(ptr, (0, BK))`
6. **Accumulate in float32** — cast to fp16/bf16 only when storing
7. **`constexpr` required for shapes** — anything in `tl.arange` or tensor dims
8. **`tl.exp` is fast but approximate** (like `__expf` in CUDA)
9. **Strides are element counts**, not bytes
10. **`num_warps` and `num_stages` interact** — always autotune

## Tutorial Files in resources/triton-3.6.0/python/tutorials/

| File | Topic |
|---|---|
| 01-vector-add.py | Basic model, 1D tiling, masking |
| 02-fused-softmax.py | Fusion, reductions, persistent kernels |
| 03-matrix-multiplication.py | 2D tiling, L2 optimization, autotune |
| 04-low-memory-dropout.py | Philox PRNG, seeded dropout |
| 05-layer-norm.py | Forward + backward, parallel reduction |
| 06-fused-attention.py | Flash Attention v2, online softmax |
| 08-grouped-gemm.py | Device-side scheduling |
| 09-persistent-matmul.py | TMA, warp specialization |
