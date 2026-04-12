# Inline PTX Assembly Guide

## Syntax

```cuda
asm [volatile] ( "template-string"
    : output-operands      // "constraint"(variable), ...
    : input-operands       // "constraint"(variable), ...
    : clobber-list          // "memory", ...
);
```

All colons are positional. Omitted sections still need colons if a later section is present.

```cuda
asm("mov.s32 %0, 2;" : "=r"(i));                    // output only
asm("mov.s32 r1, %0;" :: "r"(i));                    // input only
asm volatile("membar.gl;" ::: "memory");              // no operands, memory clobber
asm volatile("bar.sync %0, %1;" : : "r"(id), "r"(n) : "memory");  // inputs + clobber
```

Operand references: `%0` = first operand (outputs numbered first, then inputs). `%%` escapes literal `%` for special registers.

```cuda
asm volatile("mov.u32 %0, %%clock;" : "=r"(x));
asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank) :);
```

## Constraint Letters

| Constraint | PTX Type | C++ Type | Size | Description |
|---|---|---|---|---|
| `r` | `.u32` | `uint32_t`, `int` | 32-bit | Word register (most common) |
| `l` | `.u64` | `uint64_t`, `void*` | 64-bit | Long register, pointers |
| `f` | `.f32` | `float` | 32-bit | Single-precision float |
| `d` | `.f64` | `double` | 64-bit | Double-precision float |
| `h` | `.u16` | `uint16_t` | 16-bit | Half-word register |
| `n` | — | compile-time int | — | Immediate constant (must be known at compile time) |
| `C` | — | `const char[]` | — | Compile-time string substitution |

### Output Modifiers

| Modifier | Meaning |
|---|---|
| `=` | Write-only output: `"=r"(result)` |
| `+` | Read-write (both input AND output): `"+r"(accumulator)` |

### Type Matching Rules

The C++ variable MUST match the constraint's size:
- `"r"` → `uint32_t`, `int32_t`, `int`
- `"l"` → `uint64_t`, pointers
- `"f"` → `float`
- `"d"` → `double`
- `"h"` → `uint16_t`

Using the wrong type is a compile error. One constraint per operand only (no `"rf"`). Only scalar types (no `int4`, no arrays).

## volatile and "memory" Clobber

### When to Use `asm volatile`

Always use when the instruction has side effects beyond its outputs:
- Memory stores, barriers, synchronization
- Reading hardware state (clocks, special registers)
- Must not be deleted, moved, or reordered

Without `volatile`, the compiler may optimize away or reorder the asm.

### The "memory" Clobber

Tells compiler that asm reads/writes memory beyond operands. Prevents reordering across this point.

```cuda
asm volatile("st.shared.u32 [%0], %1;\n" : : "r"(ptr), "r"(val) : "memory");
```

Triple-colon shorthand `:::` = "no outputs, no inputs, clobbers follow":
```cuda
asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
```

### Compiler Fence Pattern (Empty asm)

Prevents compiler from reordering register/memory accesses. Generates no instructions:
```cuda
asm volatile("" : "+r"(reg) :: "memory");   // for uint32_t
asm volatile("" : "+f"(reg) :: "memory");   // for float
```

## Scoping and .reg Declarations

**Always wrap multi-instruction blocks in `{ }` to prevent register name collisions when inlined:**

```cuda
// CORRECT — braces create local scope:
asm("{\n\t"
    " .reg .u32 t1;\n\t"
    " mul.lo.u32 t1, %1, %1;\n\t"
    " mul.lo.u32 %0, t1, %1;\n\t"
    "}"
    : "=r"(y) : "r"(x));
```

Common .reg declarations:
```
.reg .pred p;       // predicate register
.reg .b32 tmp;      // 32-bit temporary
.reg .u64 addr;     // 64-bit temporary
.reg .f16x2 val;    // packed f16x2
```

## Predicates

PTX uses predicate registers (`.pred`) for conditional execution. C++ bools can't be passed directly — convert inside the asm block.

### Pattern: Convert int → predicate, conditional execute

```cuda
asm volatile(
    "{\n"
    "  .reg .pred p;\n"
    "  setp.ne.b32 p, %5, 0;\n"           // p = (pred_guard != 0)
    "  mov.b32 %0, %6;\n"                  // default value
    "  mov.b32 %1, %7;\n"
    "  mov.b32 %2, %8;\n"
    "  mov.b32 %3, %9;\n"
    "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"  // conditional load
    "}\n"
    : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
    : "l"(ptr), "r"((int)pred_guard),
      "r"(d.x), "r"(d.y), "r"(d.z), "r"(d.w));
```

### Pattern: Extract predicate result back to int

```cuda
asm volatile(
    "{\n\t"
    ".reg .pred P1;\n\t"
    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
    "selp.b32 %0, 1, 0, P1;\n\t"          // convert pred → int
    "}"
    : "=r"(waitComplete)
    : "r"(smem_addr), "r"(phase)
    : "memory");
```

## Real-World Examples

### cp.async (SM80 — Async Global→Shared)

```cuda
// 16-byte async copy with L2 cache hint
asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
    :: "r"(smem_int_ptr), "l"(gmem_ptr), "n"(sizeof(TS)));

// With predication (CUTLASS pattern)
asm volatile(
    "{\n"
    "  .reg .pred p;\n"
    "  setp.ne.b32 p, %0, 0;\n"
    "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
    "}\n"
    :: "r"((int)pred_guard), "r"(smem_int_ptr), "l"(global_ptr), "n"(SizeInBytes));

// Commit and wait
asm volatile("cp.async.commit_group;\n" ::);
asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
```

### TMA Loads (SM90 — Tensor Memory Accelerator)

```cuda
// TMA load 2D: global → shared, with mbarrier tracking + L2 hint
asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
    " [%0], [%1, {%3, %4}], [%2], %5;"
    :
    : "r"(smem_int_ptr),     // shared memory dest (uint32_t)
      "l"(gmem_int_desc),    // TMA descriptor (uint64_t)
      "r"(smem_int_mbar),    // mbarrier address (uint32_t)
      "r"(crd0), "r"(crd1),  // coordinates
      "l"(cache_hint)        // L2 hint (uint64_t)
    : "memory");

// TMA with multicast (note "h" constraint for uint16_t mask)
asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
    " [%0], [%1, {%4, %5}], [%2], %3, %6;"
    :
    : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
      "h"(multicast_mask), "r"(crd0), "r"(crd1), "l"(cache_hint)
    : "memory");
```

### TMA Stores (SM90 — Shared→Global)

```cuda
// Step 1: Fence shared memory
asm volatile("fence.proxy.async.shared::cta;");

// Step 2: Issue TMA store
asm volatile(
    "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
    : : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1) : "memory");

// Step 3: Commit and wait
asm volatile("cp.async.bulk.commit_group;");
asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(0) : "memory");
```

### WGMMA (SM90 — Warpgroup MMA)

```cuda
// Fence → MMA → Commit → Wait sequence
asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, %4, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16 "
    "{%0, %1}, %2, %3, p, %5, %6, %7, %8;\n"
    "}\n"
    : "+r"(d0), "+r"(d1)
    : "l"(desc_a), "l"(desc_b),
      "r"(int32_t(scale_D)),
      "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
      "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));

asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
```

### MMA (SM80 — Tensor Core)

```cuda
// mma.sync 16x8x16: F32 += F16 * F16
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3));
```

### Barrier Operations

```cuda
// Named barrier
asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads) : "memory");

// Cluster barrier
asm volatile("barrier.cluster.arrive.aligned;\n" : : );
asm volatile("barrier.cluster.wait.aligned;\n" : : );

// mbarrier init
asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;"
    : : "r"(arrive_count), "r"(smem_addr) : "memory");

// mbarrier arrive with expected transaction bytes
asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;"
    : : "r"(transaction_bytes), "r"(smem_addr) : "memory");

// mbarrier try_wait with spin loop
asm volatile(
    "{\n\t"
    ".reg .pred P1;\n\t"
    "LAB_WAIT:\n\t"
    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n\t"
    "@P1 bra DONE;\n\t"
    "bra LAB_WAIT;\n\t"
    "DONE:\n\t"
    "}"
    : : "r"(smem_addr), "r"(phase), "r"(ticks) : "memory");

// Remote mbarrier arrive (cross-CTA in cluster)
asm volatile(
    "{\n\t"
    ".reg .b32 remAddr32;\n\t"
    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
    "mbarrier.arrive.shared::cluster.b64 _, [remAddr32];\n\t"
    "}"
    : : "r"(smem_addr), "r"(cta_id) : "memory");
```

### Shared Memory Load/Store

```cuda
// 32-bit load/store
asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(val) : "r"(ptr));
asm volatile("st.shared.u32 [%0], %1;\n" : : "r"(ptr), "r"(val) : "memory");

// 128-bit vector load/store
asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w) : "r"(ptr));
asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
    : : "r"(ptr), "r"(s.x), "r"(s.y), "r"(s.z), "r"(s.w) : "memory");
```

### Register Reconfig (SM90 — Warp Specialization)

```cuda
asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
```

### Fences

```cuda
asm volatile("fence.proxy.async.shared::cta;");       // before TMA store
asm volatile("fence.proxy.async.global;");             // global async fence
asm volatile("fence.mbarrier_init.release.cluster;" :: : "memory");
```

## Shared Memory Pointer Conversion

PTX shared memory instructions expect a 32-bit address with `"r"` constraint. Convert generic pointers:

```cuda
uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
// Or use CUTLASS: cute::cast_smem_ptr_to_uint(smem_ptr)
```

## Pitfalls

1. **Errors appear at ptxas stage, not compile time** — the CUDA compiler does NOT parse the template string. Type mismatches and bad instructions only surface when ptxas assembles the PTX.

2. **Memory space must be explicit** — use `ld.global`, `ld.shared`, `ld.local` correctly. Wrong space = undefined behavior.

3. **Always use `{ }` braces** around multi-instruction blocks to avoid register name collisions when inlined.

4. **Match constraint sizes** — `"r"` with `uint32_t`, `"l"` with `uint64_t`/pointers, `"f"` with `float`. Mismatches are silent errors.

5. **`"n"` must be compile-time known** — template parameters and `sizeof()` work, runtime variables don't.

6. **Outputs are numbered before inputs** — `%0` is the first output operand, inputs continue the count.
