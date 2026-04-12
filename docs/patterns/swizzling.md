# Shared Memory Swizzling

## Why Swizzling Is Needed

Shared memory has 32 banks, 4 bytes wide. Address `A` maps to bank `(A/4) % 32`. When multiple threads in a warp access the same bank at different addresses → **bank conflict** → serialized access.

With naive row-major layouts, tensor core tiles create systematic bank conflicts because multiple rows/columns alias to the same banks. A 32-way bank conflict turns a 1-cycle access into 32 cycles.

### Traditional Fix: Padding (wasteful)
```c
__shared__ float tile[32][32+1];  // +1 padding shifts banks
```
Wastes shared memory, doesn't compose well with tensor core access patterns.

### Modern Fix: XOR-Based Swizzling (zero waste)

Applies a bitwise XOR to the offset, permuting bank assignments across rows. No wasted memory, matches ldmatrix/WGMMA access patterns exactly.

## The Mathematical Model: `Swizzle<B,M,S>`

From `cute/swizzle.hpp`:

```cpp
template <int BBits, int MBase, int SShift>
struct Swizzle;
```

Operates on the binary representation of a shared memory offset:

```
0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                               ^--^ MBase: least-sig bits kept constant
                  ^-^       ^-^     BBits: number of bits in XOR mask
                    ^---------^     SShift: distance between Y and Z fields
```

### The Three Parameters

| Parameter | Meaning | Common Values |
|---|---|---|
| **B** (BBits) | Number of XOR bits. Controls swizzle width. | 0 (none), 1 (32B), 2 (64B), 3 (128B) |
| **M** (MBase) | Low bits unchanged. Determines alignment. Max vectorized access = 2^M bytes. | 3 (8B), 4 (16B), 5 (32B) |
| **S** (SShift) | Signed shift between Y (source) and Z (target) bits. Constraint: `abs(S) >= B`. | 2, 3 |

### The Core Operation

```cpp
result = offset XOR ((offset AND yyy_mask) >> S)
```

Reads Y bits from the offset, shifts to align with Z bits, XOR's into Z bits.

**Key property**: XOR is an **involution** — applying it twice returns the original. `swizzle(swizzle(x)) = x`. This makes it trivially invertible.

## Concrete Example: `Swizzle<3,3,3>`

Common swizzle for SM80 ldmatrix-based kernels.

- B=3: 3 XOR bits → 8 different permutations
- M=3: 8-byte alignment (bits 0-2 unchanged)
- S=3: Y bits are 3 positions above Z bits

Derived masks:
- Z bits = positions [3,4,5] (bank selection)
- Y bits = positions [6,7,8] (row index)

Operation: `result[bits 3..5] = offset[bits 3..5] XOR offset[bits 6..8]`

### Before Swizzle (column access → bank conflicts)
```
Row 0 (Y=000): col 0→bank 0, col 1→bank 2, col 2→bank 4, ...
Row 1 (Y=001): col 0→bank 0, col 1→bank 2, col 2→bank 4, ...
Row 2 (Y=010): col 0→bank 0, col 1→bank 2, col 2→bank 4, ...
→ Column 0 hits bank 0 on every row → 8-way bank conflict!
```

### After `Swizzle<3,3,3>` (conflict-free)
```
Row 0 (Y=000): XOR 000 → col 0→bank 0
Row 1 (Y=001): XOR 001 → col 0→bank 1
Row 2 (Y=010): XOR 010 → col 0→bank 2
Row 3 (Y=011): XOR 011 → col 0→bank 3
Row 4 (Y=100): XOR 100 → col 0→bank 4
Row 5 (Y=101): XOR 101 → col 0→bank 5
Row 6 (Y=110): XOR 110 → col 0→bank 6
Row 7 (Y=111): XOR 111 → col 0→bank 7
→ Column 0 hits banks {0,1,2,3,4,5,6,7} → zero conflicts!
```

Each row gets a different permutation of the bank selection bits.

## Common Swizzle Configurations

### SM80 (Ampere) — ldmatrix-based kernels

```cpp
// From CUTLASS sgemm_sm80.cu
auto swizzle_atom = composition(Swizzle<3,3,3>{},
    Layout<Shape<_8, Shape<_8,_8>>, Stride<_8, Stride<_1,_64>>>{});
auto smem_layout = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
```

`Swizzle<3,3,3>`: B=3 (128B swizzle), M=3 (8B alignment for half_t).

### SM90 (Hopper) — WGMMA shared memory operands

All Hopper GMMA swizzles use **M=4, S=3** (16-byte / 128-bit alignment for uint128_t access):

| B | Name | Swizzle | Width | Layout Type |
|---|---|---|---|---|
| 0 | INTERLEAVE | `Swizzle<0,4,3>` | No swizzle | Identity |
| 1 | SW32 | `Swizzle<1,4,3>` | 32 bytes | B32 |
| 2 | SW64 | `Swizzle<2,4,3>` | 64 bytes | B64 |
| 3 | SW128 | `Swizzle<3,4,3>` | 128 bytes | B128 |

**WGMMA hardware requires M=4, S=3.** Only B varies. The GMMA descriptor encodes the layout type (INTERLEAVE/B32/B64/B128).

### SM100 (Blackwell) — tcgen05

```
SW128_32B: Swizzle<2,5,2>  — 128B base with 32B granularity
```

### Automatic Selection (CUTLASS `ss_smem_selector`)

From `cutlass/gemm/collective/builders/sm90_common.inl`:

```cpp
// Prefers largest swizzle that divides the tile dimension
if (BLK_MN % size(SW128_Atom) == 0) return SW128;
else if (BLK_MN % size(SW64_Atom) == 0) return SW64;
else if (BLK_MN % size(SW32_Atom) == 0) return SW32;
else return INTERLEAVE;
```

Rule of thumb: **use the largest B that divides your tile dimension**.

## How CuTe Composes Swizzles with Layouts

### ComposedLayout

```
ComposedLayout<Swizzle<B,M,S>, Offset, LayoutB>
```

Maps: `coord → Swizzle(Offset + LayoutB(coord))`

- **LayoutB**: defines the shape/stride of data in shared memory
- **Offset**: tracks static vs dynamic bit positions (`MixedBits`)
- **Swizzle**: applied to the index after LayoutB computes it

### Building a Swizzle Layout (SM90 example)

```cpp
// Pre-swizzle layout: arranges data with right tile structure
using PreSwizzleLayout = Layout<Shape<Shape<_32,_4>, _64>,
                                Stride<Stride<_1,_2048>, _32>>;

// Compose with swizzle
using SmemLayout = ComposedLayout<Swizzle<3,4,3>,
                                  smem_ptr_flag_bits<sizeof_bits<ElementAcc>::value>,
                                  PreSwizzleLayout>;
```

### Slicing Behavior

When you slice a swizzled layout (fix one coordinate), CuTe tries to **decay** it:
- If remaining layout hits **both Y AND Z** bits → stays `ComposedLayout`
- If hits **only one side** → becomes dynamic-normal layout (runtime strides)
- If hits **neither** → becomes static-normal layout

This matters: per-thread views of a swizzled tile often reduce to simple strided layouts.

## Interaction with Hardware Instructions

### ldmatrix (SM75+)

Each of 32 threads provides an address to a 128-bit row. Without swizzling, threads reading from rows with the same bank → conflicts. `Swizzle<3,3,3>` ensures the 32 source addresses map to different banks.

```cpp
Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom;  // ldmatrix atom
// CuTe tiles this atom to match MMA partition
// Swizzled smem layout ensures conflict-free ldmatrix
```

### WGMMA (SM90)

Operands read directly from shared memory via descriptors. The `GmmaDescriptor` encodes swizzle type:

```cpp
// From mma_traits_sm90_gmma.hpp
static_assert(M == 4 && S == 3);  // WGMMA requires these exact values
switch (B) {
    case 0: return LayoutType::INTERLEAVE;
    case 1: return LayoutType::B32;
    case 2: return LayoutType::B64;
    case 3: return LayoutType::B128;
}
```

**The hardware itself applies the swizzle when reading from shared memory.** The descriptor tells it which swizzle pattern to expect.

### stmatrix (SM90)

Stores matrix fragments from registers to shared memory in a swizzled layout. The inverse of ldmatrix.

## Quick Reference

| Architecture | Instruction | Swizzle | Use Case |
|---|---|---|---|
| SM80 (Ampere) | ldmatrix | `Swizzle<3,3,3>` | Half precision, 128-bit smem→reg |
| SM90 (Hopper) | WGMMA SS | `Swizzle<3,4,3>` (SW128) | Largest tiles |
| SM90 (Hopper) | WGMMA SS | `Swizzle<2,4,3>` (SW64) | Medium tiles |
| SM90 (Hopper) | WGMMA SS | `Swizzle<1,4,3>` (SW32) | Smaller tiles |
| SM90 (Hopper) | WGMMA SS | `Swizzle<0,4,3>` (INTER) | No swizzle |
| SM100 (Blackwell) | UMMA | `Swizzle<2,5,2>` | 128B base, 32B granularity |
