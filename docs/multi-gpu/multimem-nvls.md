# Multimem and NVLS Reference

## What Multimem Is

A **multimem address** points to multiple memory locations across GPUs simultaneously. `multimem.*` PTX instructions access all locations in a single operation, leveraging **NVLS (NVLink SHARP)** for in-network computation in the NVSwitch fabric.

Only `multimem.*` instructions can access multimem addresses. Regular `ld`/`st` on them = **undefined behavior**.

## Why It's Fast

- **In-network reduction**: `multimem.ld_reduce` reads from ALL GPUs and reduces in the NVSwitch, returning the result directly — no intermediate stores
- **In-network broadcast**: `multimem.st` writes once, NVSwitch distributes to all GPUs
- **8x traffic reduction** vs naive approaches (N loads → 1 multimem instruction)
- Requires **sm_90** (Hopper) or higher

## PTX Instructions

### `multimem.ld_reduce` — Load-and-Reduce from All GPUs

```
multimem.ld_reduce{.sem}{.scope}{.ss}.op.type  d, [a];
```

Loads from ALL memory locations at the multimem address, reduces them (e.g., add, min, max), returns result in `d`.

```
// Integer: AND reduction across all GPUs
multimem.ld_reduce.and.b32  val, [addr];

// Float: ADD with FP32 accumulation across GPUs
multimem.ld_reduce.acquire.gpu.global.add.acc::f32.v2.f16x2  {v0, v1}, [addr];

// FP8 with FP16 accumulation (Blackwell sm_100a+)
multimem.ld_reduce.add.acc::f16.v4.e4m3  {v0, v1, v2, v3}, [addr];
```

### `multimem.st` — Broadcast Store to All GPUs

```
multimem.st{.sem}{.scope}{.ss}.type  [a], b;
```

Stores to ALL memory locations at the multimem address. NVSwitch broadcasts.

```
multimem.st.relaxed.gpu.b32  [addr], val;
```

### `multimem.red` — Reduction Operation on All GPUs

```
multimem.red{.sem}{.scope}{.ss}.op.type  [a], b;
```

Applies operand `b` as a reduction to each remote location.

```
// Reduce-add 4x f32 across all GPUs
multimem.red.release.cta.global.add.v4.f32  [addr], {v0, v1, v2, v3};
```

### `multimem.cp.async.bulk` — Async Bulk Copy to All GPUs

```
multimem.cp.async.bulk.global.shared::cta.bulk_group  [dstMem], [srcMem], size;
```

Async broadcast from shared memory to ALL GPUs. Size must be multiple of 16, 16-byte aligned. Uses bulk async-group completion.

Optional `.cp_mask` (sm_100+) enables selective byte copying via 16-bit mask.

### `multimem.cp.reduce.async.bulk` — Async Bulk Reduction

```
multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32  [dstMem], [srcMem], size;
```

Each element at the multimem destination is reduced with the source element from shared memory. Supported ops: add, and, or, xor, inc, dec, min, max.

## Supported Operations and Types

| Operation | Integer types | Float types |
|---|---|---|
| `.add` | u32, u64, s32, s64 | f16, f16x2, bf16, bf16x2, f32, f64, e4m3, e5m2 |
| `.min` | u32, u64, s32, s64 | f16, f16x2, bf16, bf16x2, f32, f64 |
| `.max` | u32, u64, s32, s64 | f16, f16x2, bf16, bf16x2, f32, f64 |
| `.and` | b32, b64 | — |
| `.or` | b32, b64 | — |
| `.xor` | b32, b64 | — |

### Accumulator Precision

For FP16/BF16/FP8 reductions, specify accumulator precision:
```
multimem.ld_reduce.add.acc::f32.v2.f16x2  {v0, v1}, [addr];    // FP32 accumulation
multimem.ld_reduce.add.acc::f16.v4.e4m3   {v0, v1, v2, v3}, [addr];  // FP16 (sm_100a+)
```

### Vector Width

Total bitwidth of `.vec` × `.type` must be 32, 64, or 128 bits. `.f64` cannot use `.vec`.

| .vec | Supported base types |
|---|---|
| none | f16x2, bf16x2, f32, f64, e5m2x4, e4m3x4 |
| .v2 | f16, f16x2, bf16, bf16x2, f32, e5m2x2, e4m3x2 |
| .v4 | f16, f16x2, bf16, bf16x2, f32, e5m2, e4m3 |
| .v8 | f16, bf16, e5m2, e4m3 |

## Multicast Address Setup

### With CUDA Driver API (low-level)

```c
// 1. Allocate physical memory on each GPU
cuMemCreate(&uc_handle, size, &prop, 0);

// 2. Create multicast object (rank 0)
cuMulticastCreate(&mc_handle, &mcprop);

// 3. Export/import handle across ranks
cuMemExportToShareableHandle(&shareableHandle, mc_handle, ...);
// ... distribute via IPC ...
cuMemImportFromShareableHandle(&mc_handle, shareableHandle, ...);

// 4. Add each device to the multicast group
cuMulticastAddDevice(mc_handle, device);

// 5. Bind physical memory
cuMulticastBindMem(mc_handle, 0, uc_handle, 0, size, 0);

// 6. Map multicast virtual address
cuMemAddressReserve(&mc_va, size, 0, 0, 0);
cuMemMap(mc_va, size, 0, mc_handle, 0);
cuMemSetAccess(mc_va, size, &accessDesc, 1);
// mc_va is now the multimem address for multimem.* instructions
```

### With NVSHMEM (simplified)

```python
local_tensor = nvshmem.core.tensor((M, N), dtype=torch.float32)
mc_tensor = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, local_tensor)
```

---

## TMA Multicast (Cluster-Level)

TMA supports multicast for copying tensor data from global memory to shared memory of **multiple CTAs in a cluster**.

```
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster
    [dstMem], [tensorMap, {c0, c1}], [mbar], ctaMask;
```

- `ctaMask`: 16-bit mask, each bit = one CTA's `%cluster_ctarank`
- Source data multicast to same offset in each destination CTA's shared memory
- mbarrier signal also multicast
- Requires sm_90a (Hopper), sm_100a/f (Blackwell)

**Use case**: Efficient allgather/broadcast within thread block clusters.

---

## Practical: Two-Shot All-Reduce with Multimem

From `resources/cutlass-4.4.2/examples/python/CuTeDSL/distributed/all_reduce_two_shot_multimem.py`:

1. Divide data into N chunks (one per GPU)
2. **Reduce phase**: Each GPU calls `multimem.ld_reduce.add` on its chunk → NVSwitch returns sum across all GPUs
3. **Broadcast phase**: Each GPU calls `multimem.st` to write its chunk → NVSwitch broadcasts to all GPUs
4. Synchronization via `multimem.red.release.sys.add` on flag variables + spin-lock wait

Result: **8x NVLink traffic reduction** vs naive approaches.
