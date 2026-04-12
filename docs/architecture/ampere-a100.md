# Ampere A100 Architecture Deep-Dive

## SM Architecture

- **SM count**: 108 SMs (A100 SXM/PCIe). Full GA100 die: 128 SMs. 7 GPCs, 7-8 TPCs/GPC, 2 SMs/TPC, up to 16 SMs/GPC.
- **Per SM**: 64 FP32 CUDA cores (6912 total), 4 third-gen Tensor Cores (432 total)
- **CC 8.0** (A100): Max 64 warps/SM, 2048 threads/SM, 32 blocks/SM
- **CC 8.6** (RTX 3090, A40): Max 48 warps/SM, 1536 threads/SM, 16 blocks/SM. 128 FP32 cores/SM (doubled via concurrent INT32+FP32 pipeline).
- **Register file**: 256 KB (64K × 32-bit registers), max 255 regs/thread
- **Shared memory + L1**: 192 KB combined (1.5x V100's 128 KB)
  - CC 8.0: configurable up to 164 KB. Carveout options: 0, 8, 16, 32, 64, 100, 132, 164 KB
  - CC 8.6: configurable up to 100 KB. Carveout options: 0, 8, 16, 32, 64, 100 KB (out of 128 KB combined)
  - Static shared memory limited to 48 KB; dynamic above 48 KB requires `cudaFuncSetAttribute` opt-in
- **Compute capability**: 8.0 (A100, A30), 8.6 (RTX 3090, A40), 8.7 (Jetson AGX Orin), 8.9 (RTX 4090, L40S — Ada Lovelace, uses same Ampere SM class)

## Memory Hierarchy

### L1 Cache
- Unified with shared memory and texture cache in the 192 KB (CC 8.0) or 128 KB (CC 8.6) combined structure
- Whatever remains after shared memory carveout serves as L1

### L2 Cache
- **40 MB** (7x larger than V100's ~6 MB). Divided into two partitions for higher bandwidth and lower latency.
- **2.3x the L2 read bandwidth** of V100
- **Programmable residency control**: pin frequently accessed data in a set-aside portion of L2
  ```cpp
  cudaStreamAttrValue attr;
  attr.accessPolicyWindow.base_ptr = data;
  attr.accessPolicyWindow.num_bytes = min(size, l2_size / 2);
  attr.accessPolicyWindow.hitRatio = 1.0f;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
  ```
- **Compute Data Compression**: up to 4x improvement in DRAM and L2 bandwidth, up to 2x improvement in L2 capacity

### HBM2
- **A100 SXM**: 40/80 GB, 5 HBM2 stacks, 10 × 512-bit memory controllers. 2039 GB/s bandwidth
- **A100 PCIe**: 40/80 GB, 1555 GB/s bandwidth
- 1215 MHz DDR data rate
- SECDED ECC protection on HBM, L2, L1, and register files

## Tensor Cores (3rd Generation)

### Supported Data Types
TF32, FP16, BF16, FP64, INT8, INT4, binary — all (except FP64 and binary) with structured sparsity support.

### Peak Performance (A100 SXM, at boost clock)

| Data Type | Dense | With 2:4 Sparsity |
|---|---|---|
| FP16 TC | 312 TFLOPS | 624 TFLOPS |
| BF16 TC | 312 TFLOPS | 624 TFLOPS |
| TF32 TC | 156 TFLOPS | 312 TFLOPS |
| FP64 TC | 19.5 TFLOPS | — |
| INT8 TC | 624 TOPS | 1248 TOPS |
| INT4 TC | 1248 TOPS | 2496 TOPS |
| FP32 (CUDA cores) | 19.5 TFLOPS | — |
| FP16 (CUDA cores) | 78 TFLOPS | — |

Each Tensor Core performs 256 FP16/FP32 FMA operations per clock. Four per SM = 1024 dense FP16/FP32 FMA ops per clock per SM.

### TF32 (TensorFloat-32)
- 19-bit format: 8-bit exponent + 10-bit mantissa + 1 sign bit
- **Range of FP32** with **precision of FP16**
- 10x faster than V100 FP32 FMA, 20x with sparsity
- **Drop-in acceleration**: existing FP32 code gets TF32 speedup on Ampere tensor cores without code changes when `allow_tf32=True`

### BF16 (Brain Floating Point)
- 16-bit: 8-bit exponent + 7-bit mantissa + 1 sign bit
- Same dynamic range as FP32, less precision than FP16
- Popular for training (better range than FP16, avoids loss scaling)

### MMA Instruction Sizes (PTX)
- HMMA (FP16/BF16): 16×8×8, 16×8×16
- HMMA (TF32): 16×8×4
- IMMA (INT8): 8×8×16, 16×8×16, 16×8×32
- IMMA (INT4): 8×8×32, 16×8×32, 16×8×64
- BMMA (binary): 8×8×128, 16×8×128, 16×8×256
- DMMA (FP64): 8×8×4

### Structured Sparsity (2:4)
- Fine-grained pattern: exactly 2 non-zero values in every 4-entry vector, applied per row
- **Workflow**: (1) train with dense weights → (2) apply 2:4 pruning → (3) fine-tune remaining weights
- Doubles tensor core throughput by skipping zero-value computations
- Well-defined structure allows efficient compression: ~2x memory reduction with metadata

## New Features vs Volta

### Async Copy (cp.async)

Loads data directly from global memory into shared memory, **bypassing the register file**:

- Truly asynchronous: copy proceeds in background while SM computes
- Reduces register file bandwidth pressure and power consumption
- Can bypass L1 cache for streaming access patterns
- Accessed via `cuda::pipeline` or `cuda::memcpy_async` APIs

```cpp
// Using pipeline API
__shared__ float smem[N];
cuda::pipeline pipe;
cuda::memcpy_async(&smem[tid], &global[offset], sizeof(float), pipe);
pipe.producer_commit();
pipe.consumer_wait();
__syncthreads();
```

PTX: `cp.async.ca.shared.global [smem], [gmem], 16;`

### Hardware Async Barriers (Split Arrive/Wait)

- Hardware-accelerated barriers stored in shared memory
- Split `arrive` (non-blocking signal) and `wait` (blocking) operations
- Enables overlapping async copies from global memory with SM computation
- Compatible with cp.async operations
- Foundation for producer-consumer patterns

```cpp
// arrive: signal without blocking
barrier.arrive();
// ... do other work ...
// wait: block until all arrivals complete
barrier.wait();
```

### L2 Cache Residency Controls

Programmable control over which data stays in or is evicted from L2:
- **Persisting**: data is kept in the set-aside L2 portion
- **Streaming**: data passes through without caching (evict-first)
- Use for frequently accessed lookup tables, KV caches, etc.

### Multi-Instance GPU (MIG)

Partitions A100 into up to **7 separate GPU instances** for multi-tenant use:
- Each instance has **fully isolated** paths: crossbar ports, L2 cache banks, memory controllers, DRAM address buses
- Provides defined QoS — no interference between tenants
- Each instance has its own NVDEC/NVJPG
- Beneficial for cloud service providers (VMs, containers)

### Warp-Level Reduction Operations (Hardware)

Native hardware support for 32-bit reductions within a warp:
- Arithmetic: add, min, max (signed/unsigned integers)
- Bitwise: and, or, xor (unsigned integers)
- Faster than shuffle-based software reductions

### Interconnect

| Feature | Spec |
|---|---|
| **NVLink 3rd gen** | 12 links, 50 GB/s per link, 600 GB/s total bidirectional (2x V100's 300 GB/s) |
| **PCIe Gen 4** | 31.5 GB/s per x16 slot (2x Gen 3). SR-IOV support. |

**NVLink TLB**: 64 GB reach per remote GPU. Applications with remote random accesses should constrain remotely accessed region to 64 GB per peer.

Peer access requires `cudaDeviceEnablePeerAccess()`.

## Compilation

```bash
nvcc -arch=sm_80 -O3 -lineinfo kernel.cu           # A100
nvcc -arch=sm_86 -O3 -lineinfo kernel.cu           # RTX 3090, A40
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_86,code=compute_86 \     # + PTX for forward compat
     kernel.cu
```

**CC 8.6 note**: compile explicitly for SM86 to benefit from the doubled FP32 throughput (concurrent INT32+FP32 pipeline). SM80 code runs on SM86 but does not exploit this.

## Tuning Recommendations

1. Overall, expect similar occupancy as Volta without code changes
2. Use `cudaFuncSetAttribute` with `cudaFuncAttributePreferredSharedMemoryCarveout` to configure shared memory (up to 164 KB on CC 8.0)
3. **cp.async**: use the pipeline API to overlap data movement with computation, avoiding register file intermediary. Pair with hardware async barriers for producer-consumer patterns.
4. **L2 residency**: use for frequently accessed data (KV cache, lookup tables). Set hitRatio and persisting/streaming policies.
5. **Structured sparsity**: for inference, prune weights to 2:4 pattern for 2x tensor core throughput
6. **TF32**: ensure `torch.backends.cuda.matmul.allow_tf32 = True` for automatic FP32→TF32 speedup
7. Compile explicitly for CC 8.6 when targeting RTX 3090/A40 class GPUs
8. Ensure coalesced global memory access. Minimize redundant global memory reads.
9. Avoid long sequences of diverged execution within the same warp
10. Minimize host-device data transfers; use pinned memory and async transfers
11. For NVLink peer access: constrain remote random access regions to 64 GB per peer for optimal TLB performance

## Comparative Summary: Ampere vs Hopper vs Blackwell

| Feature | A100 (Ampere) | H100 (Hopper) | B200 (Blackwell) |
|---|---|---|---|
| Process | TSMC 7nm | TSMC 4N | TSMC 4NP |
| Transistors | 54.2B | 80B | 208B (2-die) |
| SMs | 108 | 132 | 148 |
| FP32 cores/SM | 64 | 128 | 128 |
| Tensor Core gen | 3rd | 4th | 5th |
| FP8 support | No | Yes | Yes |
| FP4 support | No | No | Yes |
| Peak FP16 TC (dense) | 312 TFLOPS | 989 TFLOPS | 2250 TFLOPS |
| Shared mem + L1 | 192 KB | 256 KB | 256 KB |
| Max shared mem | 164 KB | 228 KB | 228 KB |
| L2 cache | 40 MB | 50 MB | 96-126 MB |
| HBM | 80 GB, 2 TB/s | 80 GB, 3.35 TB/s | 192 GB, 8 TB/s |
| NVLink | 12 links, 600 GB/s | 18 links, 900 GB/s | 1800 GB/s |
| TMA | No | Yes | Yes (enhanced) |
| TMEM | No | No | Yes |
| Thread Block Clusters | No | Yes | Yes |
| DSMEM | No | Yes | Yes |

## Physical Specs

- TSMC 7nm (N7), 54.2 billion transistors, 826 mm² die
