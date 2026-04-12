# Occupancy Tuning

## What Is Occupancy

Occupancy = active warps per SM / maximum warps per SM. Higher occupancy = more warps to hide latency. But **max occupancy is not always optimal** — sometimes fewer warps with more registers gives better performance.

## What Limits Occupancy

### Registers per thread

Each SM has 64K × 32-bit registers. More registers per thread → fewer warps can fit.

Example on H100 (max 64 warps/SM):
- 32 regs/thread → 64 warps → 100%
- 128 regs/thread → 16 warps → 25%
- 255 regs/thread → 8 warps → 12.5%

### Shared memory per block

More shared memory per block → fewer blocks per SM → fewer warps.

### Block size

Max threads per SM is fixed (2048 on most GPUs). Very small blocks waste occupancy slots.

See `docs/profiling/gpu-specs.md` for exact limits per architecture.

## __launch_bounds__ in Production

### DeepGemm: 1 block per SM (maximum registers)

From `deepgemm-2.1.1/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh`:

```cpp
__global__ __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_fp8_gemm_1d1d_impl(...)
```

`minBlocks=1` — forces exactly 1 block per SM. This gives each thread maximum register allocation, needed for large WGMMA accumulators.

### TensorRT-LLM: Three-argument form (CUDA 12+)

From `tensorrt-llm-1.2.0/cpp/tensorrt_llm/kernels/flashMLA/flash_fwd_mla_kernel.h`:

```cpp
__global__ void __launch_bounds__(Kernel_traits::kNThreads, 1, 1)
    flash_fwd_splitkv_mla_kernel(...)

__global__ void __launch_bounds__(256, 1, 1)
    flash_fwd_splitkv_mla_combine_kernel(...)
```

Three arguments: `(maxThreads, minBlocks, maxBlocks)`. `256, 1, 1` = 256 threads, exactly 1 block per SM.

### ThunderKittens: 2 blocks per SM (lighter kernel)

From `thunderkittens-main/kernels/based/linear_attn.cu`:

```cpp
__global__ __launch_bounds__(NUM_THREADS, 2)
void based_linear_attention(const __grid_constant__ based_globals g)
```

`minBlocks=2` — this kernel uses less shared memory, so 2 blocks coexist per SM for better latency hiding.

## Register Redistribution (SM90+)

Hopper can dynamically redistribute registers between warp groups at runtime via `setmaxnreg`.

### FlashAttention: Producer gets 24, consumer gets 240

From `flash-attention-fa4-v4.0.0.beta4/hopper/flash_fwd_kernel_sm90.h`:

```cpp
// Computed based on number of MMA warpgroups and whether TMA is used
static constexpr uint32_t LoadRegisterRequirement =
    NumMmaWarpGroups == 2 ? (Use_TMA_KV ? 24 : 40) : 32;
static constexpr uint32_t MmaRegisterRequirement =
    NumMmaWarpGroups == 2 ? (Use_TMA_KV ? 240 : 232) : 160;

// In the kernel body:
if (warp_group_idx == 0) {  // Producer
    cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();  // Release registers
    // ... TMA loads ...
} else {  // Consumer
    cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();     // Claim registers
    // ... WGMMA compute ...
}
```

Producer only needs 24 registers (just issuing TMA commands). Consumer gets 240 (holding large WGMMA accumulator tiles).

### DeepGemm: 48 for TMA, 224 for math

From `deepgemm-2.1.1/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh`:

```cpp
constexpr uint32_t kNumTMARegisters = 48;
constexpr uint32_t kNumMathRegisters = 224;

// TMA warp-group
cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

// Math warp-groups
cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
```

## Shared Memory Carveout

### Standard opt-in (>48KB)

From `flash-attention-fa4-v4.0.0.beta4/hopper/flash_bwd_launch_template.h`:

```cpp
if (smem_size >= 48 * 1024) {
    CHECK_CUDA(cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
}
```

48KB is the default limit. Anything above requires explicit opt-in.

### Graceful fallback (TensorRT-LLM)

From `tensorrt-llm-1.2.0/cpp/tensorrt_llm/kernels/layernormKernels.cu`:

```cpp
bool use_shmem = true;
if (shmem_size >= (48 << 10)) {
    cudaError_t ret = cudaFuncSetAttribute(
        generalLayerNorm<T, QuantT, true, USE_DIFF_OF_SQUARES>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    use_shmem = (ret == cudaSuccess);  // fall back if GPU can't provide enough
}
```

Production pattern: if the GPU can't provide the requested smem, fall back to a no-shared-memory path instead of crashing.

## Occupancy Calculation (Programmatic)

### TensorRT-LLM: Runtime occupancy for kernel selection

From `tensorrt-llm-1.2.0/cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/compute_occupancy.h`:

```cpp
template <typename GemmKernel>
inline int compute_occupancy_for_kernel() {
    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    // Check if config is feasible
    if (smem_size > (48 << 10)) {
        int max_smem_per_block = 0;
        cudaDeviceGetAttribute(&max_smem_per_block,
            cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        if (smem_size >= max_smem_per_block) return 0;  // skip this config

        cudaFuncSetAttribute(cutlass::device_kernel<GemmKernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    int max_active_blocks = -1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
        cutlass::device_kernel<GemmKernel>,
        128 * (GemmKernel::NumLoadWarpGroups + GemmKernel::NumMmaWarpGroups),
        smem_size);
    return max_active_blocks;
}
```

Used at runtime to choose the best GEMM tile configuration: returns 0 for infeasible configs, actual occupancy for feasible ones.

### FlashInfer: Occupancy-driven grid sizing

From `flashinfer-0.6.7/include/flashinfer/norm.cuh`:

```cpp
int num_blocks_per_sm = 0, num_sms = 0;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks_per_sm, kernel, num_warps * 32, smem_size);
cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);
// Grid size for persistent kernel = num_blocks_per_sm * num_sms
```

## When Low Occupancy Is OK

Low occupancy is fine when the kernel is **compute-bound** and near peak throughput:

- GEMM with `__launch_bounds__(threads, 1)`: 1 block per SM but 90%+ of peak TFLOPS
- FlashAttention with 240 regs/thread: ~25% occupancy but fully utilizing tensor cores

Check ncu: if `stall_not_selected` is high, you have more warps than needed — could reduce occupancy to gain registers.

