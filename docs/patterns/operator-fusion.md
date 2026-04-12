# Operator Fusion

## Why Fuse

Each standalone kernel reads from and writes to DRAM. Fusing multiple ops into one kernel eliminates intermediate DRAM traffic.

| Fusion Pattern | Unfused DRAM | Fused DRAM | Savings |
|---|---|---|---|
| GEMM + bias + ReLU | MNK + 3MN | MNK + MN | ~2MN eliminated |
| Attention (QKV) | O(N^2) materialized | O(N*d) | N/d ratio (typically 32x) |
| LayerNorm + residual + bias | ~9N per token | ~6N per token | ~33% |
| Standalone softmax | 8MN + 4M | 2MN | ~4x |

For memory-bound ops, this translates directly to wall-clock speedup.

## Pattern 1: GEMM + Epilogue Fusion

Without fusion: GEMM writes output D to DRAM, then a separate kernel reads D, adds bias, applies activation, writes again. With fusion: bias + activation applied to accumulator registers *before* the single store.

### How CUTLASS does it

CUTLASS uses a compile-time fusion operation selected as a template parameter. From `cutlass-4.4.2/include/cutlass/epilogue/fusion/operations.hpp`:

```cpp
// D = activation(alpha * acc + beta * C + per-row bias)
template<
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  ...
>
struct LinCombPerRowBiasEltAct
    : LinCombPerRowBias<ElementOutput_, ElementCompute_, ElementBias_, ...> {
  using ActivationFn = ActivationFn_<ElementCompute_>;
  static constexpr bool IsEltActSupported = true;
};
```

Used concretely in `cutlass-4.4.2/examples/54_hopper_fp8_warp_specialized_gemm/`:

```cpp
using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltActAmaxAux<
    LayoutAux, cutlass::epilogue::thread::ReLU, ElementD, ElementCompute,
    ElementAux, ElementAmax, ElementBias, ElementC>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType,
    ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD, EpilogueSchedule,
    FusionOperation          // <--- fused bias + ReLU + amax + aux output
  >::CollectiveOp;
```

The activation functions from `cutlass/epilogue/thread/activation.h`:

```cpp
template <typename T>
struct ReLu {
  CUTLASS_HOST_DEVICE T operator()(T value) const {
    return max(value, T(0));
  }
};

template <typename T>
struct GELU {
  CUTLASS_HOST_DEVICE T operator()(T const &value) const {
    return T(half * value * (one + (T)erff((float)(value * half_root_two))));
  }
};
```

The entire chain `Z = scale_a * scale_b * alpha * acc + beta * scale_c * C + bias; D = scale_d * activation(Z)` executes in registers. The accumulator never touches DRAM between the MMA and the final store.

## Pattern 2: Fused Attention (FlashAttention)

Standard attention: `O = softmax(Q @ K^T) @ V` in 3 kernels. The N×N attention matrix materializes in DRAM (quadratic!). FlashAttention fuses all three using **online softmax** — tiling over the N dimension, keeping the attention matrix entirely in registers.

### The fused loop

From `flash-attention-fa4-v4.0.0.beta4/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`, the `fwd_step` lambda:

```cpp
auto fwd_step = [&](int const n_block, auto mask_fn, auto check_inf_type) {
    // 1. GEMM: Q @ K^T for current block → scores in registers
    Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
    flash::gemm<true>(tiled_mma_qk, tSrQ, tSrK(...), tSrS);

    // 2. Rescale running O accumulator (from previous iteration's max)
    if constexpr (RescaleOBeforeGemm) { softmax.rescale_o(tOrO, scores_scale); }

    // 3. GEMM: P @ V for PREVIOUS block's softmax output → accumulate into O
    flash::gemm<false>(tiled_mma_pv, tOrP, tOrV(...), tOrO);

    // 4. Online softmax on current QK scores
    scoremod_premask_fn(tSrS);
    mask_fn(tSrS, n_block);
    cute::copy(softmax.template max_get_scale<false, Check_inf>(tSrS), scores_scale);
    softmax.template online_softmax<false, Check_inf>(tSrS);

    // 5. Convert S → P for next iteration's PV GEMM
    convert_type_out(make_tensor(tSrS.data(), tOrP.layout()), tOrP);
};
```

The online softmax from `flash-attention-fa4/hopper/softmax.h`:

```cpp
template<bool Is_first, bool Check_inf, typename Tensor0>
__forceinline__ __device__ TensorT max_get_scale(Tensor0 &acc_s) {
    Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
    if constexpr (Is_first) {
        reduce_max<true>(scores, row_max);
        cute::fill(scores_scale, 1.f);
    } else {
        cute::copy(row_max, scores_max_prev);
        reduce_max<false>(scores, row_max);
        for (int mi = 0; mi < size(row_max); ++mi) {
            scores_scale(mi) = exp2f((scores_max_prev(mi) - row_max(mi)) * softmax_scale_log2);
            row_sum(mi) *= scores_scale(mi);  // rescale running sum
        }
    }
    return scores_scale;
};
```

**Key insight**: The loop interleaves GEMMs with softmax — while iteration N's QK GEMM executes on tensor cores, iteration N-1's softmax output drives the PV GEMM. The N×N attention matrix **never materializes in DRAM**.

## Pattern 3: Fused LayerNorm + Residual + Bias

A Transformer block computes: `x = input + residual + bias; y = layernorm(x) * gamma + beta`. Without fusion: 3 kernels, ~9N DRAM accesses per token. With fusion: 1 kernel, ~6N accesses.

### How TensorRT-LLM does it

From `tensorrt-llm-1.2.0/cpp/tensorrt_llm/kernels/fusedLayernormKernels/low_latency_layernorm.cuh`:

```cpp
static __device__ void compute(const Param param, Shared* shared) {
    // Phase 1: Load everything into registers
    load_to_register(param.bias, r_bias, param.n);
    load_to_register(param.gamma, r_gamma, param.n);
    load_to_register(param.beta, r_beta, param.n);
    load_to_register(&param.input[work_id * param.n], data, param.n);
    load_to_register(&param.residual[work_id * param.n], r_residual, param.n);

    // Phase 2: Fused add + compute mean/variance in one pass
    for (int i = 0; i < PACKED_PER_N_BLOCK; i++) {
        for (int j = 0; j < Traits::PACKED_ELEMS_PER_COMPUTE; j++) {
            if constexpr (Traits::BIAS == SCALE_TYPE::VECTOR)
                data[i][j] += r_bias[i][j];
            if constexpr (Traits::RESIDUAL)
                data[i][j] += r_residual[i][j];
            mean += data[i][j];
            variance += data[i][j] * data[i][j];
        }
    }

    // Phase 3: Cross-warp reduction for mean & variance
    reduceSum<N_THREADS>(var_and_mean, shared->reduce, thread_id, 0);
    variance = rsqrtf(var_and_mean[0] / param.n - mean*mean + eps);

    // Phase 4: Normalize, scale, store (single write)
    for (int i = 0; i < PACKED_PER_N_BLOCK; i++) {
        for (int j = 0; j < Traits::PACKED_ELEMS_PER_COMPUTE; j++) {
            normed_out = (data[i][j] - mean) * variance;
            if constexpr (Traits::GAMMA) normed_out *= r_gamma[i][j];
            if constexpr (Traits::BETA)  normed_out += r_beta[i][j];
        }
        store(normed_output, &param.normed_output[work_id * param.n + n_base]);
    }
}
```

The `Traits` template controls which operations are active — same kernel handles RMSNorm, LayerNorm, with/without residual, with/without bias, with/without output quantization.

TensorRT-LLM also has a warp-specialized variant (`ws_layernorm.cuh`) that uses separate DMA warps with `cp.async.bulk` for Hopper.

## Pattern 4: Fused Softmax

From the Triton tutorial (`triton-3.6.0/python/tutorials/02-fused-softmax.py`):

> When implemented naively in PyTorch, computing `y = softmax(x)` requires reading **5MN + 2M** elements and writing **3MN + 2M**. A fused kernel reads and writes only **MN** each, for a ~4x speedup.

The naive PyTorch (5 separate operations = 5 kernel launches):
```python
def naive_softmax(x):
    x_max = x.max(dim=1)[0]             # kernel 1: read MN, write M
    z = x - x_max[:, None]              # kernel 2: read MN+M, write MN
    numerator = torch.exp(z)             # kernel 3: read MN, write MN
    denominator = numerator.sum(dim=1)   # kernel 4: read MN, write M
    ret = numerator / denominator[:, None] # kernel 5: read MN+M, write MN
```

The fused Triton kernel (1 kernel, 1 load, 1 store per row):
```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))  # ONE load
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        tl.store(output_ptrs, softmax_output, mask=mask)           # ONE store
```

Constraint: each row must fit in SRAM (BLOCK_SIZE elements, padded to power of 2).

## When NOT to Fuse

- **Compute-bound ops with different parallelism**: GEMM (thread-block parallel over output) + reduction (needs different launch config) — may be better separate.
- **When fusion increases register pressure too much**: if fusing drops occupancy significantly and the kernel becomes latency-bound.
- **When cuBLAS is just too good**: for standalone large GEMMs, cuBLAS's hand-tuned kernels are hard to beat. Fuse the epilogue, but keep the GEMM core from cuBLAS/CUTLASS.

For real implementations of these patterns, see `docs/MAPPING.md` → `resources/`.
