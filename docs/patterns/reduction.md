# Reduction Patterns

## Warp-Level Reduction

From `flash-attention-fa4-v4.0.0.beta4/hopper/utils.h`:

```cpp
template<typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(T const &x, T const &y) { return max(x, y); }
};

template<typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(T const &x, T const &y) { return x + y; }
};

// Recursive butterfly allreduce — works for 4, 8, 16, 32 threads
template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

template<>
struct Allreduce<2> {
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
        return x;
    }
};
```

FlashAttention uses `Allreduce<4>` for online softmax — in SM90's WGMMA layout, 4 threads share each row.

## Block-Level Reduction (Cross-Warp)

From `flashinfer-0.6.7/include/flashinfer/norm.cuh` (RMSNorm kernel):

```cpp
// Step 1: Each thread accumulates over its elements
float sum_sq = 0.f;
for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    input_vec.load(input + offset);
    for (uint32_t j = 0; j < VEC_SIZE; j++)
        sum_sq += float(input_vec[j]) * float(input_vec[j]);
}

// Step 2: Warp-level butterfly reduce
for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2)
    sum_sq += math::shfl_xor_sync(sum_sq, offset);

// Step 3: Cross-warp reduce via shared memory (one float per warp)
smem[ty] = sum_sq;
__syncthreads();
if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2)
        sum_sq += math::shfl_xor_sync(sum_sq, offset);
    smem[0] = sum_sq;
}
__syncthreads();
float rms_rcp = math::rsqrt(smem[0] / float(d) + eps);
```

Pattern: thread-local accumulate → warp shuffle → shared memory → warp 0 reduces → broadcast via shared memory.

## Online Softmax (FlashAttention)

The key algorithmic enabler for fused attention. Computes softmax incrementally across K-tiles without materializing the full N×N attention matrix.

From `flash-attention-fa4-v4.0.0.beta4/hopper/softmax.h`:

### State: running max and sum per row

```cpp
template <int kNRows, int Max_offset=0>
struct Softmax {
    TensorT row_max, row_sum;
    float const softmax_scale_log2;
```

### Update max and compute rescaling factor

```cpp
template<bool Is_first, bool Check_inf, typename Tensor0>
__forceinline__ __device__ TensorT max_get_scale(Tensor0 &acc_s) {
    Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
    TensorT scores_scale;
    if constexpr (Is_first) {
        reduce_max<true>(scores, row_max);
        cute::fill(scores_scale, 1.f);
    } else {
        Tensor scores_max_prev = make_fragment_like(row_max);
        cute::copy(row_max, scores_max_prev);
        reduce_max<false>(scores, row_max);                    // update max
        for (int mi = 0; mi < size(row_max); ++mi) {
            scores_scale(mi) = exp2f((scores_max_prev(mi) - row_max(mi)) * softmax_scale_log2);
            row_sum(mi) *= scores_scale(mi);                   // rescale previous sum
        }
    }
    return scores_scale;
};
```

### Apply exp2 and accumulate row sums

```cpp
template<bool Is_first, bool Check_inf, typename Tensor0>
__forceinline__ __device__ void online_softmax(Tensor0 &acc_s) {
    Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
    scale_apply_exp2<true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);
    reduce_sum<Is_first, /*warp_reduce=*/false>(scores, row_sum);
};
```

### Rescale output accumulator when max changes

```cpp
template<typename Tensor1>
__forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) {
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi)
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni)
            acc_o_rowcol(mi, ni) *= scores_scale(mi);
};
```

### exp2 with fused scale and max subtraction

```cpp
template <bool Scale_max=true, bool Check_inf=true, int Max_offset=0>
__forceinline__ __device__ void scale_apply_exp2(Tensor &tensor, Tensor const &max, float scale) {
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * scale - float(Max_offset);
        for (int ni = 0; ni < size<1>(tensor); ++ni)
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
    }
}
```

**How it fits together**: Each K-tile iteration calls `max_get_scale` (update running max, compute rescale factor) → `rescale_o` (adjust output accumulator for new max) → `online_softmax` (apply exp2, accumulate sum) → the softmax output drives the PV GEMM. The full N×N attention matrix never leaves registers.

### Layout conversion: MMA accumulator → rows/columns

```cpp
// SM90: ((2, 2, V), MMA_M, MMA_N) → (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
template<typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_rowcol(Layout0 acc_layout) {
    auto l = acc_layout;
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)),          // nrow
                       make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)));  // ncol
};
```

This reshapes the MMA accumulator fragment so row-wise operations (max, sum) map naturally to the thread layout.

## Triton Fused Softmax

For comparison — Triton's approach from `triton-3.6.0/python/tutorials/02-fused-softmax.py`:

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))   # ONE load
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        tl.store(output_ptrs, softmax_output, mask=mask)            # ONE store
```

Constraint: entire row fits in SRAM (BLOCK_SIZE = next_power_of_2(n_cols)). Persistent scheduling with `tl.range`. Triton reports ~4x speedup over naive PyTorch (2MN vs 8MN DRAM traffic).

## RMSNorm (FlashInfer)

From `flashinfer-0.6.7/include/flashinfer/norm.cuh`:

```cpp
// Launch: one block per row, dim3(32, num_warps)
// Pass 1: sum of squares with vectorized loads
float sum_sq = 0.f;
for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    input_vec.load(input + offset);
    for (uint32_t j = 0; j < VEC_SIZE; j++)
        sum_sq += float(input_vec[j]) * float(input_vec[j]);
}
// Warp reduce → shared memory → warp 0 reduce → broadcast
float rms_rcp = math::rsqrt(sum_sq / float(d) + eps);

// Pass 2: normalize + weight (re-reads input)
for (uint32_t i = 0; i < rounds; i++) {
    input_vec.load(input + offset);
    weight_vec.load(weight + offset);
    for (uint32_t j = 0; j < VEC_SIZE; j++)
        output_vec[j] = float(input_vec[j]) * rms_rcp * (weight_bias + float(weight_vec[j]));
    output_vec.store(output + offset);
}
```

Two-pass: compute sum-of-squares, then normalize + scale. Uses `vec_t<T, VEC_SIZE>` for vectorized loads. The Hopper variant adds programmatic dependent launch (PDL) for kernel overlap.

