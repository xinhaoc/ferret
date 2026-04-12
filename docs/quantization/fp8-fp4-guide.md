# FP8 / FP4 Quantization Reference

## FP8 Formats

### E4M3 (4 exponent, 3 mantissa)

| Property | Value |
|---|---|
| Bit layout | 1 sign + 4 exponent + 3 mantissa |
| Exponent bias | 7 |
| Max value | 448 |
| Dynamic range | ~18 binades |
| Has infinity | **No** (extends range by 1 binade) |
| Has NaN | Yes (single pattern) |
| Precision | 8 distinct mantissa values per exponent |

**When to use**: Forward pass (weights and activations). Higher precision matters more than range for these tensors.

### E5M2 (5 exponent, 2 mantissa)

| Property | Value |
|---|---|
| Bit layout | 1 sign + 5 exponent + 2 mantissa |
| Exponent bias | 15 |
| Max value | 57,344 |
| Dynamic range | ~30 binades |
| Has infinity | Yes (IEEE 754 compliant) |
| Precision | 4 distinct mantissa values per exponent |

**When to use**: Backward pass (gradients). Gradients vary wildly in magnitude, needing wider range.

### Comparison

| | E4M3 | E5M2 | BF16 | FP16 |
|---|---|---|---|---|
| Exponent bits | 4 | 5 | 8 | 5 |
| Mantissa bits | 3 | 2 | 7 | 10 |
| Max value | 448 | 57,344 | ~3.4e38 | 65,504 |
| Precision | Higher | Lower | Much higher | Highest |

### HYBRID Format (Recommended for Training)

- **Forward pass**: E4M3 for weights and activations
- **Backward pass**: E5M2 for gradients

With block-scaled methods (MXFP8), E4M3 throughout is sufficient because block-level scaling provides adequate dynamic range.

## FP4 Formats

### E2M1 (Base Element Type)

| Property | Value |
|---|---|
| Bit layout | 1 sign + 2 exponent + 1 mantissa |
| Max value | 6 |
| Representable positive values | {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} |

Only 15 distinct values (7 positive + 7 negative + zero). Requires block scaling for practical use.

### NVFP4 (NVIDIA's FP4 — Blackwell)

- **Element type**: E2M1
- **Block size**: 16 elements
- **Level-1 scale**: FP8 E4M3 per block of 16 elements
- **Level-2 scale**: FP32 per-tensor global factor
- **Storage**: ~4.5 bits/value (element + amortized scale)
- **Memory savings**: 3.5x vs FP16, 1.8x vs FP8
- **Accuracy**: <1% degradation on language benchmarks vs FP8

Dequantization: `x = x_q × s_block × s_tensor`

### MXFP4 (OCP Microscaling FP4)

- **Element type**: E2M1
- **Block size**: 32 elements
- **Scale type**: E8M0 (power-of-2 only)
- **Limitation**: Power-of-2 scales → higher MSE than NVFP4

### MXFP8 (OCP Microscaling FP8)

- **Element type**: E4M3 or E5M2
- **Block size**: 32 elements
- **Scale type**: E8M0 (power-of-2 only)
- **Key advantage**: All values can use E4M3 (block scaling handles range)
- **Hardware**: Blackwell native, AMD CDNA 3+
- **Transpose**: Requires requantization (create both regular + transposed from high-precision source)

### E8M0 Scale Format

8 exponent bits, 0 mantissa, no sign. Represents unsigned powers of 2. Used as block scaling factors in MX formats.

## Scaling Strategies

### Per-Tensor Delayed Scaling

```
scale = FP8_MAX / amax_history / (2^margin)
```

- Maintains rolling amax history buffer (default 1024 entries)
- Selection: `'max'` or `'most_recent'` from history
- Scale applied in NEXT iteration (stale by 1 step)
- **Pros**: No extra passes, smooths outliers
- **Cons**: Stale scales, outliers dominate history

### Per-Tensor Current (Just-in-Time) Scaling

```
scale = FP8_MAX / amax(current_tensor)
```

- Computed from the tensor's amax during current pass
- **Pros**: Immediate adaptation, better convergence
- **Cons**: Reacts to transient spikes, slightly more compute

### Per-Block Scaling (Generic)

```
scale_per_block = max_i(abs(x_i)) / FP8_MAX    (per configurable block)
```

- Block dimensions configurable (e.g., 1×128, 128×128)
- FP32 scale factors (optionally power-of-2 constrained)
- Transposed tensor requires separate quantization

### MXFP8 Block Scaling (Hardware-Accelerated)

```
scale_E8M0 = roundUpToE8M0(max_i(abs(x_i)) / qTypeMax)    (per 32 elements)
```

- E8M0 (power-of-2) scale per 32-element block
- Hardware computes scales inside the GEMM path on Blackwell
- Both regular and transposed copies from high-precision input

### NVFP4 Two-Level Block Scaling

```
s_block = quantize_to_e4m3(max_i(abs(x_i)) / FP4_MAX, scale=s_tensor)
x_q = castToFp4(x / (s_block × s_tensor))
```

- Level 1: FP8 E4M3 scale per 16 elements
- Level 2: FP32 per-tensor global scale (offline calibrated)
- Additional techniques for training:
  - **Stochastic rounding**: Eliminates quantization bias in gradients
  - **2D scaling**: 16×16 block factors for transpose sensitivity
  - **Random Hadamard Transforms**: Smooth outlier distributions

### Strategy Selection

| Strategy | Granularity | Scale Type | Hardware | Best For |
|---|---|---|---|---|
| Delayed | Per-tensor | FP32 | Hopper+ | Stable training |
| Current | Per-tensor | FP32 | Hopper+ | Dynamic distributions |
| Block (generic) | Configurable | FP32 | Hopper+ | Fine-grained control |
| MXFP8 | 32-element | E8M0 | Blackwell | Training + inference |
| NVFP4 | 16-element | E4M3+FP32 | Blackwell | Inference, memory-bound |

## How FP8 GEMM Works

### Formula

```
D = alpha × (scale_A × scale_B) × A_fp8 × B_fp8 + beta × scale_C × C
```

### Scale Factor Roles

- `scale_A`, `scale_B`: Dequantize inputs (expand from FP8 range)
- `scale_C`: Dequantize accumulator input
- `scale_D`: Quantize output (compress back to FP8)

### Accumulation Precision

**Critical**: Hopper FP8 Tensor Cores use a **14-bit internal accumulator**, not full FP32. This causes error for large K dimensions.

**Fix (DeepGEMM approach)**: Two-level accumulation:
1. Inner accumulation in TC's native 14-bit precision
2. Periodic promotion to FP32 on CUDA cores (~1 in 4 WGMMAs)

A and B are in FP8; final result accumulated in FP32.

### Dimension Constraints

Both dimensions must be divisible by 16 for FP8 Linear layers (pad sequence length as needed).

## Quantize / Dequantize Formulas

### FP8 E4M3

```
Quantize:   x_q = roundNearestEven(clamp(x / scale, -448, 448))
Dequantize: x   = x_q × scale
Scale:      scale = amax(tensor) / 448
```

### FP8 E5M2

```
Quantize:   x_q = roundNearestEven(clamp(x / scale, -57344, 57344))
Dequantize: x   = x_q × scale
```

### NVFP4 (two-level)

```
Quantize:   x_q = roundToFP4(clamp(x / (s_block × s_tensor), -6, 6))
Dequantize: x   = x_q × s_block × s_tensor
```

### MXFP8 (block-scaled)

```
scale_E8M0 = roundUpToE8M0(max_i(abs(x_i)) / 448)    per 32-element block
Quantize:   x_q = roundNearestEven(x / decode_E8M0(scale))
Dequantize: x   = x_q × decode_E8M0(scale)
```

### INT8 (for reference)

```
Quantize:   x_q = roundNearestEven(clamp(x / scale, -128, 127))
Dequantize: x   = x_q × scale
```

## Best Practices

### Training

1. **HYBRID format** for per-tensor delayed scaling. **E4M3 throughout** for block-scaled (MXFP8).
2. **Dimensions divisible by 16** (pad sequences).
3. **Last layers in higher precision** — final LLM layers are sensitive. Use MXFP8 or BF16 for them.
4. **Multi-GPU**: Set `reduce_amax=True` to sync across distributed groups.
5. **NVFP4 training requires**: stochastic rounding + 2D scaling + random Hadamard transforms.
6. **Monitor**: MXFP8 should track BF16 validation perplexity closely.

### Inference

1. **E4M3 throughout** (no backward pass).
2. **Per-tensor activations + per-channel weights** for best accuracy.
3. **Dynamic quantization** for activations (compute scales at runtime per block).
4. **NVFP4** when memory bandwidth is the bottleneck. Verify <1% accuracy loss.

### Kernel Development

1. **Always accumulate in FP32** — never accumulate FP8 × FP8 in FP8.
2. **Periodic FP32 promotion** in the K-loop to mitigate 14-bit accumulator error on Hopper.
3. **Block scales in shared memory** for fast access during dequantization.
4. **Use cuBLAS/CUTLASS FP8 GEMM** rather than writing custom FP8 matmul unless you need custom epilogues.
5. **Test with per-block scaling** even if deploying per-tensor — it reveals sensitivity to value distribution.

## Transformer Engine Recipe Reference

```python
from transformer_engine.common.recipe import (
    DelayedScaling,        # Per-tensor delayed
    Float8CurrentScaling,  # Per-tensor current
    Float8BlockScaling,    # Generic per-block
    MXFP8BlockScaling,     # MXFP8 (32-element, E8M0 scales)
    NVFP4BlockScaling,     # NVFP4 (16-element, two-level)
    Format,
)

# Delayed scaling (Hopper)
recipe = DelayedScaling(margin=0, fp8_format=Format.HYBRID, amax_history_len=1024)

# MXFP8 (Blackwell)
recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)

# NVFP4 (Blackwell inference)
recipe = NVFP4BlockScaling(fp4_format=Format.E2M1)

# Usage
with te.autocast(enabled=True, recipe=recipe):
    out = model(inp)
# Backward OUTSIDE autocast
out.backward(grad)
```
