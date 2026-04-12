# cuBLAS / cuDNN Performance Baselines

Reference numbers for common operations. Use these to judge whether a custom kernel is worth optimizing further. If cuBLAS already achieves 85%+ of peak, beating it is very hard.

## How to Measure Baselines

```python
import torch
import torch.utils.benchmark as benchmark

# GEMM baseline
M, N, K = 4096, 4096, 4096
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)

timer = benchmark.Timer(
    stmt="torch.mm(a, b)",
    globals={"a": a, "b": b},
)
result = timer.blocked_autorange()
print(f"Time: {result.median * 1e3:.3f} ms")

flops = 2 * M * N * K
tflops = flops / result.median / 1e12
print(f"TFLOPS: {tflops:.1f}")
```

## GEMM (torch.mm → cuBLAS)

Matrix multiply: `C[M,N] = A[M,K] × B[K,N]`
FLOPs = 2 × M × N × K

### Square GEMM (M = N = K)

| GPU | dtype | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|
| **A100-SXM** | FP16 | ~200 TFLOPS | ~280 TFLOPS | ~290 TFLOPS | ~300 TFLOPS | ~305 TFLOPS |
| **A100-SXM** | FP32 (TF32) | ~100 TFLOPS | ~140 TFLOPS | ~150 TFLOPS | ~155 TFLOPS | ~155 TFLOPS |
| **H100-SXM** | FP16 | ~500 TFLOPS | ~750 TFLOPS | ~850 TFLOPS | ~900 TFLOPS | ~930 TFLOPS |
| **H100-SXM** | FP8 | ~800 TFLOPS | ~1300 TFLOPS | ~1600 TFLOPS | ~1750 TFLOPS | ~1850 TFLOPS |
| **B200** | FP16 | ~1000 TFLOPS | ~1600 TFLOPS | ~1900 TFLOPS | ~2050 TFLOPS | ~2100 TFLOPS |
| **B200** | FP8 | ~1600 TFLOPS | ~2800 TFLOPS | ~3500 TFLOPS | ~3900 TFLOPS | ~4200 TFLOPS |
| **B200** | FP4 | ~1600 TFLOPS | ~2800 TFLOPS | ~3500 TFLOPS | ~3900 TFLOPS | ~4200 TFLOPS |

Note: B200 FP4 dense = 9000 TFLOPS (2x FP8 dense of 4500). FP4 doubles both compute throughput and memory efficiency.

**Key insight**: cuBLAS achieves ~90-95% of peak for large GEMMs (≥4096). Small GEMMs are latency-bound.

### Common LLM GEMM Shapes

For transformer inference (batch=1, hidden=4096, FFN=11008):

| Operation | Shape (M×N×K) | A100 FP16 | H100 FP16 | Bound |
|---|---|---|---|---|
| QKV projection | 1×12288×4096 | ~5 us | ~3 us | Memory |
| Attention output | 1×4096×4096 | ~3 us | ~2 us | Memory |
| FFN up | 1×11008×4096 | ~8 us | ~5 us | Memory |
| FFN down | 1×4096×11008 | ~8 us | ~5 us | Memory |
| Large batch QKV | 256×12288×4096 | ~0.3 ms | ~0.15 ms | Compute |

**Key insight**: Batch=1 inference GEMMs are memory-bound. Custom kernels rarely beat cuBLAS here. Focus custom kernels on fused operations instead.

## Attention (FlashAttention via torch.nn.functional.scaled_dot_product_attention)

`softmax(QK^T / √d) × V`, Q,K,V shape: [batch, heads, seq_len, head_dim]

### Typical Performance (head_dim=128)

| GPU | batch×heads | seq_len=2048 | seq_len=4096 | seq_len=8192 |
|---|---|---|---|---|
| **A100** | 32 | ~300 TFLOPS | ~290 TFLOPS | ~280 TFLOPS |
| **H100** | 32 | ~700 TFLOPS | ~750 TFLOPS | ~730 TFLOPS |

FlashAttention achieves ~60-75% of peak FP16 TFLOPS on attention.

## Element-wise / Reduction Operations

These are **memory-bound** — the metric that matters is bandwidth, not FLOPS.

| Operation | A100 BW (GB/s) | H100 BW (GB/s) | Notes |
|---|---|---|---|
| Vector add (FP32, large) | ~1600 | ~2800 | ~80% of peak |
| ReLU (FP16, large) | ~1700 | ~3000 | Read + write |
| Softmax (FP16, dim=-1) | ~800-1200 | ~1500-2500 | Multiple passes |
| LayerNorm (FP16) | ~800-1200 | ~1500-2500 | Reduction + normalize |
| Sum reduction (FP32) | ~1500 | ~2800 | Near peak read BW |

**Key insight**: For memory-bound ops, the ceiling is peak HBM bandwidth. A good kernel achieves 80%+ of peak. The win comes from **fusing** multiple ops to reduce total DRAM traffic.

## Convolution (cuDNN)

2D convolution: input [N,C,H,W], filter [K,C,R,S]

| GPU | dtype | 1×64×224×224, 3×3 | 32×128×56×56, 3×3 | 32×256×28×28, 3×3 |
|---|---|---|---|---|
| A100 | FP16 | ~50 TFLOPS | ~250 TFLOPS | ~280 TFLOPS |
| H100 | FP16 | ~100 TFLOPS | ~600 TFLOPS | ~750 TFLOPS |

## How to Interpret These Numbers

### "Is my kernel good enough?"

| Your kernel vs cuBLAS/cuDNN | Verdict |
|---|---|
| **> 90%** of baseline | Excellent — you're near vendor-optimized |
| **70-90%** | Good for a custom kernel. Check if fusion gives net win over separate cuBLAS calls |
| **50-70%** | Room for improvement. Check occupancy, memory access patterns, tensor core usage |
| **< 50%** | Likely a bug or fundamental approach issue |

### When custom kernels win over cuBLAS

1. **Operator fusion**: Combining GEMM + bias + activation + residual into one kernel eliminates intermediate DRAM writes. Even if each individual op is slower than cuBLAS, the fused version can be 2-3x faster end-to-end.

2. **Non-standard shapes**: cuBLAS is tuned for common shapes. Very small M (batch=1 inference), very large K, or irregular shapes may have room for custom kernels.

3. **Custom precision**: FP8 with custom quantization, mixed-precision accumulation, or non-standard number formats.

4. **Memory-bound ops**: Fusing multiple element-wise ops (norm + activation + dropout) saves DRAM bandwidth.

## How to Run Your Own Baselines

```bash
# Quick benchmark script
python -c "
import torch, time

def bench(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

M, N, K = 4096, 4096, 4096
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

t = bench(lambda: torch.mm(a, b))
tflops = 2*M*N*K / t / 1e12
print(f'GEMM {M}x{N}x{K} FP16: {t*1e3:.3f} ms, {tflops:.0f} TFLOPS')
"
```

## Notes

- All numbers are approximate and vary by driver version, CUDA toolkit version, and thermals.
- cuBLAS auto-selects algorithms. Use `torch.backends.cuda.matmul.allow_tf32 = True` for TF32 on Ampere+.
- cuBLAS has a startup cost (~10-50 us) for the first call per shape. Subsequent calls are faster.
- For FP8 GEMM on H100, use `torch._scaled_mm()` or cuBLAS FP8 API directly.
