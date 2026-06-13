# Handoff brief — fix the precision gap in Kimi MoE w13 INT4 kernel

**For a successor agent working under the ferret repo. All paths are relative to `~/repos/ferret/`.**

## Mission

The current kernel — **v049**, saved as `examples/kimi-moe-w13-int4/v049_swapab_n8_int4_silu_1.37x_min.cu` and live in `workspace/kernel.cu` — is **1.37× faster than Marlin** for Kimi K2.6/K2.7 MoE w13 + silu·mul on B200, but its output is **3–4× noisier than Marlin** against the FP32 dequant→BF16 reference. Both pass the loose `maxae < 0.05` validator, but Marlin reaches `maxae ≈ 0.006–0.009` while v049 sits at `maxae ≈ 0.022–0.023`. Your job: **close that precision gap WITHOUT changing the kernel's main architecture or losing the latency win**. Target: `maxae < 0.012` at both T=1 and T=8, with min_ratio still ≥ 1.30 over Marlin.

## Current measured state of v049 (the kernel you start from)

```
config   v049_us   v049_TFLOPS   v049_maxae   Marlin_us   Marlin_TFLOPS   Marlin_maxae   ratio
T=1      19.81     2.96          0.0232       25.63       2.291           0.0064         1.37×
T=8      35.62     13.19         0.0217       51.20       9.175           0.0091         1.43×

min_ratio (perf vs Marlin): 1.367
maxae gap (vs Marlin): 3.6× at T=1, 2.4× at T=8
```

## Hard constraints (read first, do not break)

- **DO NOT change the main architecture.** Specifically keep:
  - `cta_group::1` with `swapAB` (tokens on N=8, weights on M=64)
  - `BLOCK_M=64`, `BLOCK_N=8`, `BLOCK_K=128`, 5 pipeline stages
  - 10 warps (8 dequant + 1 TMA producer + 1 MMA launcher)
  - `tcgen05.mma.cta_group::1.kind::f16` BF16 MMA with FP32 accumulator
  - The fused silu·mul epilogue from TMEM
  - The INT4→BF16 dequant via the Marlin `__hfma2(v, MUL, ADD) * scale` trick
- **DO NOT regress latency** by more than 2% on either config. Final min_ratio must stay ≥ 1.30× over Marlin.
- **DO NOT use cuBLAS / cuDNN / Marlin / vLLM / DeepGEMM / FlashInfer** in your kernel. You may read their source.
- **The K-split atomicAdd reduction is already ruled out as the cause** (proven below). Don't chase it.

## Files you need

```
ferret/
├── workspace/
│   ├── kernel.cu                                  # the current kernel (HEAD)
│   ├── kernel                                     # compiled binary
│   └── progress.md                                # what prior agents tried
├── examples/kimi-moe-w13-int4/
│   ├── v049_swapab_n8_int4_silu_1.37x_min.cu     # frozen v049 reference (1.37× over Marlin)
│   └── v040_swapab_n8_int4_silu_1.29x_min.cu     # earlier checkpoint, archived
├── baselines/kimi-moe-w13-int4/
│   └── baseline_marlin.py                         # vLLM Marlin call site
├── tasks/
│   └── kimi-moe-w13-int4-grouped-gemm.yaml        # task spec (shapes, validator)
├── resources/
│   ├── vllm-csrc-marlin/csrc/libtorch_stable/
│   │   ├── moe/marlin_moe_wna16/marlin_template.h          # MoE Marlin source
│   │   └── quantization/marlin/marlin_template.h           # underlying Marlin
│   └── marlin-standalone/marlin/marlin_cuda_kernel.cu      # IST-DASLab original
└── correctness/                                              # YOU ARE HERE
    ├── kimi-moe-w13-precision-fix.md              # this file
    └── compare_marlin_vs_ref.py                   # Marlin-vs-FP32-oracle comparison
```

The kernel's CPU `reference()` function (in `workspace/kernel.cu`, around line 405) is the oracle the validator compares against. It does:

```
nv_bfloat16 w_bf16 = __float2bfloat16(__bfloat162float(__float2bfloat16((float)(nib-8))) * scale);
acc_fp32 += __bfloat162float(w_bf16) * __bfloat162float(activation_bf16);
// after K, FP32 silu(gate)·up, then cast to bf16
```

This is the **dequant→BF16 + FP32-accumulate + silu·mul** oracle, semantically what Marlin also computes against.

## How to run the kernel

```bash
# Compile (-gencode form is REQUIRED — bare -arch=sm_100a rejects tcgen05)
cd ~/repos/ferret/workspace
/usr/local/cuda/bin/nvcc -O3 --use_fast_math -std=c++17 \
  -gencode arch=compute_100a,code=sm_100a \
  --expt-relaxed-constexpr \
  -I ~/repos/ferret/resources/cutlass-4.4.2/include \
  -o kernel kernel.cu -lcuda

# Use any free GPU (avoid GPU 1 if another ferret run is in progress there)
CUDA_VISIBLE_DEVICES=2 ./kernel
```

Expected v049 output (this is the baseline you need to improve):

```
Reference: T1=2.291 T8=9.175 TFLOPS
T1: 32 blocks, 8 active, P=8, smem=102616
  k_split=4 grid=128
T1: maxre=1.57 maxae=0.0232 errs=0/2048     ← drive maxae DOWN
T1: 19.81 us, 2.964 TFLOPS, bw=0.74 TB/s     ← keep latency within 2%
T8: 248 blocks, 62 active, P=64, smem=102616
  k_split=1 grid=86
T8: maxre=1.27 maxae=0.0217 errs=0/16384     ← drive maxae DOWN
T8: 35.62 us, 13.19 TFLOPS, bw=3.19 TB/s     ← keep latency within 2%
KERNEL_RESULT {"T1": 2.96, "T8": 13.19}
```

The kernel prints `maxae` against its own FP32 oracle. **That is the metric to drive down.**

## How to run Marlin for reference

```bash
# vllm 0.20.1 lives in /home/xinhaoc/vllm-venv on catalyst
CUDA_VISIBLE_DEVICES=2 /home/xinhaocheng/vllm-venv/bin/python3 \
    ~/repos/ferret/baselines/kimi-moe-w13-int4/baseline_marlin.py
```

Expected output:

```
T1:   25.63 us   2.291 TFLOPS  weight_bw= 0.57 TB/s  (8 active experts)
T8:   51.20 us   9.175 TFLOPS  weight_bw= 2.29 TB/s  (64 active experts)
```

`baseline_marlin.py` calls `ops.moe_wna16_marlin_gemm + apply_moe_activation(SILU)` — same operation as the kernel.

## How to compare Marlin OUTPUT vs the FP32 oracle (the real precision check)

`correctness/compare_marlin_vs_ref.py` feeds Marlin the same INT4 weights / scales / activations the kernel sees and compares Marlin's bf16 output against a Python FP32 oracle that matches the kernel's `reference()` function bit-for-bit:

```bash
CUDA_VISIBLE_DEVICES=2 /home/xinhaocheng/vllm-venv/bin/python3 \
    ~/repos/ferret/correctness/compare_marlin_vs_ref.py
```

Measured on catalyst B200:

```
                  v049 kernel       Marlin
T=1   maxae       0.0232            0.0064       (Marlin 3.6× tighter)
T=1   maxre       1.57              2.00         (both fail re<5e-3 — spec is unsatisfiable)
T=8   maxae       0.0217            0.0091       (Marlin 2.4× tighter)
T=8   maxre       1.27              2.00
```

The `maxre ≈ 2.0` is **structural**: silu(very_negative)·up ≈ 0, so any non-zero residue on near-zero outputs blows up the relative metric. **Both** Marlin and v049 have it; the relative gate is unsatisfiable for BF16 GEMM at K=7168. The `maxae` is the meaningful, achievable target.

## What's already been ruled out (DON'T REPEAT)

The previous investigation attempted the obvious fix and it **did not work**:

- **Replaced atomicAdd split-K reduction with a deterministic per-slice FP32 reduce** (equivalent to Marlin's `use_fp32_reduce=True` path). Latency improved 3–6% from removing atomicAdd contention. **`maxae` did NOT change** — identical to v049 (0.0232/0.0217).
- **Critical proof:** T=8 uses `k_split=1` (no atomicAdd at all) and still has `maxae = 0.0217`. So the noise source is **NOT** the split-K reduction.

Don't repeat this experiment.

## Where to look next (ranked by suspicion)

1. **K-tile boundary precision (highest suspect)**: the kernel iterates `BLOCK_K=128` then calls `tcgen05.mma` to accumulate into FP32 TMEM. If any intermediate register or SMEM hop drops to BF16/FP16 before re-accumulating, that's the leak. Compare SASS of v049's K-iteration body to Marlin's MMA inner loop:

   ```bash
   /usr/local/cuda/bin/cuobjdump --dump-sass ~/repos/ferret/workspace/kernel \
     | grep -A 80 'moe_kernel' \
     | grep -E 'HMMA|HADD|HFMA|FFMA|F2FP|FRND' | head -50
   ```

   The hot path should be `UHMMA...F32.BF16.BF16` (FP32 accumulator). Look for any HMMA/HFMA where there shouldn't be one in the K-accumulation.

2. **SMEM round-trip of the dequanted BF16 (medium)**: after `dequant_int4_bf16_scaled` (~line 89), the BF16 weights go to SMEM via `st.shared.b64`, then MMA reads them back. Marlin (see `resources/vllm-csrc-marlin/csrc/libtorch_stable/quantization/marlin/marlin_template.h`) likely keeps the dequant fragment in registers and never round-trips through SMEM. If so, that round-trip adds a BF16 store→load with attendant rounding noise. Port Marlin's no-SMEM-round-trip approach if architecturally permitted.

3. **Activation accumulation order (low)**: `BLOCK_K=128` vs Marlin's likely smaller K-tile means fewer / larger FP32 accumulator updates. The number of accumulator round-trips through TMEM could matter.

4. **`__hfma2` corner case in dequant (low prob)**: the `EX`/`ADD` constants produce `bf16(nib - 8)` exactly for nib ∈ {0..15}. Verified bit-identical to the Python oracle. Probably not the source — but worth one printf at a single (e, k, n) to confirm.

5. **MMA atom internal accumulator (very low prob)**: `tcgen05.mma.kind::f16` is documented as FP32-accumulating, and `tcgen05.ld` returns float (= FP32). Almost certainly FP32 already.

## Definition of success

The fix is done when:

1. `./kernel` reports `T1: maxae < 0.012` AND `T8: maxae < 0.012`.
2. `T1 us` stays within 2% of v049's number (≤ 20.2 μs).
3. `T8 us` stays within 2% of v049's number (≤ 36.3 μs).
4. Latency speedup over Marlin stays ≥ 1.30× (current v049: 1.37×).
5. All other code (architecture, schedule, dequant trick, silu epilogue, grouped-MoE dispatch) is structurally unchanged — your diff should be small and localized to the suspected leak point.

If you cannot reach `maxae < 0.012` without regressing latency, document exactly what was tried, the maxae achieved, and the latency cost, and stop. Don't ship a slower kernel for marginal precision gains.

## What to ignore

- Anything about INT4 tensor cores. `tcgen05.mma` has NO INT4 atom on SM100; BF16-dequant + BF16-MMA is the only path. Hard PTX ISA fact.
- The old `re<5e-3 AND ae<1.0` validator from earlier versions of the spec. It is unsatisfiable for any BF16 W4A16 GEMM at K=7168 (Marlin itself has maxre ≈ 2.0). Don't waste budget on it.

## TL;DR

> v049 passes the correctness gate but is 3–4× noisier than Marlin (maxae 0.023 vs 0.008). Likely cause: K-iteration boundary precision or SMEM round-trip of dequanted BF16. Patch the leak surgically, keep the architecture and the 1.37× latency win. Target: `maxae < 0.012` on both T=1 and T=8.
