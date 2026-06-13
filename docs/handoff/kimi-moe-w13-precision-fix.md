# Handoff brief — fix the precision gap in Kimi MoE w13 INT4 kernel

**For a successor agent working under the ferret repo. Paths are relative to `~/repos/ferret/`.**

## Mission (one paragraph)

The kernel at `examples/kimi-moe-w13-int4/v040_swapab_n8_int4_silu_1.29x_min.cu` (and its descendants up to **v049** sitting in `workspace/kernel.cu`) is **1.37× faster than Marlin** for Kimi K2.6/K2.7 MoE w13 + silu·mul on B200, but its output is **3–4× noisier than Marlin** against the FP32 dequant→BF16 reference. Both pass the loose `maxae < 0.05` validator, but Marlin reaches `maxae ≈ 0.006–0.009` while this kernel sits at `maxae ≈ 0.022–0.023`. Your job: **close that precision gap WITHOUT changing the kernel's main architecture or losing the latency win**. Target: `maxae < 0.012` at both T=1 and T=8.

## Hard constraints (read first, do not break)

- **DO NOT change the main architecture.** That means:
  - Keep `cta_group::1` with `swapAB` (tokens on N=8, weights on M=64).
  - Keep `BLOCK_M=64`, `BLOCK_N=8`, `BLOCK_K=128`, 5 pipeline stages, 10 warps (8 dequant + 1 TMA + 1 MMA).
  - Keep `tcgen05.mma.cta_group::1.kind::f16` BF16 MMA with FP32 accumulator.
  - Keep the fused silu·mul epilogue from TMEM.
  - Keep the INT4→BF16 dequant via the Marlin `__hfma2(v, MUL, ADD) * scale` trick.
- **DO NOT regress latency** by more than 2% on either config. Final min_ratio must stay ≥ 1.30 over Marlin.
- **DO NOT use cuBLAS / cuDNN / Marlin / vLLM / DeepGEMM / FlashInfer** in your kernel. You may read their source.
- **The K-split atomicAdd reduction is already ruled out as the cause** (proven below). Don't chase that lead.

## Files you need to know about

```
ferret/
├── examples/kimi-moe-w13-int4/
│   └── v040_swapab_n8_int4_silu_1.29x_min.cu     # frozen baseline artifact
├── workspace/                                     # current agent run state
│   ├── kernel.cu                                  # current kernel (latest tag v049)
│   ├── kernel                                     # compiled binary
│   └── progress.md                                # what the prior agent tried
├── baselines/kimi-moe-w13-int4/
│   └── baseline_marlin.py                         # invokes Marlin's ops.moe_wna16_marlin_gemm
├── tasks/
│   └── kimi-moe-w13-int4-grouped-gemm.yaml        # task spec (shapes, constraints)
├── resources/
│   ├── vllm-csrc-marlin/csrc/libtorch_stable/
│   │   ├── moe/marlin_moe_wna16/marlin_template.h          # the MoE Marlin source
│   │   └── quantization/marlin/marlin_template.h           # the underlying Marlin
│   └── marlin-standalone/marlin/marlin_cuda_kernel.cu      # IST-DASLab original
└── docs/handoff/kimi-moe-w13-precision-fix.md     # this file
```

The kernel's CPU `reference()` function (in `workspace/kernel.cu`, around line 405) is the **oracle** the validator compares against. It computes:

```
nv_bfloat16 w_bf16 = __float2bfloat16(__bfloat162float(__float2bfloat16((float)(nib-8))) * scale);
acc_fp32 += __bfloat162float(w_bf16) * __bfloat162float(activation_bf16);
// after K, FP32 silu(gate)·up, then cast to bf16
```

This is the **dequant→BF16 + FP32-accumulate + silu·mul** oracle. Same semantics as Marlin's correctness criterion.

## How to run the current kernel (single GPU)

```bash
# Compile (note: -gencode form, NOT -arch=sm_100a — sm_100 alone won't accept tcgen05)
cd ~/repos/ferret/workspace
/usr/local/cuda/bin/nvcc -O3 --use_fast_math -std=c++17 \
  -gencode arch=compute_100a,code=sm_100a \
  --expt-relaxed-constexpr \
  -I ~/repos/ferret/resources/cutlass-4.4.2/include \
  -o kernel kernel.cu -lcuda

# Run (use any idle GPU on catalyst; GPU 2 is usually free if the running agent has GPU 1)
CUDA_VISIBLE_DEVICES=2 ./kernel
```

Expected output:
```
Reference: T1=2.291 T8=9.175 TFLOPS
T1: 32 blocks, 8 active, P=8, smem=102616
  k_split=4 grid=128
T1: maxre=1.57 maxae=0.0232 errs=0/2048
T1: 19.81 us, 2.964 TFLOPS, bw=0.74 TB/s
T8: 248 blocks, 62 active, P=64, smem=102616
  k_split=1 grid=86
T8: maxre=1.27 maxae=0.0217 errs=0/16384
T8: 35.62 us, 13.19 TFLOPS, bw=3.19 TB/s
KERNEL_RESULT {"T1": 2.96, "T8": 13.19}
```

The kernel emits `maxae` against its own FP32 oracle. **That is the metric to drive down.**

## How to run Marlin for reference

```bash
# vllm 0.20.1 lives in /home/xinhaoc/vllm-venv on catalyst
CUDA_VISIBLE_DEVICES=2 /home/xinhaoc/vllm-venv/bin/python3 \
    ~/repos/ferret/baselines/kimi-moe-w13-int4/baseline_marlin.py
```

Expected output:
```
T1:   25.63 us   2.291 TFLOPS  weight_bw= 0.57 TB/s  (8 active experts)
T8:   51.20 us   9.175 TFLOPS  weight_bw= 2.29 TB/s  (64 active experts)
```

`baseline_marlin.py` calls `ops.moe_wna16_marlin_gemm + apply_moe_activation(SILU)` — same operation as the kernel.

## How to compare Marlin OUTPUT vs the FP32 oracle (the real precision check)

A second Python script (already on catalyst at `/tmp/compare_marlin_vs_ref.py`, see `docs/handoff/compare_marlin_vs_ref.py` in this repo for a copy) feeds Marlin the same INT4 weights / scales / activations the kernel sees and compares Marlin's output against an FP32 oracle that matches the kernel's `reference()` function bit-for-bit. Measured results on catalyst B200:

```
                   v040/v049 kernel       Marlin
T=1   maxae       0.0232                  0.0064       (Marlin 3.6× tighter)
T=1   maxre       1.99                    2.00         (both fail re<5e-3 — spec is unsatisfiable)
T=8   maxae       0.0217                  0.0091       (Marlin 2.4× tighter)
T=8   maxre       2.00                    2.00
```

The `maxre ≈ 2.0` is **structural**: silu(very_negative)·up ≈ 0 so the relative metric blows up on near-zero outputs. Both kernels have it. The `maxae` is the meaningful gap.

## What's already been ruled out

The previous investigation attempted the obvious fix and it **did not help**:

- **Replaced atomicAdd split-K reduction with deterministic per-slice FP32 reduce** (the equivalent of Marlin's `use_fp32_reduce=True` path). Latency improved 3–6% from removing atomicAdd contention. **`maxae` did NOT change.** Identical values to v049: 0.0232/0.0217.
- **Critical proof:** T=8 uses `k_split=1` (no atomicAdd at all) and still has `maxae = 0.0217`. So the noise source is **not** the split-K reduction.

Don't repeat this experiment.

## Where to look next (ranked by suspicion)

1. **K-tile boundary precision (highest)**: the kernel iterates `BLOCK_K=128` then calls `tcgen05.mma` to accumulate into FP32 TMEM. If any intermediate register or SMEM hop drops precision (BF16 intermediate before re-accumulating), that's the leak. Compare SASS of v049's K-iteration body to Marlin's corresponding section:
   ```bash
   /usr/local/cuda/bin/cuobjdump --dump-sass ~/repos/ferret/workspace/kernel | grep -A 80 'moe_kernel' | grep -E 'HMMA|HADD|HFMA|FFMA|F2FP|FRND' | head -50
   ```
   Look for any HMMA/HFMA in the K-accumulation path that shouldn't be there. The hot path should be `UHMMA.16832.F32.BF16.BF16` style (FP32 accumulator).

2. **SMEM staging of the dequanted BF16 (medium)**: after `dequant_int4_bf16_scaled` (around line 89 of v049), the BF16 weights go to SMEM via `st.shared.b64`. Inspect what Marlin does with the equivalent path in `resources/vllm-csrc-marlin/csrc/libtorch_stable/quantization/marlin/marlin_template.h`. Maybe Marlin keeps the dequanted weight in registers (frag_b) and never round-trips to SMEM, saving one round of bf16 store-precision noise. If so, port that.

3. **Activation accumulation order (low)**: `BLOCK_K=128` vs Marlin's likely smaller K-tile means fewer / larger FP32 accumulator updates. The number of accumulator round-trips through TMEM could matter. Lower priority.

4. **`__hfma2` corner case in dequant (low)**: the `EX`/`ADD` constants in `dequant_int4_bf16_scaled` produce `bf16(nib - 8)` exactly for nib ∈ {0..15}. Verified bit-identical to the Python oracle. Probably not the source — but worth one printf-of-individual-element check at a single (e, k, n) to confirm.

5. **MMA atom internal accumulator (very low)**: `tcgen05.mma.kind::f16` is documented as FP32-accumulating. If somehow the BF16 atom variant is being selected, that would explain a lot — but the `tcgen05.ld` results are bound to `=f` (float) so it's almost certainly FP32 already.

## Definition of success

The fix is done when:

1. `CUDA_VISIBLE_DEVICES=2 ./kernel` reports `T1: maxae < 0.012` AND `T8: maxae < 0.012`.
2. `T1 us` stays within 2% of the v049 number (≤ 20.2 μs).
3. `T8 us` stays within 2% of the v049 number (≤ 36.3 μs).
4. Latency speedup over Marlin stays ≥ 1.30× (current v049: 1.37×).
5. All other code (architecture, schedule, dequant trick, silu epilogue, grouped-MoE dispatch) is structurally unchanged — your diff should be small and localized to the suspected leak point.

If you cannot reach `maxae < 0.012` without losing latency, document exactly what was tried, the maxae achieved, and the latency cost, and stop. Don't ship a slower kernel for marginal precision gains without confirmation.

## What to ignore

- Anything about INT4 tensor cores. `tcgen05.mma` has NO INT4 atom on SM100; the BF16-dequant + BF16-MMA path is the only option. This is a hard fact in PTX ISA 9.x.
- The `re<5e-3 AND ae<1.0` validator from older versions of the spec. It is unsatisfiable for any BF16 W4A16 GEMM at K=7168 and was already replaced with `maxae < 0.05`. Do not waste budget on it.
- The watchdog at `scripts/run_forever.sh`. It's running the prior agent's session. Don't restart it unless you specifically need to.

## TL;DR for the new agent

> Kernel passes correctness gate. But the output is 3–4× noisier than Marlin (`maxae 0.023 vs 0.008`). Find the leak (probably K-iteration boundary or SMEM staging of dequanted BF16), patch it surgically, keep the architecture and the 1.37× latency win. Target: `maxae < 0.012` on both T=1 and T=8.
