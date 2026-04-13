# Resource ↔ Topic Mapping

Which resources are relevant for which topics. Not exhaustive — grep `resources/` for more. If you discover a new connection, add it here.

## Versions

Each resource is a git submodule pinned to a specific commit / tag. The directory name carries the pinned version suffix so the agent can reason about it without inspecting `.gitmodules`. To bump:

```bash
cd resources/<name>
git fetch --tags
git checkout <new-ref>
cd ../..
git mv resources/<old-name> resources/<new-name>   # if the version suffix changes
git add resources/<new-name>
# then update every reference in docs/ and tools/doc_loader.py to the new suffix
python scripts/check_resource_refs.py              # verify nothing is stale
```

`scripts/check_resource_refs.py` walks all `*.md` + `*.py` under the repo and verifies every `resources/<name>/<path>` reference resolves to an existing file in the pinned submodule. It also detects drift between `.gitmodules` and on-disk directories (renamed, orphaned, uncloned). Run it after any submodule change; exits 1 on any issue.

Current pins:

| Dir | Upstream | Ref |
|---|---|---|
| `cutlass-4.4.2` | github.com/NVIDIA/cutlass | `v4.4.2` |
| `deepgemm-2.1.1.post3` | github.com/deepseek-ai/DeepGEMM | `v2.1.1.post3` |
| `documentsass` | github.com/0xD0GF00D/DocumentSASS | main (pinned commit) |
| `flash-attention-fa4-v4.0.0.beta8` | github.com/Dao-AILab/flash-attention | `fa4-v4.0.0.beta8` |
| `flashinfer-0.6.7.post3` | github.com/flashinfer-ai/flashinfer | `v0.6.7.post3` |
| `flashmla-main` | github.com/deepseek-ai/FlashMLA | main (pinned commit) |
| `nccl-2.29.7-1` | github.com/NVIDIA/nccl | `v2.29.7-1` |
| `tensorrt-llm-1.2.0` | github.com/NVIDIA/TensorRT-LLM | `v1.2.0` |
| `thunderkittens-main` | github.com/HazyResearch/ThunderKittens | main (pinned commit) |
| `triton-3.6.0` | github.com/triton-lang/triton | `v3.6.0` |

## By Topic → Resources

### Attention Kernels
- `cutlass-4.4.2` — `examples/88_hopper_fmha/`, `examples/77_blackwell_fmha/`, `examples/41_fused_multi_head_attention/`, Python CuTeDSL: `hopper/fmha.py`, `blackwell/fmha.py`, `blackwell/mla.py`
- `flash-attention-fa4-v4.0.0.beta8` — `hopper/flash_fwd_kernel_sm90.h` (Hopper), `hopper/flash_fwd_kernel_sm80.h` (Ampere), `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`
- `flashinfer-0.6.7.post3` — `include/flashinfer/attention/` (20+ variants), `include/flashinfer/attention/hopper/`, `include/flashinfer/attention/blackwell/`, `include/flashinfer/attention/mla.cuh`, prefill/decode/persistent variants
- `flashmla-main` — `csrc/sm90/decode/dense/splitkv_mla.cuh`, `csrc/sm100/decode/`, `csrc/sm100/prefill/`
- `tensorrt-llm-1.2.0` — `cpp/tensorrt_llm/kernels/flashMLA/` (FP8/BF16/FP16 variants), `cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/`, `cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/`
- `thunderkittens-main` — `kernels/attention/mha_h100/`, `kernels/attention/bf16_b300_mha_causal/`, `kernels/attention/bf16_b300_mha_noncausal/`, `kernels/linear_attention/`
- `deepgemm-2.1.1.post3` — `deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh`, `sm100_fp8_paged_mqa_logits.cuh`
- `triton-3.6.0` — `python/tutorials/06-fused-attention.py`

### GEMM / Matrix Multiply
- `cutlass-4.4.2` — `include/cutlass/gemm/kernel/` (40+ variants), CuTe GEMM tutorials, collective mainloops
- `deepgemm-2.1.1.post3` — `deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh`, `sm90_fp8_gemm_*.cuh`, `sm100_bf16_gemm.cuh`, `sm100_fp8_gemm_*.cuh`
- `tensorrt-llm-1.2.0` — `cpp/tensorrt_llm/thop/fp8PerTensorScalingTrtllmGenGemm.cpp`, `fp8BlockScalingGemm.cpp`, `fp4Gemm.cpp`, `weightOnlyQuantGemm.cpp`
- `flashinfer-0.6.7.post3` — `include/flashinfer/gemm/tgv_gemm.cuh`, `group_gemm_fp8_groupwise_sm100.cuh`, `group_gemm_mxfp4_groupwise_sm100.cuh`
- `thunderkittens-main` — `kernels/gemm/fp8_h100/`, `kernels/gemm/bf16_h100/`, `kernels/gemm/fp8_b200/`, `kernels/gemm/bf16_b200/`, `kernels/gemm/mxfp8_b200/`, `kernels/gemm/nvfp4_b200/`
- `triton-3.6.0` — `python/tutorials/03-matrix-multiplication.py`, `python/tutorials/09-persistent-matmul.py`

### TMA (Tensor Memory Accelerator)
- `cutlass-4.4.2` — `include/cute/arch/copy_sm90_tma.hpp`, `include/cutlass/pipeline/sm90_pipeline.hpp`, `include/cutlass/pipeline/sm100_pipeline.hpp`
- `flash-attention-fa4-v4.0.0.beta8` — `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`
- `flashinfer-0.6.7.post3` — Hopper/Blackwell attention kernels
- `flashmla-main` — `csrc/sm100/prefill/dense/collective/*_tma_warpspecialized.hpp`
- `thunderkittens-main` — `include/ops/thread/memory/tile/tma.cuh`, `include/ops/group/util/tma_cluster.cuh`

### Warp Specialization / Async Pipeline
- `flash-attention-fa4-v4.0.0.beta8` — `hopper/flash_fwd_kernel_sm90.h` (producer/consumer warps, register redistribution), `hopper/named_barrier.hpp`, `hopper/sm90_pipeline_no_cluster.hpp`
- `cutlass-4.4.2` — `include/cutlass/pipeline/sm90_pipeline.hpp` (PipelineTmaAsync), `include/cutlass/pipeline/sm100_pipeline.hpp` (PipelineUmmaAsync), `include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp`
- `flashmla-main` — Warp specialization in SM90/SM100 MLA kernels

### Tensor Cores (MMA/WGMMA/tcgen05)
- `cutlass-4.4.2` — `include/cute/arch/mma_sm80.hpp` (SM80 mma.sync), `include/cute/arch/mma_sm90_gmma.hpp` (SM90 wgmma), `include/cute/arch/mma_sm100.hpp` (SM100 tcgen05), `include/cute/atom/mma_traits_sm90_gmma.hpp`
- `flash-attention-fa4-v4.0.0.beta8` — WGMMA usage in Hopper attention kernels
- `thunderkittens-main` — `include/ops/thread/mma/tcgen05.cuh` (Blackwell), tile MMA abstractions

### Swizzling / Shared Memory Layout
- `cutlass-4.4.2` — `include/cute/swizzle.hpp`, `include/cute/swizzle_layout.hpp`, `include/cute/atom/mma_traits_sm90_gmma.hpp` (GMMA swizzle atoms), `examples/50_hopper_gemm_with_epilogue_swizzle/`, `examples/cute/tutorial/sgemm_sm80.cu`

### FP8 / FP4 / Quantization
- **Start here for FP4**: `docs/patterns/blockscaled-mma.md` — PTX syntax, scale layout, TMEM, instruction descriptor, kernel skeleton
- `deepgemm-2.1.1.post3` — FP8 GEMM with per-block scaling, two-level accumulation (SM90 + SM100)
- `tensorrt-llm-1.2.0` — `cpp/tensorrt_llm/thop/fp8*.cpp`, `fp4*.cpp`, `weightOnlyQuantGemm.cpp`
- `flashinfer-0.6.7.post3` — FP8 GEMM, FP4 quantize, MXFP8 in `tests/gemm/`, `include/flashinfer/gemm/`
- `thunderkittens-main` — `kernels/gemm/fp8_h100/`, `kernels/gemm/mxfp8_b200/`, `kernels/gemm/nvfp4_b200/`
- `cutlass-4.4.2` — `include/cutlass/float8.h`, `examples/72_blackwell_narrow_precision_gemm/`, FP8 GEMM tests
- `flashmla-main` — `csrc/sm90/decode/sparse_fp8/` (FP8 sparse decoding)

### Multi-GPU / NCCL / Multimem
- `nccl-2.29.7-1` — `src/collectives/`, `src/device/`, `docs/examples/06_device_api/`
- `cutlass-4.4.2` — `examples/python/CuTeDSL/distributed/` (multimem two-shot allreduce, TMA allreduce), `include/cutlass/experimental/distributed/`
- `flashinfer-0.6.7.post3` — `include/flashinfer/comm/`
- `tensorrt-llm-1.2.0` — `cpp/tensorrt_llm/runtime/ipcNvlsMemory.cu`, `cpp/tensorrt_llm/kernels/allReduce/`, `cpp/tensorrt_llm/plugins/ncclPlugin/`
- `thunderkittens-main` — `kernels/parallel/ag_gemm/`, `kernels/parallel/gemm_rs/`, `kernels/parallel/ring_attn/`, `kernels/parallel/ulysses_attn/`, `kernels/parallel/all_to_all/`

### Softmax / Reduction / LayerNorm
- `flash-attention-fa4-v4.0.0.beta8` — `hopper/softmax.h` (online softmax), in fwd/bwd mainloops
- `cutlass-4.4.2` — `include/cutlass/reduction/kernel/reduce_softmax_final.h`, `examples/37_gemm_layernorm_gemm_fusion/`
- `triton-3.6.0` — `python/tutorials/02-fused-softmax.py`, `python/tutorials/05-layer-norm.py`
- `thunderkittens-main` — `kernels/layernorm/`
- `flashinfer-0.6.7.post3` — `include/flashinfer/norm.cuh`
- `deepgemm-2.1.1.post3` — `deep_gemm/include/deep_gemm/common/reduction.cuh`

### Inline PTX / SASS
- `cutlass-4.4.2` — `include/cute/arch/` (PTX wrappers for every operation), `include/cutlass/arch/` (barrier, memory, register reconfig)
- `flash-attention-fa4-v4.0.0.beta8` — `hopper/named_barrier.hpp`, `hopper/utils.h`, `csrc/flash_attn/src/utils.h` (conversion, ReLU via PTX)
- `flashinfer-0.6.7.post3` — `include/flashinfer/utils.cuh`, `include/flashinfer/mma.cuh`, `include/flashinfer/pos_enc.cuh`
- `thunderkittens-main` — `include/ops/thread/util/util.cuh`, `include/ops/thread/mma/tcgen05.cuh`
- `deepgemm-2.1.1.post3` — `deep_gemm/include/deep_gemm/common/utils.cuh`, `common/sm90_utils.cuh`
- `documentsass` — Instruction encodings and latencies extracted from nvdisasm

### Triton Kernel Development
- `triton-3.6.0` — Compiler, all tutorials, Proton profiler
- `deepgemm-2.1.1.post3` — Production Triton FP8 GEMM
- `flashinfer-0.6.7.post3` — `flashinfer/triton/kernels/`

### CuTe / CUTLASS Framework
- `cutlass-4.4.2` — `include/cute/`, `include/cutlass/`, `media/docs/cpp/cute/`, `examples/`

### Cluster Launch Control (CLC) / Work-Stealing Scheduler
Hardware dynamic tile distribution via `clusterlaunchcontrol.try_cancel` PTX (Blackwell SM100+).
- `cutlass-4.4.2` — `media/docs/cpp/blackwell_cluster_launch_control.md` (conceptual overview), `include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp` (production scheduler implementation)
- `thunderkittens-main` — `include/ops/group/util/util.cuh` (raw PTX inline-asm wrappers for try_cancel/query_cancel, ~90 lines total, copy-pastable)
- `docs/cuda-programming-guide/12-cluster-launch-control.md` (NVIDIA CLC spec, includes vector-scalar example kernel)
- `docs/ptx-isa-9.2/09-instruction-set/parallel-synchronization-and-communication-instructions.md` (PTX instruction reference)
- `docs/patterns/threadblock-clusters.md` (CLC section starts at "CLC with Clusters")

### Cluster-cooperative MMA (cta_group::2)
Two SMs cooperatively execute a single MMA via DSMEM, effectively doubling M per instruction (Blackwell SM100+).
- `deepgemm-2.1.1.post3` — `deep_gemm/include/deep_gemm/common/sm100_utils.cuh` (struct `SM100_MMA_F16BF16_2x1SM_SS` has the `tcgen05.mma.cta_group::2.kind::f16` inline PTX; also `MXF8F6F4_2x1SM_SS` variant)
- `deepgemm-2.1.1.post3` — `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh` (full kernel using the 2x1SM MMA)
- `cutlass-4.4.2` — `include/cute/atom/mma_traits_sm100.hpp`, `include/cute/arch/mma_sm100_umma.hpp` (CuTe abstractions)

### TMA Multicast (cluster broadcast)
Single TMA load delivers data to shared memory of multiple CTAs in the cluster via `cp.async.bulk.tensor...multicast::cluster`.
- `cutlass-4.4.2` — `include/cute/arch/copy_sm100_tma.hpp` (multicast TMA copy wrappers)
- `thunderkittens-main` — `include/ops/group/util/tma_cluster.cuh` (cluster-scope barrier + multicast bytes, ~120 lines)
- `deepgemm-2.1.1.post3` — `deep_gemm/include/deep_gemm/common/sm100_utils.cuh` (multicast expect-bytes utilities)
- `docs/patterns/threadblock-clusters.md` (TMA multicast section)

### Blackwell (SM100) Specific
- `cutlass-4.4.2` — `include/cute/arch/mma_sm100.hpp`, `include/cute/atom/mma_traits_sm100.hpp`, `include/cutlass/pipeline/sm100_pipeline.hpp`, `examples/python/CuTeDSL/`
- `flashinfer-0.6.7.post3` — `include/flashinfer/attention/blackwell/`, `include/flashinfer/gemm/group_gemm_*_sm100.cuh`
- `flashmla-main` — `csrc/sm100/`
- `thunderkittens-main` — `kernels/attention/bf16_b300_*`, `kernels/gemm/*_b200/`
- `deepgemm-2.1.1.post3` — `deep_gemm/include/deep_gemm/impls/sm100_*.cuh`

## By Resource → Topics

| Resource | Topics Covered |
|---|---|
| `cutlass-4.4.2` | Attention, GEMM, TMA, warp specialization, tensor cores, swizzling, FP8, inline PTX, CuTe, multi-GPU (minimal), softmax/reduction, Blackwell |
| `flash-attention-fa4` | Attention, TMA, warp specialization, async pipeline, inline PTX, online softmax |
| `flashinfer-0.6.7.post3` | Attention, GEMM, softmax/norm, multi-GPU (allreduce), FP8/FP4, TMA, inline PTX, Triton, Blackwell |
| `flashmla-main` | Attention (MLA), TMA, warp specialization, FP8 sparse, Blackwell |
| `deepgemm-2.1.1.post3` | GEMM (FP8), attention (MQA logits), reduction, inline PTX, Triton, Blackwell |
| `thunderkittens-main` | Attention, GEMM (all precisions), layernorm, multi-GPU (ring/ulysses attn, AG+RS), FP8/MXFP8/NVFP4, TMA, tcgen05, Blackwell |
| `triton-3.6.0` | Triton compiler, tutorials (attention, GEMM, softmax, layernorm, dropout) |
| `nccl-2.29.7-1` | Multi-GPU collectives, device API examples |
| `tensorrt-llm-1.2.0` | Attention (MHA, MLA, FlashMLA), GEMM (FP8/FP4/INT8), multi-GPU (NCCL, NVLS), quantization |
| `documentsass` | SASS instruction encodings, latencies |

## Maintenance

This mapping is not exhaustive. If you discover a relevant file while working, add it here.
