# Knowledge Base Index

## "I want to learn about X" → Read these files

### GPU Architecture
- **Ampere A100**: `architecture/ampere-a100.md` — SM80, cp.async, L2 residency, TF32, 2:4 sparsity, MIG
- **Hopper H100**: `architecture/hopper-h100.md` — SM90, TMA, clusters, DSMEM, wgmma, warp specialization
- **Blackwell B200**: `architecture/blackwell-b200.md` — SM100, tcgen05, TMEM, FP4, SM100 vs SM120 differences
- **GPU specs table**: `profiling/gpu-specs.md` — Peak TFLOPS/BW per GPU, shared memory/register limits

### Writing CUDA Kernels
- **CUDA language**: `cuda-programming-guide/` — 27 chapters (memory model, extensions, compute capabilities)
- **Best practices**: `cuda-best-practices-guide/` — 21 chapters (memory optimization, execution config)
- **PTX ISA**: `ptx-isa-9.2/` — 35 files (all instructions, registers, directives)
- **Inline PTX**: `patterns/inline-ptx.md` — asm syntax, constraints, real examples from CUTLASS/FlashAttention

### Kernel Optimization Patterns
- **Memory access**: `patterns/memory-access.md` — coalescing, bank conflicts, L2 cache, TMA, TMEM
- **Double buffering**: `patterns/double-buffering.md` — software pipelining, multi-stage, warp specialization
- **GEMM tiling**: `patterns/gemm-tiling.md` — basic tiling, register blocking, tensor cores, CUTLASS hierarchy
- **Swizzling**: `patterns/swizzling.md` — XOR math, Swizzle<B,M,S>, SM80/SM90/SM100 configs
- **Warp primitives**: `patterns/warp-primitives.md` — shuffle, vote, match, reduce, cooperative groups
- **Reductions**: `patterns/reduction.md` — warp/block/grid, online softmax, Welford variance
- **Operator fusion**: `patterns/operator-fusion.md` — why/when/how, common fusion patterns
- **Occupancy tuning**: `patterns/occupancy-tuning.md` — register/smem limits, launch bounds
- **Async execution**: `patterns/async-execution.md` — producer-consumer, TMA+mbarrier, wgmma, pitfalls
- **BF16 tcgen05 MMA (SM100)**: `patterns/bf16-tcgen05-mma.md` — tcgen05.mma.kind::f16 PTX wrappers, SMEM descriptor encoding, TMEM alloc/load/store, instruction descriptor builder, fences, complete GEMM tile example, MLA decode TMEM layout
- **Blockscaled MMA (FP4)**: `patterns/blockscaled-mma.md` — tcgen05.mma.blockscaled PTX syntax, NVFP4/MXFP4 scale factor layout, TMEM scale storage, instruction descriptor encoding, 2-CTA cluster pattern, kernel skeleton
- **Thread block clusters**: `patterns/threadblock-clusters.md` — cluster launch, DSMEM, TMA multicast, cluster barriers, CLC persistent scheduling, deadlock pitfalls, when clusters help vs hurt
- **Warp specialization**: `patterns/warp-specialization.md` — Hopper vs Blackwell, register redistribution, pingpong scheduling
- **Persistent kernels**: `patterns/persistent-kernels.md` — SM-count grid, tile schedulers, pipeline continuity
- **Grid swizzling**: `patterns/grid-swizzling.md` — L2 tile ordering, GROUP_SIZE_M, SUPER_M
- **SASS reading**: `sass/sass-reading.md` — cuobjdump, instruction mnemonics, control codes, analysis
- **SASS instruction sets**: `sass/instruction-sets.md` — complete instruction tables per arch (Turing, Ampere, Hopper, Blackwell)
- **CUDA binary utilities**: `build/cuda-binary-utilities.md` — cuobjdump, nvdisasm, cu++filt, nvprune reference

### Libraries & Frameworks
- **CUTLASS/CuTe**: `cutlass-tutorial/cute-cutlass-guide.md` — layouts, tensors, MMA atoms, CUTLASS 3.x
- **Triton**: `triton-guide/triton-guide.md` — block programming model, API, autotune, patterns

### Quantization
- **FP8/FP4**: `quantization/fp8-fp4-guide.md` — E4M3/E5M2, NVFP4, MXFP8, scaling strategies, GEMM mechanics

### Multi-GPU
- **NCCL**: `multi-gpu/nccl-guide.md` — all collectives, P2P, RMA, group API, env vars, tuning
- **Multimem/NVLS**: `multi-gpu/multimem-nvls.md` — multimem PTX instructions, TMA multicast, two-shot all-reduce

### Profiling & Analysis
- **ncu guide**: `profiling/ncu-guide.md` — CLI flags, kernel filtering, CSV format, recipes
- **ncu metrics**: `profiling/ncu-metrics.md` — all metric names, stall reasons, SOL interpretation
- **nsys guide**: `profiling/nsys-guide.md` — commands, trace types, SQLite schema, stats reports
- **Roofline**: `profiling/roofline-analysis.md` — arithmetic intensity, bound classification, optimization
- **Baselines**: `baselines/cublas-cudnn-baselines.md` — cuBLAS/cuDNN performance per GPU

### Build & Debug
- **nvcc**: `build/nvcc-build-guide.md` — compiler flags, architectures, CMake, torch integration
- **compute-sanitizer**: `build/compute-sanitizer.md` — memcheck, racecheck, initcheck, synccheck
- **cuda-gdb**: `build/cuda-gdb.md` — debugging cheat sheet

## Code Repos (resources/)

| Repo | Focus |
|---|---|
| `cutlass-4.4.2` | GEMM templates, CuTe layouts |
| `flash-attention-fa4-v4.0.0.beta4` | Attention kernels (Ampere + Hopper) |
| `flashinfer-0.6.7` | Inference attention, PagedKV, allreduce |
| `flashmla-main` | DeepSeek MLA kernel |
| `deepgemm-2.1.1` | DeepSeek FP8 GEMM with fine-grained scaling |
| `thunderkittens-main` | Tile-based CUDA kernel framework |
| `triton-3.6.0` | GPU compiler, tutorials, Proton profiler |
| `nccl-2.29.7` | Multi-GPU collective communication |
| `tensorrt-llm-1.2.0` | LLM inference kernels (cpp/ only) |
