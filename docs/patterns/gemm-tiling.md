# GEMM Tiling Patterns

## Why Tiling

Naive GEMM: each output element reads a full row of A and column of B from DRAM. Tiled GEMM: load tiles into shared memory, reuse across threads. Data reuse ratio = TILE_SIZE → reduces DRAM traffic proportionally.

## Tiling Hierarchy

```
Grid level:  Blocks tile the output C matrix
             Each block → CTA tile (e.g., 128×256)

Block level: Warps within a block tile the CTA tile
             Each warp → Warp tile (e.g., 64×64)

Warp level:  Tensor core instructions tile the warp tile
             Each MMA → Instruction tile (e.g., 16×8×16)
```

## Real Example: CUTLASS CuTe SM80 GEMM

From `cutlass-4.4.2/examples/cute/tutorial/sgemm_sm80.cu`:

### Tiling Parameters

```cpp
// CTA tile sizes
auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int< 64>{};
auto cta_tiler = make_shape(bM, bN, bK);
auto bP = Int<3>{};  // Pipeline depth = 3 stages

// Swizzled shared memory layout (bank-conflict-free, see docs/patterns/swizzling.md)
auto swizzle_atom = composition(Swizzle<3,3,3>{},
                                Layout<Shape <_8,Shape <_8, _8>>,
                                       Stride<_8,Stride<_1,_64>>>{});
auto sA = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
auto sB = tile_to_shape(swizzle_atom, make_shape(bN, bK, bP));

// MMA atom: SM80 16x8x16 tensor core, tiled 2x2 = 32x32x16
TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                               Layout<Shape<_2,_2>>{},     // 2x2 MMA atoms
                               Tile<_32,_32,_16>{});        // 32x32x16 per warp

// Global→shared copy via cp.async (128-bit loads)
TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
                                  Layout<Shape<_16,_8>, Stride<_8,_1>>{},
                                  Layout<Shape<_1,_8>>{});

// Shared→register copy via ldmatrix
Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom;
```

### Data Flow: Global → Shared → Registers → Tensor Core

```cpp
// 1. Full tensors in global memory
Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);  // (M,K)
Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);  // (N,K)

// 2. CTA-level tile from global
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)

// 3. Shared memory (swizzled, with pipeline dimension)
Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);    // (BLK_M,BLK_K,PIPE)

// 4. Partition copies across threads
ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
Tensor tAgA = thr_copy_a.partition_S(gA);    // thread's view of global A
Tensor tAsA = thr_copy_a.partition_D(sA);    // thread's view of shared A

// 5. Register fragments for MMA
ThrMMA thr_mma = mma.get_slice(threadIdx.x);
Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));   // (MMA,MMA_M,MMA_K)
Tensor tCrC = thr_mma.make_fragment_C(tCgC);             // accumulator in registers
```

### Pipelined Mainloop

```cpp
// 3-stage pipeline: while computing on stage[read], load stage[write]
while (k_tile_count > -(K_PIPE_MAX-1)) {
  for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
    if (k_block == K_BLOCK_MAX - 1) {
      cp_async_wait<K_PIPE_MAX-2>();         // wait for next stage
      __syncthreads();
    }
    // Load shared→registers for next k_block (ldmatrix)
    copy(s2r_atom, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
    copy(s2r_atom, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));
    // Issue cp.async for next k_tile (global→shared)
    if (k_block == 0) {
      copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
      cp_async_fence();
    }
    // Tensor core MMA on current k_block
    gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
  }
}
```

**Summary:** CTA=128×128×64, MMA=16×8×16 (tiled 2×2 to 32×32×16), Pipeline=3 stages. Data flows: Global →(cp.async)→ Shared (swizzled) →(ldmatrix)→ Registers →(mma)→ Accumulator.

## Real Example: ThunderKittens H100 GEMM

From `thunderkittens-main/kernels/gemm/bf16_h100/bf16_h100_gemm.cu`:

```cpp
template<int M_BLOCK=2, int N_BLOCK=4, int SUPER_M=12>
struct matmul_template {
    using base_tile = st_bf<64, 64>;                  // 64×64 shared tile in bf16

    struct input_block  { base_tile a[M_BLOCK], b[N_BLOCK]; };   // shared memory per stage
    struct consumer_state { rt_fl<16, N_BLOCK*64> accum; };      // register accumulator

    static constexpr int INPUT_PIPE_STAGES = 4;       // 4-stage pipeline
    static constexpr int NUM_CONSUMER_WARPS = M_BLOCK * 4;

    // Producer: TMA loads (one thread per warp)
    struct producer {
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for (int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.coord.x+i, args.iter}, args.inputs_arrived);
                for (int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.iter, args.coord.y+i}, args.inputs_arrived);
            }
        }
    };

    // Consumer: WGMMA from shared memory
    struct consumer {
        __device__ static void setup(...) {
            warpgroup::increase_registers<232>();       // claim registers for accumulator
            zero(args.state.accum);
        }
        __device__ static void compute(...) {
            warpgroup::mma_AB(args.state.accum,       // register accumulator
                              args.input.a[warpgroup::groupid()],  // A from shared
                              reinterpret_cast<wide_tile&>(args.input.b));  // B from shared
            warpgroup::mma_async_wait();
        }
    };
};
```

**Summary:** CTA=128×256 (2×64 × 4×64), base_tile=64×64 bf16. Pipeline=4 stages. Producer warpgroup loads via TMA, consumer warpgroups compute via WGMMA from shared memory. Register accumulators `rt_fl<16, 256>` (16 rows × 256 cols in FP32). SUPER_M=12 for L2 swizzling.

## Real Example: DeepGemm FP8 GEMM (Hopper)

From `deepgemm-2.1.1/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh`:

```cpp
// WGMMA atom: 64×N×32 for FP8 (e4m3 × e4m3 → FP32)
// CTA tile: BLOCK_M × BLOCK_N × 128 (BLOCK_K=128 for per-128-channel FP8 scaling)
// Each k-block: 128/32 = 4 WGMMA instructions

for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
    // Wait for TMA to fill shared memory
    full_barriers[stage_idx]->wait(phase);

    // Read per-row/per-col FP8 scale factors from shared memory
    auto scale_a = ld_shared(smem_sfa[stage_idx] + row_idx);
    auto scales_b = ld_shared(reinterpret_cast<float2*>(smem_sfb[stage_idx] + col_idx));

    // Issue 4 WGMMAs per k-block (128/32=4)
    warpgroup_arrive();
    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++k) {
        auto desc_a = make_smem_desc(smem_a[stage_idx] + ..., 1);
        auto desc_b = make_smem_desc(smem_b[stage_idx] + ..., 1);
        WGMMA::wgmma(desc_a, desc_b, accum, k);
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    empty_barrier_arrive(stage_idx);

    // Promote with FP8 scales: final_accum += scale_a * scale_b * accum
    for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++i) {
        final_accum[i*4+0] += scale_a_0 * scales_b[i].x * accum[i*4+0];
        final_accum[i*4+1] += scale_a_0 * scales_b[i].y * accum[i*4+1];
        // ...
    }
}
```

**Summary:** WGMMA=64×N×32 (FP8), BLOCK_K=128 (4 WGMMAs per k-block). Split TMA/math warps: 128 TMA threads load A, B, and FP8 scales; math warpgroups issue WGMMA from shared memory descriptors. After each k-block, accumulators promoted with per-row/per-col FP8 scaling factors. Persistent scheduling.

## Choosing Tile Sizes

| Constraint | Rule |
|---|---|
| Shared memory | `(TILE_M × TILE_K + TILE_K × TILE_N) × bytes × stages ≤ smem limit` |
| Occupancy | Larger tiles = more smem → fewer blocks per SM |
| Register pressure | Accumulator size = `TM × TN` floats per thread |
| Warp efficiency | Tile dimensions should be multiples of 32 (warp size) |
| Tensor cores | Tile dimensions must be multiples of MMA tile size (16 for SM80, 64 for WGMMA) |
| Wave quantization | Grid dimensions should evenly divide into SM count |

Typical configs from the reference implementations:

| Architecture | CTA Tile | MMA Tile | Pipeline | Source |
|---|---|---|---|---|
| SM80 (A100) | 128×128×64 | 16×8×16 (2×2) | 3 stages | CUTLASS sgemm_sm80.cu |
| SM90 (H100) | 128×256×64 | 64×N×16 (WGMMA) | 4 stages | ThunderKittens bf16_h100 |
| SM90 (H100) FP8 | BLOCK_M×BLOCK_N×128 | 64×N×32 | multi-stage | DeepGemm sm90_fp8 |

