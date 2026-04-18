// Qwen3-8B decode-phase linear projections at M=16 on B200
// Uses tcgen05 cg2 with swapab trick for small M
// Based on examples/tcgen05-gemm/{05b, 06, 07}
//
// C[M, N] = A[M, K] @ W[N, K]^T  (+ residual for O, Down)
// Via swapab: C^T[N, M] = W[N, K] @ A[K, M]^T
//   MMA "A" operand = W[N, K] (large, row-major)
//   MMA "B" operand = A^T stored as A[M, K] row-major (small)
//   MMA "C" result  = C^T[N, M_padded]

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cublas_v2.h>

// ── Helpers ──

constexpr int WARP_SIZE = 32;

__host__ __device__ inline constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

template <typename T>
__device__ inline T warp_uniform(T x) { return __shfl_sync(0xFFFFFFFF, x, 0); }

__device__ inline uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
    "{\n\t.reg .pred %%px;\n\t"
    "elect.sync _|%%px, %1;\n\t"
    "@%%px mov.s32 %0, 1;\n\t}"
    : "+r"(pred) : "r"(0xFFFFFFFF));
  return pred;
}

__device__ inline void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

__device__ inline void mbarrier_wait(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;
  asm volatile(
    "{\n\t.reg .pred P1;\n\t"
    "LAB_WAIT:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
    "@P1 bra.uni DONE;\n\t"
    "bra.uni LAB_WAIT;\n\t"
    "DONE:\n\t}"
    :: "r"(mbar_addr), "r"(phase), "r"(ticks));
}

__device__ inline void mbarrier_arrive_expect_tx(int mbar_addr, int size) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
              :: "r"(mbar_addr), "r"(size) : "memory");
}

__device__ inline void mbarrier_arrive(int mbar_addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" :: "r"(mbar_addr) : "memory");
}

template <int CTA_GROUP = 1>
__device__ inline void tma_3d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::%6 "
              "[%0], [%1, {%2, %3, %4}], [%5];"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "n"(CTA_GROUP)
              : "memory");
}

// TMA load with L2 cache hint
template <int CTA_GROUP = 1>
__device__ inline void tma_3d_gmem2smem_l2hint(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr, uint64_t cache_hint) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::%6.L2::cache_hint "
              "[%0], [%1, {%2, %3, %4}], [%5], %7;"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "n"(CTA_GROUP), "l"(cache_hint)
              : "memory");
}

template <int CTA_GROUP = 1>
__device__ inline void tcgen05_alloc(int smem_addr, int size) {
  asm volatile("tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;"
              :: "r"(smem_addr), "r"(size), "n"(CTA_GROUP));
}

template <int CTA_GROUP = 1>
__device__ inline void tcgen05_dealloc(int taddr, int size) {
  asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;"
              :: "r"(taddr), "r"(size), "n"(CTA_GROUP));
}

template <int CTA_GROUP = 1>
__device__ inline void tcgen05_mma_f16(int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc, int enable_input_d) {
  asm volatile(
    "{\n\t.reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::%5.kind::f16 [%0], %1, %2, %3, p;\n\t}"
    :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(enable_input_d), "n"(CTA_GROUP));
}

template <int CTA_GROUP = 1>
__device__ inline void tcgen05_commit_mcast(int mbar_addr, int16_t cta_mask) {
  asm volatile("tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
              :: "r"(mbar_addr), "h"(cta_mask), "n"(CTA_GROUP) : "memory");
}

__device__ inline constexpr uint64_t desc_encode(uint64_t x) { return (x & 0x3'FFFFULL) >> 4ULL; }

inline void check_cu(CUresult err) {
  if (err == CUDA_SUCCESS) return;
  const char *msg;
  cuGetErrorString(err, &msg);
  fprintf(stderr, "CUDA driver error: %s\n", msg);
  exit(1);
}

inline void check_cuda(cudaError_t err) {
  if (err == cudaSuccess) return;
  fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
  exit(1);
}

// ── Kernel ──

constexpr int NUM_WARPS = 6;  // 1 TMA, 1 MMA, 4 epilogue
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
constexpr int BLOCK_M = 128;   // covers N_real dimension (after swap)
constexpr int BLOCK_N = 16;    // covers M_real dimension (M_real=16, exact fit!)
constexpr int BLOCK_K = 128;   // 2 z-slices of 64 each (128-byte swizzle)
constexpr int MMA_K = 16;
constexpr int CTA_GROUP = 2;

// Split-K kernel: computes partial GEMM for a slice of K and writes fp32 to workspace
// When SPLIT_K=1, behaves exactly like the non-split-K kernel (writes bf16 to output)
// W_L2_HINT: 0=EVICT_FIRST (streaming, for large W), 1=EVICT_NORMAL (for LMHead where L2 helps)
template <bool HAS_RESIDUAL, int M_REAL = 16, int SPLIT_K = 1, int W_L2_HINT = 0>
__global__
__cluster_dims__(CTA_GROUP, 1, 1)
__launch_bounds__(TB_SIZE)
void swapab_persistent_kernel(
  const __grid_constant__ CUtensorMap W_tmap,   // "A" for MMA: W[N, K]
  const __grid_constant__ CUtensorMap A_tmap,    // "B" for MMA: A[M_pad, K]
  nv_bfloat16 *C_ptr,          // output C[M_real, N_real] row-major
  const nv_bfloat16 *res_ptr,  // residual (same layout as C), nullptr if !HAS_RESIDUAL
  int N_real, int K,
  int M_mma, int N_mma,  // after swap: M_mma=N_real, N_mma=M_pad(=32)
  float *workspace         // only used when SPLIT_K > 1: float[SPLIT_K * M_REAL * N_real]
) {
  const int tid = threadIdx.x;
  const int bid = warp_uniform(blockIdx.x);
  const int num_bids = warp_uniform(gridDim.x);
  const int warp_id = warp_uniform(tid / WARP_SIZE);
  const int lane_id = tid % WARP_SIZE;

  const int grid_m = M_mma / BLOCK_M;
  const int grid_n = N_mma / BLOCK_N;   // = 1 since N_mma=32, BLOCK_N=32

  int cta_rank;
  asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int W_size = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);   // 128*128*2 = 32768
  constexpr int A_size = (BLOCK_N / CTA_GROUP) * BLOCK_K * sizeof(nv_bfloat16);  // 16*128*2 = 4096

  constexpr int AB_per_stage = W_size + A_size;  // 36864
  constexpr int NUM_STAGES = 6;  // 6*36864 = 221184 < 228KB SMEM

  // Mbarrier layout in shared memory (after pipeline stages)
  const int tma_mbar_addr = smem + AB_per_stage * NUM_STAGES;
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;
  const int epilogue_mbar_addr = mainloop_mbar_addr + 2 * 8;

  // Prefetch TMA descriptors early
  if (warp_id == 0 && elect_sync()) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&W_tmap));
    asm volatile("prefetch.tensormap [%0];" :: "l"(&A_tmap));
  }

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES; i++) {
      mbarrier_init(tma_mbar_addr + i * 8, CTA_GROUP);
      mbarrier_init(mma_mbar_addr + i * 8, 1);
    }
    for (int i = 0; i < 2; i++) {
      mbarrier_init(mainloop_mbar_addr + i * 8, 1);
      mbarrier_init(epilogue_mbar_addr + i * 8, 4 * CTA_GROUP * WARP_SIZE);
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  }

  asm volatile("barrier.cluster.arrive.relaxed.aligned;");
  asm volatile("barrier.cluster.wait.acquire.aligned;");

  const int num_spatial_tiles = grid_m * grid_n;
  const int total_k_iters = K / BLOCK_K;
  const int iters_per_slice = total_k_iters / SPLIT_K;  // must divide evenly
  const int num_tiles = num_spatial_tiles * SPLIT_K;

  // With grid_n=1, compute_bid is simplified
  auto compute_bid = [&](int spatial_idx) -> int {
    constexpr int GROUP_M = 2;
    return spatial_idx / (grid_n * GROUP_M) * GROUP_M + (spatial_idx % GROUP_M);
  };

  if (warp_id == NUM_WARPS - 2) {
    // TMA warp
    if (elect_sync()) {
      int tma_stage = 0;
      int mma_phase = 1;
      const int tma_mbar_addr_ = tma_mbar_addr & 0xFEFFFFFF;

      for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
        const int spatial_idx = this_bid % num_spatial_tiles;
        const int k_slice = this_bid / num_spatial_tiles;
        const int k_start = k_slice * iters_per_slice;

        int bid_m = compute_bid(spatial_idx);
        const int off_m = bid_m * BLOCK_M;
        const int off_n = cta_rank * (BLOCK_N / CTA_GROUP);

        for (int i = 0; i < iters_per_slice; i++) {
          const int iter_k = k_start + i;
          const int mbar_addr = tma_mbar_addr_ + tma_stage * 8;
          const int W_smem = smem + tma_stage * AB_per_stage;
          const int A_smem = W_smem + W_size;

          mbarrier_wait(mma_mbar_addr + tma_stage * 8, mma_phase);

          const int z_coord = iter_k * (BLOCK_K / 64);  // z in units of 64-element slices
          // L2 hints: W=EVICT_FIRST (streaming), A=EVICT_LAST (reused)
          constexpr uint64_t L2_EVICT_FIRST = 0x12F0000000000000ULL;
          constexpr uint64_t L2_EVICT_LAST  = 0x14F0000000000000ULL;
          constexpr uint64_t L2_EVICT_NORMAL = 0x16F0000000000000ULL;
          constexpr uint64_t W_HINT = (W_L2_HINT == 0) ? L2_EVICT_FIRST : L2_EVICT_NORMAL;
          tma_3d_gmem2smem_l2hint<CTA_GROUP>(W_smem, &W_tmap, 0, off_m, z_coord, mbar_addr, W_HINT);
          tma_3d_gmem2smem_l2hint<CTA_GROUP>(A_smem, &A_tmap, 0, off_n, z_coord, mbar_addr, L2_EVICT_LAST);
          mbarrier_arrive_expect_tx(mbar_addr, W_size + A_size);

          tma_stage = (tma_stage + 1) % NUM_STAGES;
          if (tma_stage == 0) mma_phase ^= 1;
        }
      }
    }
  }
  else if (warp_id == NUM_WARPS - 1) {
    // MMA warp
    tcgen05_alloc<CTA_GROUP>(epilogue_mbar_addr + 8 * 2, BLOCK_N * 2);

    constexpr uint32_t MMA_M_val = BLOCK_M * CTA_GROUP;   // 256
    constexpr uint32_t MMA_N_val = BLOCK_N;                // 32
    constexpr uint32_t i_desc = (1U << 4U)    // dtype=FP32
                              | (1U << 7U)    // atype=BF16
                              | (1U << 10U)   // btype=BF16
                              | (MMA_N_val >> 3U << 17U)
                              | (MMA_M_val >> 4U << 24U);

    constexpr uint64_t AB_desc = (desc_encode(8 * 128) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);

    if (cta_rank == 0 && elect_sync()) {
      int tma_stage = 0;
      int tma_phase = 0;
      int mainloop_stage = 0;
      int epilogue_phase = 1;
      constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;

      for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
        mbarrier_wait(epilogue_mbar_addr + mainloop_stage * 8, epilogue_phase);

        for (int i = 0; i < iters_per_slice; i++) {
          const int W_smem = smem + tma_stage * AB_per_stage;
          const int A_smem = W_smem + W_size;
          const int tmem = mainloop_stage * BLOCK_N;

          uint64_t a_desc = AB_desc | (W_smem >> 4);
          uint64_t b_desc = AB_desc | (A_smem >> 4);

          mbarrier_wait(tma_mbar_addr + tma_stage * 8, tma_phase);
          asm volatile("tcgen05.fence::after_thread_sync;");

          // First z-slice (k=0..63): 4 MMA steps of 16 each
          {
            tcgen05_mma_f16<CTA_GROUP>(tmem, a_desc, b_desc, i_desc, i);
            for (int k2 = 1; k2 < 64 / MMA_K; k2++) {
              a_desc += (32 >> 4);
              b_desc += (32 >> 4);
              tcgen05_mma_f16<CTA_GROUP>(tmem, a_desc, b_desc, i_desc, 1);
            }
          }
          // Remaining z-slices (each 64 K-elements, offset by height*128 bytes)
          for (int k1 = 1; k1 < BLOCK_K / 64; k1++) {
            uint64_t a2 = AB_desc | ((W_smem + k1 * BLOCK_M * 128) >> 4);
            uint64_t b2 = AB_desc | ((A_smem + k1 * (BLOCK_N / CTA_GROUP) * 128) >> 4);
            for (int k2 = 0; k2 < 64 / MMA_K; k2++) {
              tcgen05_mma_f16<CTA_GROUP>(tmem, a2, b2, i_desc, 1);
              a2 += (32 >> 4);
              b2 += (32 >> 4);
            }
          }

          tcgen05_commit_mcast<CTA_GROUP>(mma_mbar_addr + tma_stage * 8, cta_mask);

          tma_stage = (tma_stage + 1) % NUM_STAGES;
          if (tma_stage == 0) tma_phase ^= 1;
        }

        tcgen05_commit_mcast<CTA_GROUP>(mainloop_mbar_addr + mainloop_stage * 8, cta_mask);
        mainloop_stage = (mainloop_stage + 1) % 2;
        if (mainloop_stage == 0) epilogue_phase ^= 1;
      }
    }
  }
  else {
    // Epilogue warps (warp_id 0-3)
    int mainloop_stage = 0;
    int mainloop_phase = 0;

    for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
      const int spatial_idx = this_bid % num_spatial_tiles;
      const int k_slice = this_bid / num_spatial_tiles;
      int bid_m = compute_bid(spatial_idx);

      // All epilogue warps independently wait for mainloop completion
      // (no bar.sync needed - each warp reads non-overlapping TMEM rows)
      mbarrier_wait(mainloop_mbar_addr + mainloop_stage * 8, mainloop_phase);
      asm volatile("tcgen05.fence::after_thread_sync;");

      const int n_real = bid_m * BLOCK_M + warp_id * 32 + lane_id;

      // Load columns 0-15 (the valid M_real columns)
      {
        const int t_row = cta_rank * 128 + warp_id * 32;
        const int t_col = mainloop_stage * BLOCK_N;
        const int t_addr = (t_row << 16) + t_col;

        if (n_real < N_real) {
          float f[16];
          asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x16.b32\n"
            "  {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=f"(f[0]), "=f"(f[1]), "=f"(f[2]), "=f"(f[3]),
              "=f"(f[4]), "=f"(f[5]), "=f"(f[6]), "=f"(f[7]),
              "=f"(f[8]), "=f"(f[9]), "=f"(f[10]), "=f"(f[11]),
              "=f"(f[12]), "=f"(f[13]), "=f"(f[14]), "=f"(f[15])
            : "r"(t_addr));
          asm volatile("tcgen05.wait::ld.sync.aligned;");

          if constexpr (SPLIT_K == 1) {
            // Direct output: add residual, convert to bf16, store
            if constexpr (HAS_RESIDUAL) {
              #pragma unroll
              for (int m = 0; m < M_REAL; m++) {
                f[m] += __bfloat162float(res_ptr[m * N_real + n_real]);
              }
            }
            #pragma unroll
            for (int m = 0; m < M_REAL; m++) {
              nv_bfloat16 val = __float2bfloat16(f[m]);
              asm volatile("st.global.L1::no_allocate.b16 [%0], %1;"
                :: "l"(C_ptr + m * N_real + n_real), "h"(*(uint16_t*)&val) : "memory");
            }
          } else {
            // Split-K: write fp32 partial sums to workspace
            float *ws_base = workspace + k_slice * M_REAL * N_real;
            #pragma unroll
            for (int m = 0; m < M_REAL; m++) {
              asm volatile("st.global.L1::no_allocate.b32 [%0], %1;"
                :: "l"(ws_base + m * N_real + n_real), "f"(f[m]) : "memory");
            }
          }
        }
      }

      // Signal epilogue completion
      const int mbar_addr = (epilogue_mbar_addr + mainloop_stage * 8) & 0xFEFFFFFF;
      mbarrier_arrive(mbar_addr);

      mainloop_stage = (mainloop_stage + 1) % 2;
      if (mainloop_stage == 0) mainloop_phase ^= 1;
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp_id == 0)
      tcgen05_dealloc<CTA_GROUP>(0, BLOCK_N * 2);
  }
}

// ── Split-K Reduction Kernel ──
// Sums SPLIT_K fp32 partial results, adds optional residual, converts to bf16
template <int M_REAL, bool HAS_RESIDUAL, int SPLIT_K>
__global__ void splitk_reduce_kernel(
  float *workspace,          // float[SPLIT_K * M_REAL * N_real]
  nv_bfloat16 *C_ptr,       // output [M_REAL, N_real]
  const nv_bfloat16 *res_ptr, // residual [M_REAL, N_real] or nullptr
  int N_real
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = M_REAL * N_real;
  if (idx >= total) return;

  float sum = 0.0f;
  #pragma unroll
  for (int s = 0; s < SPLIT_K; s++) {
    sum += workspace[s * total + idx];
  }

  if constexpr (HAS_RESIDUAL) {
    sum += __bfloat162float(res_ptr[idx]);
  }

  C_ptr[idx] = __float2bfloat16(sum);
}

// ── Host launch ──

// W_L2_HINT: 0=EVICT_FIRST, 1=EVICT_NORMAL
template <bool HAS_RESIDUAL, int SPLIT_K = 1, bool PROMOTE_W_L2 = true, int W_L2_HINT = 0>
void launch_kernel(
  const nv_bfloat16 *W_ptr,   // weight [N_real, K] row-major
  const nv_bfloat16 *A_ptr,   // input [M_real, K] row-major
  nv_bfloat16 *C_ptr,         // output [M_real, N_real] row-major
  const nv_bfloat16 *res_ptr, // residual [M_real, N_real] or nullptr
  int M_real, int N_real, int K,
  float *workspace = nullptr  // only for SPLIT_K > 1
) {
  // After swap: M_mma = N_real, N_mma = M_real = 16
  const int M_mma = N_real;
  const int N_mma = BLOCK_N;  // 16 (exact match with M_real=16)

  CUtensorMap W_tmap, A_tmap;

  auto init_tmap = [&](CUtensorMap *tmap, const nv_bfloat16 *ptr,
                       uint64_t global_height, uint32_t shared_height, bool l2_promote = false) {
    constexpr uint32_t rank = 3;
    uint64_t globalDim[rank]       = {64, global_height, (uint64_t)K / 64};
    uint64_t globalStrides[rank-1] = {(uint64_t)K * sizeof(nv_bfloat16), 128};
    uint32_t boxDim[rank]          = {64, shared_height, (uint32_t)BLOCK_K / 64};
    uint32_t elementStrides[rank]  = {1, 1, 1};
    check_cu(cuTensorMapEncodeTiled(
      tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, (void *)ptr,
      globalDim, globalStrides, boxDim, elementStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      l2_promote ? CU_TENSOR_MAP_L2_PROMOTION_L2_128B : CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  };

  // Per-config L2 policy: promote W only when it fits in L2 (<96MB)
  // A is always small (reused across tiles) - always promote
  init_tmap(&W_tmap, W_ptr, N_real, BLOCK_M, PROMOTE_W_L2);
  init_tmap(&A_tmap, A_ptr, N_mma, BLOCK_N / CTA_GROUP, true);

  const int num_spatial_tiles = (M_mma / BLOCK_M) * (N_mma / BLOCK_N);
  const int num_tiles = num_spatial_tiles * SPLIT_K;
  int grid = std::min(148, num_tiles);  // 148 SMs on B200

  constexpr int AB_per_stage = (BLOCK_M + BLOCK_N / CTA_GROUP) * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int NUM_STAGES = 6;  // matches kernel
  constexpr int mbar_size = NUM_STAGES * 2 * 8 + 4 * 8 + 16;
  int smem_size = AB_per_stage * NUM_STAGES + mbar_size;

  auto kern = swapab_persistent_kernel<HAS_RESIDUAL, 16, SPLIT_K, W_L2_HINT>;
  check_cuda(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  kern<<<grid, TB_SIZE, smem_size>>>(W_tmap, A_tmap, C_ptr, res_ptr,
                                      N_real, K, M_mma, N_mma, workspace);
  check_cuda(cudaGetLastError());

  // If split-K, launch reduction kernel
  if constexpr (SPLIT_K > 1) {
    const int total_elems = 16 * N_real;
    const int reduce_threads = 256;
    const int reduce_blocks = cdiv(total_elems, reduce_threads);
    splitk_reduce_kernel<16, HAS_RESIDUAL, SPLIT_K>
      <<<reduce_blocks, reduce_threads>>>(workspace, C_ptr, res_ptr, N_real);
    check_cuda(cudaGetLastError());
  }
}

// ── Benchmark harness ──

struct GemmConfig {
  const char *name;
  int M, N, K;
  bool has_residual;
};

int main() {
  // Initialize CUDA
  check_cuda(cudaSetDevice(0));

  GemmConfig configs[] = {
    {"QKV",    16, 6144,   4096,  false},
    {"O",      16, 4096,   4096,  true},
    {"GateUp", 16, 24576,  4096,  false},
    {"Down",   16, 4096,   12288, true},
    {"LMHead", 16, 153600, 4096,  false},
  };

  // cuBLAS handle
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

  // L2 flush buffer (128 MB > 96 MB L2)
  size_t flush_sz = 128 * 1024 * 1024;
  char *flush_buf;
  check_cuda(cudaMalloc(&flush_buf, flush_sz));

  // Split-K workspaces
  constexpr int DOWN_SPLIT_K = 4;
  constexpr int O_SPLIT_K = 4;
  constexpr int QKV_SPLIT_K = 2;     // 48*2=96 tiles, balance between parallelism and overhead
  constexpr int GATEUP_SPLIT_K = 2;  // 192*2=384 tiles, better wave utilization
  
  // Allocate max workspace: max across all split-K configs
  size_t max_ws_size = (size_t)std::max({DOWN_SPLIT_K * 16 * 12288,
                                          QKV_SPLIT_K * 16 * 6144,
                                          GATEUP_SPLIT_K * 16 * 24576}) * sizeof(float);
  float *splitk_workspace;
  check_cuda(cudaMalloc(&splitk_workspace, max_ws_size));

  printf("KERNEL_RESULT {");
  bool first_kr = true;

  double ref_tflops[5];
  double ker_tflops[5];

  for (int ci = 0; ci < 5; ci++) {
    auto &cfg = configs[ci];
    int M = cfg.M, N = cfg.N, K = cfg.K;
    double flops = 2.0 * M * N * K;

    // Allocate
    nv_bfloat16 *dA, *dW, *dC, *dC_ref, *dRes;
    size_t padM = BLOCK_N; // 32

    check_cuda(cudaMalloc(&dA, (size_t)padM * K * sizeof(nv_bfloat16)));
    check_cuda(cudaMemset(dA, 0, (size_t)padM * K * sizeof(nv_bfloat16)));
    check_cuda(cudaMalloc(&dW, (size_t)N * K * sizeof(nv_bfloat16)));
    check_cuda(cudaMalloc(&dC, (size_t)M * N * sizeof(nv_bfloat16)));
    check_cuda(cudaMalloc(&dC_ref, (size_t)M * N * sizeof(nv_bfloat16)));

    if (cfg.has_residual)
      check_cuda(cudaMalloc(&dRes, (size_t)M * N * sizeof(nv_bfloat16)));
    else
      dRes = nullptr;

    // Initialize with random data
    {
      size_t nA = (size_t)M * K;
      size_t nW = (size_t)N * K;
      size_t nC = (size_t)M * N;
      nv_bfloat16 *hA = new nv_bfloat16[nA];
      nv_bfloat16 *hW = new nv_bfloat16[nW];
      srand(42 + ci);
      for (size_t i = 0; i < nA; i++) hA[i] = __float2bfloat16((rand() % 201 - 100) / 100.0f);
      for (size_t i = 0; i < nW; i++) hW[i] = __float2bfloat16((rand() % 201 - 100) / 100.0f);
      check_cuda(cudaMemcpy(dA, hA, nA * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
      check_cuda(cudaMemcpy(dW, hW, nW * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
      if (cfg.has_residual) {
        nv_bfloat16 *hRes = new nv_bfloat16[nC];
        for (size_t i = 0; i < nC; i++) hRes[i] = __float2bfloat16((rand() % 201 - 100) / 100.0f);
        check_cuda(cudaMemcpy(dRes, hRes, nC * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        delete[] hRes;
      }
      delete[] hA;
      delete[] hW;
    }

    // ── cuBLAS reference ──
    {
      float alpha = 1.0f, beta = 0.0f;
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                   N, M, K,
                   &alpha,
                   dW, CUDA_R_16BF, K,
                   dA, CUDA_R_16BF, K,
                   &beta,
                   dC_ref, CUDA_R_16BF, N,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
      check_cuda(cudaDeviceSynchronize());

      for (int i = 0; i < 20; i++) {
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                     &alpha, dW, CUDA_R_16BF, K, dA, CUDA_R_16BF, K,
                     &beta, dC_ref, CUDA_R_16BF, N,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
      }
      check_cuda(cudaDeviceSynchronize());

      cudaEvent_t ev0, ev1;
      cudaEventCreate(&ev0);
      cudaEventCreate(&ev1);
      float times[100];
      for (int i = 0; i < 100; i++) {
        check_cuda(cudaMemset(flush_buf, 0, flush_sz));
        check_cuda(cudaDeviceSynchronize());
        cudaEventRecord(ev0);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                     &alpha, dW, CUDA_R_16BF, K, dA, CUDA_R_16BF, K,
                     &beta, dC_ref, CUDA_R_16BF, N,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&times[i], ev0, ev1);
      }
      std::sort(times, times + 100);
      ref_tflops[ci] = flops / (times[50] * 1e-3) / 1e12;
      cudaEventDestroy(ev0);
      cudaEventDestroy(ev1);
    }

    // ── Our kernel ──
    check_cuda(cudaMemset(dC, 0, (size_t)M * N * sizeof(nv_bfloat16)));

    // Use split-K for configs with low tile counts
    // L2 policy: promote W when it fits in L2 (QKV 48MB, O 32MB), skip for large W
    if (ci == 0) {
      // QKV: W=48MB, split-K=2, promote W
      launch_kernel<false, QKV_SPLIT_K, true>(dW, dA, dC, nullptr, M, N, K, splitk_workspace);
    } else if (ci == 1) {
      // O: W=32MB, split-K=4, promote W
      launch_kernel<true, O_SPLIT_K, true>(dW, dA, dC, dRes, M, N, K, splitk_workspace);
    } else if (ci == 2) {
      // GateUp: W=192MB, no split-K, DON'T promote W
      launch_kernel<false, 1, false>(dW, dA, dC, nullptr, M, N, K);
    } else if (ci == 3) {
      // Down: W=96MB total, but split-K=4 means 24MB/slice fits L2 → promote
      launch_kernel<true, DOWN_SPLIT_K, true>(dW, dA, dC, dRes, M, N, K, splitk_workspace);
    } else {
      // LMHead: W=1200MB, no split-K, DON'T promote W, EVICT_NORMAL for L2 spatial reuse
      launch_kernel<false, 1, false, 1>(dW, dA, dC, nullptr, M, N, K);
    }
    check_cuda(cudaDeviceSynchronize());

    // Verify
    {
      size_t nC = (size_t)M * N;
      nv_bfloat16 *hC = new nv_bfloat16[nC];
      nv_bfloat16 *hRef = new nv_bfloat16[nC];
      nv_bfloat16 *hRes = nullptr;
      check_cuda(cudaMemcpy(hC, dC, nC * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
      check_cuda(cudaMemcpy(hRef, dC_ref, nC * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
      if (cfg.has_residual) {
        hRes = new nv_bfloat16[nC];
        check_cuda(cudaMemcpy(hRes, dRes, nC * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
      }

      float max_err = 0;
      int err_count = 0;
      for (size_t i = 0; i < nC; i++) {
        float got = __bfloat162float(hC[i]);
        float ref_val = __bfloat162float(hRef[i]);
        if (cfg.has_residual) ref_val += __bfloat162float(hRes[i]);
        float e = fabsf(got - ref_val) / fmaxf(fmaxf(fabsf(ref_val), fabsf(got)), 1.0f);
        if (e > max_err) max_err = e;
        if (e > 5e-3f) err_count++;
      }
      fprintf(stderr, "%s: max_err=%.6f errs=%d/%zu\n", cfg.name, max_err, err_count, nC);
      delete[] hC;
      delete[] hRef;
      if (hRes) delete[] hRes;
    }

    // Benchmark our kernel
    auto run_kernel = [&]() {
      if (ci == 0) {
        launch_kernel<false, QKV_SPLIT_K, true>(dW, dA, dC, nullptr, M, N, K, splitk_workspace);
      } else if (ci == 1) {
        launch_kernel<true, O_SPLIT_K, true>(dW, dA, dC, dRes, M, N, K, splitk_workspace);
      } else if (ci == 2) {
        launch_kernel<false, 1, false>(dW, dA, dC, nullptr, M, N, K);
      } else if (ci == 3) {
        launch_kernel<true, DOWN_SPLIT_K, true>(dW, dA, dC, dRes, M, N, K, splitk_workspace);
      } else {
        launch_kernel<false, 1, false, 1>(dW, dA, dC, nullptr, M, N, K);
      }
    };

    for (int i = 0; i < 20; i++) run_kernel();
    check_cuda(cudaDeviceSynchronize());

    {
      cudaEvent_t ev0, ev1;
      cudaEventCreate(&ev0);
      cudaEventCreate(&ev1);
      float times[100];
      for (int i = 0; i < 100; i++) {
        check_cuda(cudaMemset(flush_buf, 0, flush_sz));
        check_cuda(cudaDeviceSynchronize());
        cudaEventRecord(ev0);
        run_kernel();
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&times[i], ev0, ev1);
      }
      std::sort(times, times + 100);
      ker_tflops[ci] = flops / (times[50] * 1e-3) / 1e12;
      cudaEventDestroy(ev0);
      cudaEventDestroy(ev1);
    }

    if (!first_kr) printf(", ");
    printf("\"%s\": %.2f", cfg.name, ker_tflops[ci]);
    first_kr = false;

    cudaFree(dA); cudaFree(dW); cudaFree(dC); cudaFree(dC_ref);
    if (dRes) cudaFree(dRes);
  }
  printf("}\n");

  printf("KERNEL_RESULT_REFERENCE {");
  for (int ci = 0; ci < 5; ci++) {
    if (ci > 0) printf(", ");
    printf("\"%s\": %.2f", configs[ci].name, ref_tflops[ci]);
  }
  printf("}\n");

  cudaFree(flush_buf);
  cudaFree(splitk_workspace);
  cublasDestroy(handle);
  return 0;
}
