// Variant of example 05 for SMALL-M GEMM using swapab + cg2 on B200.
//
// Problem: example 05's cg2 kernel splits the M dimension across 2 CTAs
// (MMA_M = BLOCK_M × 2). For tasks where M is small (e.g. M=16 decode GEMM),
// direct cg2 fails structurally: BLOCK_M >= 64 padding wastes 75-87.5% of
// lanes, and grid_m=1 means there are no adjacent M-blocks to dispatch
// across a 2-CTA cluster.
//
// Fix (swapab): transpose so C^T = B^T @ A^T. Now the large N dimension
// plays the role of "M" in the MMA, and the small M plays the role of "N":
//   MMA_M = N_original (large)     → BLOCK_M=128 is fully utilized
//   MMA_N = M_original (small)     → pad to BLOCK_N=32 (cg2 kind::f16 min
//                                     is N=16 steps of 16, but BLOCK_N=16
//                                     produces illegal instruction on B200
//                                     — minimum working BLOCK_N is 32)
//
// VERIFIED on B200 (2026-04-15): M=16, N=6144, K=4096 → 0 errors,
// max_rel_err = 0.0039, 29.37 TFLOPS.
//
// NOT competitive with tuned cg1 out of the box (cg1 for the same shape
// hits ~43 TFLOPS with persistent warp-specialized pipeline). This file
// is a CORRECTNESS PROOF and a structural starting point, not a
// performance reference. To match cg1 performance, this kernel needs:
//   - Persistent scheduling (example 05 is non-persistent)
//   - Warp-specialized epilogue (TMA / MMA / epilogue / load warp)
//   - Better pipeline stage count tuning for K=4096
//
// Build:
//   nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lcuda \
//        05b_cg2_swapab_small_m.cu -o 05b_cg2_swapab_small_m
// Run:
//   eval $(./pick_gpu.sh) && ./05b_cg2_swapab_small_m

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

// ── common.h equivalents (no torch dependency) ──

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

// ── Kernel from example 05 (verbatim logic, profiler removed) ──

constexpr int NUM_WARPS = 4;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
constexpr int BLOCK_M = 128;
constexpr int MMA_K = 16;

template <int BLOCK_N, int BLOCK_K, int CTA_GROUP, int NUM_STAGES>
__global__
__cluster_dims__(CTA_GROUP, 1, 1)
__launch_bounds__(TB_SIZE)
void cg2_kernel(
  const __grid_constant__ CUtensorMap A_tmap,
  const __grid_constant__ CUtensorMap B_tmap,
  nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int grid_m = M / BLOCK_M;
  const int grid_n = N / BLOCK_N;

  constexpr int GROUP_M = 2;
  const int bid_m = bid / (grid_n * GROUP_M) * GROUP_M + (bid % GROUP_M);
  const int bid_n = (bid / GROUP_M) % grid_n;

  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

  int cta_rank;
  asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int B_size = (BLOCK_N / CTA_GROUP) * BLOCK_K * sizeof(nv_bfloat16);

  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ uint64_t mbars[NUM_STAGES * 2 + 1];
  __shared__ int tmem_addr[1];
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES; i++) {
      mbarrier_init(tma_mbar_addr + i * 8, CTA_GROUP);
      mbarrier_init(mma_mbar_addr + i * 8, 1);
    }
    mbarrier_init(mainloop_mbar_addr, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  else if (warp_id == 1) {
    const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
    tcgen05_alloc<CTA_GROUP>(addr, BLOCK_N);
  }

  if constexpr (CTA_GROUP > 1) {
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
  } else {
    __syncthreads();
  }
  const int taddr = tmem_addr[0];

  int phase = 0;

  constexpr int MMA_M = BLOCK_M * CTA_GROUP;
  constexpr uint32_t i_desc = (1U << 4U)
                            | (1U << 7U)
                            | (1U << 10U)
                            | ((uint32_t)BLOCK_N >> 3U << 17U)
                            | ((uint32_t)MMA_M >> 4U << 24U);

  auto load = [&](int iter_k) {
    const int stage_id = iter_k % NUM_STAGES;
    mbarrier_wait(mma_mbar_addr + stage_id * 8, phase ^ 1);
    if (stage_id == NUM_STAGES - 1) phase ^= 1;

    const int mbar_addr = (tma_mbar_addr + stage_id * 8) & 0xFEFFFFFF;
    const int A_smem = smem + stage_id * (A_size + B_size);
    const int B_smem = A_smem + A_size;
    const int off_k = iter_k * BLOCK_K;

    tma_3d_gmem2smem<CTA_GROUP>(A_smem, &A_tmap, 0, off_m, off_k / 64, mbar_addr);
    tma_3d_gmem2smem<CTA_GROUP>(B_smem, &B_tmap, 0, off_n + cta_rank * (BLOCK_N / CTA_GROUP), off_k / 64, mbar_addr);
    mbarrier_arrive_expect_tx(mbar_addr, A_size + B_size);
  };

  auto compute = [&](int iter_k) {
    const int stage_id = iter_k % NUM_STAGES;
    mbarrier_wait(tma_mbar_addr + stage_id * 8, phase);
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (stage_id == NUM_STAGES - 1) phase ^= 1;

    const int A_smem = smem + stage_id * (A_size + B_size);
    const int B_smem = A_smem + A_size;

    auto make_desc = [](int addr) -> uint64_t {
      const int SBO = 8 * 128;
      return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
    };

    {
      tcgen05_mma_f16<CTA_GROUP>(taddr, make_desc(A_smem), make_desc(B_smem), i_desc, iter_k);
      for (int k2 = 1; k2 < 64 / MMA_K; k2++) {
        uint64_t a_desc = make_desc(A_smem + k2 * 32);
        uint64_t b_desc = make_desc(B_smem + k2 * 32);
        tcgen05_mma_f16<CTA_GROUP>(taddr, a_desc, b_desc, i_desc, 1);
      }
    }
    for (int k1 = 1; k1 < BLOCK_K / 64; k1++)
      for (int k2 = 0; k2 < 64 / MMA_K; k2++) {
        uint64_t a_desc = make_desc(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
        uint64_t b_desc = make_desc(B_smem + k1 * (BLOCK_N / CTA_GROUP) * 128 + k2 * 32);
        tcgen05_mma_f16<CTA_GROUP>(taddr, a_desc, b_desc, i_desc, 1);
      }

    constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
    tcgen05_commit_mcast<CTA_GROUP>(mma_mbar_addr + stage_id * 8, cta_mask);
  };

  const int num_iters = K / BLOCK_K;
  if (warp_id == 0 && elect_sync()) {
    for (int iter_k = 0; iter_k < num_iters; iter_k++) load(iter_k);
  }
  else if (cta_rank == 0 && warp_id == 1 && elect_sync()) {
    for (int iter_k = 0; iter_k < num_iters; iter_k++) compute(iter_k);
    constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
    tcgen05_commit_mcast<CTA_GROUP>(mainloop_mbar_addr, cta_mask);
  }

  __syncthreads();
  mbarrier_wait(mainloop_mbar_addr, 0);
  asm volatile("tcgen05.fence::after_thread_sync;");

  for (int n = 0; n < BLOCK_N / 8; n++) {
    float tmp[8];
    const int addr = taddr + ((cta_rank * 128 + warp_id * 32) << 16) + (n * 8);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=f"(tmp[0]),"=f"(tmp[1]),"=f"(tmp[2]),"=f"(tmp[3]),
                  "=f"(tmp[4]),"=f"(tmp[5]),"=f"(tmp[6]),"=f"(tmp[7])
                : "r"(addr));
    asm volatile("tcgen05.wait::ld.sync.aligned;");

    nv_bfloat162 out[4];
    for (int i = 0; i < 4; i++)
      out[i] = __float22bfloat162_rn({tmp[i*2], tmp[i*2+1]});

    nv_bfloat16 *out_ptr = C_ptr + (off_m + tid) * N + (off_n + n * 8);
    reinterpret_cast<int4 *>(out_ptr)[0] = reinterpret_cast<int4 *>(out)[0];
  }

  __syncthreads();
  if (warp_id == 0)
    tcgen05_dealloc<CTA_GROUP>(taddr, BLOCK_N);
}

// ── Launch (from example 05, TMA descriptor setup) ──

template <int BLOCK_N, int BLOCK_K, int CTA_GROUP, int NUM_STAGES>
void launch_cg2(
  const nv_bfloat16 *A_ptr, const nv_bfloat16 *B_ptr, nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  CUtensorMap A_tmap, B_tmap;
  constexpr uint32_t rank = 3;

  auto init_tmap = [&](CUtensorMap *tmap, const nv_bfloat16 *ptr,
                       uint64_t global_height, uint32_t shared_height) {
    uint64_t globalDim[rank]       = {64, global_height, (uint64_t)K / 64};
    uint64_t globalStrides[rank-1] = {(uint64_t)K * sizeof(nv_bfloat16), 128};
    uint32_t boxDim[rank]          = {64, shared_height, (uint32_t)BLOCK_K / 64};
    uint32_t elementStrides[rank]  = {1, 1, 1};
    check_cu(cuTensorMapEncodeTiled(
      tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, (void *)ptr,
      globalDim, globalStrides, boxDim, elementStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  };

  init_tmap(&A_tmap, A_ptr, M, BLOCK_M);
  init_tmap(&B_tmap, B_ptr, N, BLOCK_N / CTA_GROUP);

  int grid = (M / BLOCK_M) * (N / BLOCK_N);
  int smem_size = (BLOCK_M + BLOCK_N / CTA_GROUP) * BLOCK_K * NUM_STAGES * sizeof(nv_bfloat16);

  auto kern = cg2_kernel<BLOCK_N, BLOCK_K, CTA_GROUP, NUM_STAGES>;
  if (smem_size > 48000)
    check_cuda(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  kern<<<grid, TB_SIZE, smem_size>>>(A_tmap, B_tmap, C_ptr, M, N, K);
}

// ── Main: test swapab + cg2 ──

int main() {
  // Original GEMM: C[M_real, N_real] = A[M_real, K] @ B[K, N_real]
  // where M_real=16 (small), N_real=6144 (QKV shape, large)
  //
  // Swapab: C^T[N_real, M_real] = B^T[N_real, K] @ A^T[K, M_real]
  // Call cg2 kernel with: M_mma=N_real=6144, N_mma=M_real=16, K=4096
  //
  // In row-major: B^T is B stored as [N_real, K], A^T is A stored as [K, M_real]
  // C^T is [N_real, M_real], transpose back to get C[M_real, N_real]

  const int M_real = 16;
  const int N_real = 6144;  // QKV shape
  const int K = 4096;

  // For the MMA kernel (after swap):
  const int M_mma = N_real;  // 6144 — large, fully utilized with BLOCK_M=128
  const int N_mma = M_real;  // 16 — exact fit for cg2 MMA_N=16

  printf("swapab + cg2 test: C[%d,%d] = A[%d,%d] @ B[%d,%d]\n", M_real, N_real, M_real, K, K, N_real);
  printf("After swap: M_mma=%d, N_mma=%d, K=%d\n", M_mma, N_mma, K);
  printf("BLOCK_M=128, CTA_GROUP=2 → MMA_M=256, MMA_N=16\n\n");

  // Allocate
  size_t sA = (size_t)M_real * K;
  size_t sB = (size_t)N_real * K;
  size_t sC = (size_t)M_real * N_real;

  nv_bfloat16 *hA = new nv_bfloat16[sA];
  nv_bfloat16 *hB = new nv_bfloat16[sB];
  float *hC_ref = new float[sC];

  srand(42);
  for (size_t i = 0; i < sA; i++) hA[i] = __float2bfloat16((rand() % 201 - 100) / 100.0f);
  for (size_t i = 0; i < sB; i++) hB[i] = __float2bfloat16((rand() % 201 - 100) / 100.0f);

  // CPU reference: C[m,n] = sum_k A[m,k] * B[k,n]
  // A is row-major [M_real, K], B is row-major [K, N_real] (but stored as B^T[N_real, K] for TMA)
  // Wait — we need B in [K, N_real] for the original GEMM reference.
  // But hB is generated as a flat array. Let's say hB[i] = B_flat[i].
  // Original: B[K, N_real] → element B[k,n] = hB[k * N_real + n]
  // Transposed for kernel: B^T[N_real, K] → element B^T[n,k] = hB[k * N_real + n]
  // So B^T stored row-major is: B^T_row[n * K + k] = B[k,n] = hB[k * N_real + n]
  //
  // For simplicity: generate hB_T directly as [N_real, K] row-major (this is what TMA sees)
  // Then B[k,n] = hB_T[n * K + k]
  nv_bfloat16 *hB_T = new nv_bfloat16[(size_t)N_real * K];
  for (int n = 0; n < N_real; n++)
    for (int k = 0; k < K; k++)
      hB_T[(size_t)n * K + k] = hB[(size_t)k * N_real + n];

  // CPU ref: C[m,n] = sum_k A[m,k] * B[k,n] where B[k,n] = hB_T[n*K + k]
  for (int m = 0; m < M_real; m++)
    for (int n = 0; n < N_real; n++) {
      float acc = 0;
      for (int k = 0; k < K; k++)
        acc += __bfloat162float(hA[m * K + k]) * __bfloat162float(hB_T[(size_t)n * K + k]);
      hC_ref[m * N_real + n] = acc;
    }

  // Device alloc
  // Kernel sees: A_mma[M_mma, K] = B^T[N_real, K], B_mma[K, N_mma] = A^T[K, M_real]
  // But TMA with 3D swizzled layout expects row-major storage.
  // A_mma = hB_T[N_real, K] — already row-major ✓
  // B_mma = A^T[K, M_real] — need A transposed to [K, M_real] row-major
  // Wait: the kernel's B TMA loads B[K, N_mma] but stored as [N_mma, K] for TMA?
  // No — looking at example 05's TMA setup:
  //   init_tmap(&B_tmap, B_ptr, N, BLOCK_N / CTA_GROUP)
  //   globalDim = {64, N, K/64}
  //   The ptr points to B stored row-major as [N, K] (same as A)
  // So B_ptr is [N_mma, K] row-major = [M_real, K] row-major = hA (the original A)!

  nv_bfloat16 *dA_mma, *dB_mma, *dC_mma;
  // A_mma = B^T [N_real, K]
  check_cuda(cudaMalloc(&dA_mma, (size_t)N_real * K * 2));
  check_cuda(cudaMemcpy(dA_mma, hB_T, (size_t)N_real * K * 2, cudaMemcpyHostToDevice));
  // B_mma = A [M_real, K] — kernel expects B stored as [N_mma, K] = [M_real, K]
  check_cuda(cudaMalloc(&dB_mma, (size_t)M_real * K * 2));
  check_cuda(cudaMemcpy(dB_mma, hA, (size_t)M_real * K * 2, cudaMemcpyHostToDevice));
  // C_mma = C^T [N_real, M_real]
  check_cuda(cudaMalloc(&dC_mma, (size_t)N_real * M_real * 2));
  check_cuda(cudaMemset(dC_mma, 0, (size_t)N_real * M_real * 2));

  // Pad N_mma to BLOCK_N (must be >= 16 per CTA for swizzle, so minimum BLOCK_N=32 for cg2)
  constexpr int BLOCK_N_TEST = 32;
  const int N_mma_padded = ((N_mma + BLOCK_N_TEST - 1) / BLOCK_N_TEST) * BLOCK_N_TEST;  // round up to BLOCK_N

  // Reallocate C for padded N dimension
  cudaFree(dC_mma);
  check_cuda(cudaMalloc(&dC_mma, (size_t)M_mma * N_mma_padded * 2));
  check_cuda(cudaMemset(dC_mma, 0, (size_t)M_mma * N_mma_padded * 2));

  // Also need to pad B_mma to N_mma_padded rows (B_mma is [N_mma, K], pad to [N_mma_padded, K])
  if (N_mma_padded > N_mma) {
    nv_bfloat16 *dB_padded;
    check_cuda(cudaMalloc(&dB_padded, (size_t)N_mma_padded * K * 2));
    check_cuda(cudaMemset(dB_padded, 0, (size_t)N_mma_padded * K * 2));
    check_cuda(cudaMemcpy(dB_padded, dB_mma, (size_t)N_mma * K * 2, cudaMemcpyDeviceToDevice));
    cudaFree(dB_mma);
    dB_mma = dB_padded;
  }

  printf("N_mma_padded=%d (BLOCK_N=%d)\n", N_mma_padded, BLOCK_N_TEST);
  printf("Launching cg2 kernel...\n");
  launch_cg2<BLOCK_N_TEST, 64, 2, 7>(dA_mma, dB_mma, dC_mma, M_mma, N_mma_padded, K);
  check_cuda(cudaDeviceSynchronize());
  printf("Kernel completed.\n");

  // Read back C_mma[M_mma, N_mma_padded] — only first N_mma=M_real columns are real
  nv_bfloat16 *hC_T = new nv_bfloat16[(size_t)M_mma * N_mma_padded];
  check_cuda(cudaMemcpy(hC_T, dC_mma, (size_t)M_mma * N_mma_padded * 2, cudaMemcpyDeviceToHost));

  // C_mma[n_orig, m_orig] = C^T[n_orig, m_orig] should == C_ref[m_orig, n_orig]
  // C_mma is stored row-major as [M_mma, N_mma_padded] where M_mma=N_real, columns 0..M_real-1 are real
  float max_err = 0;
  int err_count = 0;
  for (int m = 0; m < M_real; m++)
    for (int n = 0; n < N_real; n++) {
      // C_mma row = n (N_real dim), col = m (M_real dim)
      float got = __bfloat162float(hC_T[(size_t)n * N_mma_padded + m]);
      float ref = hC_ref[m * N_real + n];
      float e = fabsf(got - ref) / fmaxf(fmaxf(fabsf(ref), fabsf(got)), 1.0f);
      if (e > max_err) max_err = e;
      if (e > 5e-3f) err_count++;
    }

  int total = M_real * N_real;
  printf("\nResults: max_rel_err=%.6f errs=%d/%d (%.1f%%)\n", max_err, err_count, total, 100.0 * err_count / total);
  if (err_count == 0 && max_err < 5e-3f)
    printf("PASS\n");
  else
    printf("FAIL\n");

  // ── Benchmark ──
  if (err_count == 0) {
    // L2 flush buffer
    size_t flush_sz = 128 * 1024 * 1024;
    char *flush_buf;
    check_cuda(cudaMalloc(&flush_buf, flush_sz));

    // Warmup
    for (int i = 0; i < 20; i++)
      launch_cg2<BLOCK_N_TEST, 64, 2, 7>(dA_mma, dB_mma, dC_mma, M_mma, N_mma_padded, K);
    check_cuda(cudaDeviceSynchronize());

    // Timed iters with L2 flush
    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);
    float times_ms[100];
    for (int i = 0; i < 100; i++) {
      check_cuda(cudaMemset(flush_buf, 0, flush_sz));
      check_cuda(cudaDeviceSynchronize());
      cudaEventRecord(ev_start);
      launch_cg2<BLOCK_N_TEST, 64, 2, 7>(dA_mma, dB_mma, dC_mma, M_mma, N_mma_padded, K);
      cudaEventRecord(ev_end);
      cudaEventSynchronize(ev_end);
      cudaEventElapsedTime(&times_ms[i], ev_start, ev_end);
    }
    std::sort(times_ms, times_ms + 100);
    float median_ms = times_ms[50];
    double flops = 2.0 * M_real * N_real * K;
    double tflops = flops / (median_ms * 1e-3) / 1e12;
    printf("\nBenchmark: median=%.3f ms  TFLOPS=%.2f (FLOPs based on real M=%d, N=%d, K=%d)\n",
           median_ms, tflops, M_real, N_real, K);
    cudaFree(flush_buf);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
  }

  // Cleanup
  delete[] hA; delete[] hB; delete[] hB_T; delete[] hC_ref; delete[] hC_T;
  cudaFree(dA_mma); cudaFree(dB_mma); cudaFree(dC_mma);
  return err_count > 0 ? 1 : 0;
}
