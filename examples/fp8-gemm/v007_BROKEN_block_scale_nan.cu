// FP8 Block-Scaled GEMM for Blackwell (SM100a) - block_scale MMA
// Uses kind::mxf8f6f4.block_scale with UE8M0 scales in TMEM
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>

// Device helpers
__device__ __forceinline__ uint32_t elect_one_sync() {
    uint32_t pred = 0;
    asm volatile("{\n\t.reg .pred %%px;\n\telect.sync _|%%px, %1;\n\t@%%px mov.s32 %0, 1;\n\t}"
        : "+r"(pred) : "r"(0xFFFFFFFF));
    return pred;
}
__device__ __forceinline__ void mb_init(int a, int c) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(c));
}
__device__ __forceinline__ void mb_wait(int a, int p) {
    asm volatile("{\n\t.reg .pred P1;\n\tLW:\n\tmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t@P1 bra.uni DN;\n\tbra.uni LW;\n\tDN:\n\t}" :: "r"(a), "r"(p), "r"(0x989680));
}
__device__ __forceinline__ void mb_arrive(int a) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(a) : "memory");
}
__device__ __forceinline__ void mb_arrive_tx(int a, int s) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(s) : "memory");
}
__device__ __forceinline__ void tma_ld(int d, const void *t, int x, int y, int m) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        :: "r"(d), "l"(t), "r"(x), "r"(y), "r"(m) : "memory");
}
__device__ __forceinline__ void tma_store_2d(const void *desc, const void *smem, int x, int y) {
    uint64_t d = reinterpret_cast<uint64_t>(desc);
    uint32_t sp = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"(d), "r"(sp), "r"(x), "r"(y) : "memory");
}
__device__ __forceinline__ constexpr uint64_t denc(uint64_t x) { return (x & 0x3FFFFULL) >> 4ULL; }
// K-major 128B swizzle descriptor: SBO=1024, version=1, layout=SWIZZLE_128B(2)
__device__ __forceinline__ uint64_t mkdesc(int a) {
    return denc(a) | (denc(1024) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
}
// SF descriptor: no swizzle, SBO=8*16=128 (for 4x32dp128bit UTCCP), LBO=0
__device__ __forceinline__ uint64_t mkdesc_sf(int a) {
    return denc(a) | (denc(128) << 32ULL) | (1ULL << 46ULL);
}
__device__ __forceinline__ uint32_t ld_shared(const void* p) {
    uint32_t r; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "l"(p)); return r;
}
__device__ __forceinline__ void st_shared(const void* p, uint32_t v) {
    asm volatile("st.shared.u32 [%0], %1;" :: "l"(p), "r"(v));
}
__device__ __forceinline__ uint32_t get_lane_idx() {
    uint32_t l; asm("mov.u32 %0, %laneid;" : "=r"(l)); return l;
}

// UTCCP transpose: rearrange 128 uint32_t in SMEM for tcgen05.cp
__device__ __forceinline__ void utccp_transpose(uint32_t* ptr, uint32_t lid) {
    uint32_t v[4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        v[i] = ld_shared(ptr + (i ^ (lid >> 3)) * 32 + lid);
    __syncwarp();
    #pragma unroll
    for (int i = 0; i < 4; i++)
        st_shared(ptr + lid * 4 + (i ^ (lid >> 3)), v[i]);
}

template <int BN, int NS>
__global__ void __launch_bounds__(256, 1)
fp8_gemm_bs(
    const __grid_constant__ CUtensorMap ta,
    const __grid_constant__ CUtensorMap tb,
    const __grid_constant__ CUtensorMap tsfa,
    const __grid_constant__ CUtensorMap tsfb,
    const __grid_constant__ CUtensorMap td,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K, int num_sms
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
    constexpr int BM = 128, BK = 128, UK = 32;
    constexpr int SF_PER_LOAD = 4; // 4 UE8M0 per uint32_t
    constexpr int NE = 2;
    const int tid = threadIdx.x, wid = tid / 32;
    const uint32_t lid = get_lane_idx();
    const int nn = (N + BN - 1) / BN, nk = (K + BK - 1) / BK;
    const int total = ((M + BM - 1) / BM) * nn;

    extern __shared__ __align__(1024) uint8_t sm[];
    constexpr int SA = BM * BK, SB = BN * BK;
    constexpr int SFA_SIZE = 128 * 4; // 128 uint32_t (packed UE8M0)
    constexpr int SFB_SIZE = 128 * 4;

    auto sA = [&](int s) -> uint8_t* { return sm + s * SA; };
    auto sB = [&](int s) -> uint8_t* { return sm + NS * SA + s * SB; };
    auto sSFA = [&](int s) -> uint32_t* { return (uint32_t*)(sm + NS * (SA + SB) + s * SFA_SIZE); };
    auto sSFB = [&](int s) -> uint32_t* { return (uint32_t*)(sm + NS * (SA + SB) + NS * SFA_SIZE + s * SFB_SIZE); };

    // CD store buffer (1 stage, 128 rows * 128 bytes = 16384)
    constexpr int SCD = BM * 128;
    uint8_t* sCD = sm + NS * (SA + SB + SFA_SIZE + SFB_SIZE);

    int bar_base = NS * (SA + SB + SFA_SIZE + SFB_SIZE) + SCD;
    bar_base = (bar_base + 7) & ~7;
    auto bars = reinterpret_cast<uint64_t*>(sm + bar_base);
    int bf = __cvta_generic_to_shared(bars);        // full[NS]: TMA done
    int be = bf + NS * 8;                            // empty[NS]: SMEM free
    int bsf = be + NS * 8;                           // sf_ready[NS]: SF transposed
    int btf = bsf + NS * 8;                          // tmem_full[NE]: accum ready
    int bte = btf + NE * 8;                           // tmem_empty[NE]: accum consumed
    auto tp = reinterpret_cast<uint32_t*>(bars + NS * 3 + NE * 2);

    // TMEM layout: NE*BN accum cols + 4 SFA cols + 4 SFB cols
    constexpr int TMEM_SFA = NE * BN;
    constexpr int TMEM_SFB = TMEM_SFA + 4;
    constexpr int TMEM_TOTAL = NE * BN + 4 + 4;
    constexpr int TCA = TMEM_TOTAL <= 32 ? 32 : TMEM_TOTAL <= 64 ? 64 : TMEM_TOTAL <= 128 ? 128 : TMEM_TOTAL <= 256 ? 256 : 512;

    if (wid == 0 && elect_one_sync()) {
        asm volatile("prefetch.tensormap [%0];" :: "l"(&ta));
        asm volatile("prefetch.tensormap [%0];" :: "l"(&tb));
        asm volatile("prefetch.tensormap [%0];" :: "l"(&tsfa));
        asm volatile("prefetch.tensormap [%0];" :: "l"(&tsfb));
        asm volatile("prefetch.tensormap [%0];" :: "l"(&td));
    }
    if (wid == 1 && elect_one_sync()) {
        for (int i = 0; i < NS; i++) {
            mb_init(bf + i * 8, 1);    // TMA arrive
            mb_init(be + i * 8, 1);    // MMA commit
            mb_init(bsf + i * 8, 32);  // warp 2 (32 threads)
        }
        for (int i = 0; i < NE; i++) {
            mb_init(btf + i * 8, 1);    // MMA commit
            mb_init(bte + i * 8, 128);  // epilogue threads
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (wid == 2) {
        int a = __cvta_generic_to_shared(tp);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(TCA));
    }
    __syncthreads();
    const uint32_t taddr = *tp;

    // Block scale idesc: [4-5]=B_SF_ID, [7-9]=atype=E4M3=0, [10-12]=btype=0,
    // [17-22]=N>>3, [23]=scale=UE8M0=1, [27-28]=M>>7
    constexpr uint32_t base_idesc = ((uint32_t)(BN / 8) << 17) | (1u << 23) | ((uint32_t)(BM / 128) << 27);

    // Pipeline state
    int stage = 0, phase = 0;

    // ====== WARP 0: TMA LOAD ======
    if (wid == 0 && elect_one_sync()) {
        int s = 0, ph = 0;
        for (int iter = 0; ; iter++) {
            int bidx = iter * num_sms + blockIdx.x;
            if (bidx >= total) break;
            int bm = bidx / nn, bn = bidx % nn;
            int om = bm * BM, on = bn * BN;
            for (int ki = 0; ki < nk; ki++) {
                mb_wait(be + s * 8, ph ^ 1);
                int mb = bf + s * 8;
                int as_ = __cvta_generic_to_shared(sA(s));
                int bs_ = __cvta_generic_to_shared(sB(s));
                tma_ld(as_, &ta, ki * BK, om, mb);
                tma_ld(bs_, &tb, ki * BK, on, mb);
                int tx = SA + SB;
                // Load SF every SF_PER_LOAD K-blocks
                if (ki % SF_PER_LOAD == 0) {
                    int sfas_ = __cvta_generic_to_shared(sSFA(s));
                    int sfbs_ = __cvta_generic_to_shared(sSFB(s));
                    int sf_k = ki / SF_PER_LOAD;
                    tma_ld(sfas_, &tsfa, om, sf_k, mb);
                    tma_ld(sfbs_, &tsfb, on, sf_k, mb);
                    tx += SFA_SIZE + SFB_SIZE;
                }
                mb_arrive_tx(mb, tx);
                s = (s == NS - 1) ? 0 : s + 1;
                if (s == 0) ph ^= 1;
            }
        }
    }
    // ====== WARP 2: IDLE (SF pre-transposed on host) ======
    // ====== WARP 1: MMA ISSUE ======
    else if (wid == 1 && elect_one_sync()) {
        int s = 0, ph = 0;
        for (int iter = 0; ; iter++) {
            int bidx = iter * num_sms + blockIdx.x;
            if (bidx >= total) break;
            // Wait tmem empty at start of new tile
            int accum_idx = iter % NE;
            int accum_ph = (iter / NE) & 1;
            mb_wait(bte + accum_idx * 8, accum_ph ^ 1);
            asm volatile("tcgen05.fence::after_thread_sync;");

            for (int ki = 0; ki < nk; ki++) {
                // Wait TMA full (SF pre-transposed, no warp 2 needed)
                mb_wait(bf + s * 8, ph);
                asm volatile("tcgen05.fence::after_thread_sync;");

                // UTCCP: copy SF from SMEM to TMEM (only at SF load stages)
                if (ki % SF_PER_LOAD == 0) {
                    int sfas_ = __cvta_generic_to_shared(sSFA(s));
                    int sfbs_ = __cvta_generic_to_shared(sSFB(s));
                    uint64_t sfa_desc = mkdesc_sf(sfas_);
                    uint64_t sfb_desc = mkdesc_sf(sfbs_);
                    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                        :: "r"(TMEM_SFA), "l"(sfa_desc));
                    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                        :: "r"(TMEM_SFB), "l"(sfb_desc));
                }

                // Build idesc with SF ID for current K-block within group
                uint32_t sf_id = ki % SF_PER_LOAD;
                uint32_t idesc = base_idesc | (sf_id << 4) | (sf_id << 29);

                // Issue block_scale MMA with pre-computed base descriptors
                int as_ = __cvta_generic_to_shared(sA(s));
                int bs_ = __cvta_generic_to_shared(sB(s));
                uint64_t ad_base = mkdesc(as_);
                uint64_t bd_base = mkdesc(bs_);
                uint32_t tc = taddr + accum_idx * BN;
                // K-major stride: UK=32 bytes / 16 = 2 in descriptor space
                constexpr uint32_t K_DESC_STRIDE = UK / 16;
                uint32_t en0 = (ki > 0) ? 1u : 0u;
                asm volatile(
                    "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n\t}\n"
                    :: "r"(tc), "l"(ad_base), "l"(bd_base), "r"(idesc), "r"(en0),
                       "r"(TMEM_SFA), "r"(TMEM_SFB));
                #pragma unroll
                for (int k = 1; k < BK / UK; k++) {
                    uint64_t ad = ad_base + k * K_DESC_STRIDE;
                    uint64_t bd = bd_base + k * K_DESC_STRIDE;
                    asm volatile(
                        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], 1;\n"
                        :: "r"(tc), "l"(ad), "l"(bd), "r"(idesc),
                           "r"(TMEM_SFA), "r"(TMEM_SFB));
                }

                // Signal SMEM empty
                asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    :: "r"(be + s * 8) : "memory");

                // Signal tmem_full on last K-iteration
                if (ki == nk - 1) {
                    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                        :: "r"(btf + accum_idx * 8) : "memory");
                }

                s = (s == NS - 1) ? 0 : s + 1;
                if (s == 0) ph ^= 1;
            }
        }
    }
    // ====== WARPS 4-7: EPILOGUE (TMA STORE) ======
    else if (wid >= 4) {
        const int et = tid - 128, ew = wid - 4;
        constexpr int STORE_BN = 64; // 128B swizzle / 2B per BF16
        for (int iter = 0; ; iter++) {
            int bidx = iter * num_sms + blockIdx.x;
            if (bidx >= total) break;
            int bm = bidx / nn, bn = bidx % nn;
            int om = bm * BM, on = bn * BN;

            int accum_idx = iter % NE;
            int accum_ph = (iter / NE) & 1;
            mb_wait(btf + accum_idx * 8, accum_ph);
            asm volatile("tcgen05.fence::after_thread_sync;");

            // TMA store: TMEM -> SMEM (swizzled) -> TMA store to global
            constexpr int NSTORES = BN / STORE_BN; // 2
            #pragma unroll
            for (int si = 0; si < NSTORES; si++) {
                // Wait for previous TMA store to complete (SMEM reuse)
                if (ew == 0) {
                    asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(0) : "memory");
                }
                asm volatile("bar.sync 1, 128;");

                // Each thread: load 8 bank groups from TMEM, convert BF16, write swizzled SMEM
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    uint32_t row = lid;
                    uint32_t col = i ^ (row & 7u);
                    uint32_t tmem_addr = accum_idx * BN + si * STORE_BN + i * 8;
                    uint32_t smem_off = ew * 32 * 128 + row * 128 + col * 16;
                    uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                        : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3),
                          "=r"(v4), "=r"(v5), "=r"(v6), "=r"(v7)
                        : "r"(tmem_addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    uint32_t b0, b1, b2, b3;
                    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b0) : "r"(v0), "r"(v1));
                    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b1) : "r"(v2), "r"(v3));
                    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b2) : "r"(v4), "r"(v5));
                    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b3) : "r"(v6), "r"(v7));
                    uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(sCD + smem_off));
                    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};"
                        :: "r"(sa), "r"(b0), "r"(b1), "r"(b2), "r"(b3) : "memory");
                }

                // Signal TMEM empty at last store chunk
                if (si == NSTORES - 1) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mb_arrive(bte + accum_idx * 8);
                }

                // Fence + sync before TMA store
                asm volatile("fence.proxy.async.shared::cta;");
                asm volatile("bar.sync 1, 128;");

                // Issue TMA store (one thread)
                if (ew == 0 && elect_one_sync()) {
                    tma_store_2d(&td, sCD, on + si * STORE_BN, om);
                    asm volatile("cp.async.bulk.commit_group;");
                }
            }
        }
        // Final flush
        if (ew == 0) {
            asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(0) : "memory");
        }
    }

    __syncthreads();
    if (wid == 0)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(TCA));
#endif
}

// Host
void chk(cudaError_t e, const char* m) { if (e != cudaSuccess) { fprintf(stderr, "CUDA(%s):%s\n", m, cudaGetErrorString(e)); exit(1); } }
void chk(CUresult e, const char* m) { if (e != CUDA_SUCCESS) { const char* s; cuGetErrorString(e, &s); fprintf(stderr, "CU(%s):%s\n", m, s); exit(1); } }

// Convert float32 scale to UE8M0 (ceil to power of 2, extract exponent)
uint8_t float_to_ue8m0(float val) {
    float p2 = exp2f(ceilf(log2f(fabsf(val))));
    uint32_t bits;
    memcpy(&bits, &p2, 4);
    return (bits >> 23) & 0xFF;
}

// Prepare UE8M0 packed scale array with UTCCP pre-transpose
// Input: float scales[dim, nk] row-major
// Output: uint32_t packed[num_sf_k, dim] column-major with in-block UTCCP transpose
void prepare_sf(const float* scales, uint32_t* packed, int dim, int nk) {
    int num_sf_k = (nk + 3) / 4;
    // First pack normally into temp buffer
    std::vector<uint32_t> temp(num_sf_k * dim);
    for (int d = 0; d < dim; d++) {
        for (int sk = 0; sk < num_sf_k; sk++) {
            uint32_t val = 0;
            for (int j = 0; j < 4; j++) {
                int ki = sk * 4 + j;
                uint8_t ue = (ki < nk) ? float_to_ue8m0(scales[d * nk + ki]) : 0;
                val |= ((uint32_t)ue) << (j * 8);
            }
            temp[sk * dim + d] = val;
        }
    }
    // Apply UTCCP transpose: for each 128-element block, rearrange from 4x32 to 32x4 layout
    // The UTCCP transpose permutation: src[(j ^ (lid>>3))*32 + lid] -> dst[lid*4 + (j ^ (lid>>3))]
    for (int sk = 0; sk < num_sf_k; sk++) {
        for (int blk = 0; blk < dim; blk += 128) {
            uint32_t* src = &temp[sk * dim + blk];
            uint32_t block[128];
            // Apply transpose permutation
            for (int lid = 0; lid < 32; lid++) {
                int xor_val = lid >> 3;
                for (int j = 0; j < 4; j++) {
                    int src_idx = ((j ^ xor_val) * 32) + lid;
                    int dst_idx = lid * 4 + (j ^ xor_val);
                    block[dst_idx] = (src_idx < 128 && (blk + src_idx) < dim) ? src[src_idx] : 0;
                }
            }
            for (int i = 0; i < 128 && (blk + i) < dim; i++) {
                packed[sk * dim + blk + i] = block[i];
            }
        }
    }
}

// Expand weight scales from [N/128, nk] to [N, nk] (duplicate per 128 N-group)
void expand_sb(const float* sb, float* sb_expanded, int N, int nk) {
    int nb = (N + 127) / 128;
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < nk; k++) {
            sb_expanded[n * nk + k] = sb[(n / 128) * nk + k];
        }
    }
}

template <int BN, int NS>
void run_bs(const void* A, const void* B, const uint32_t* sfa_packed, const uint32_t* sfb_packed,
            __nv_bfloat16* C, int M, int N, int K) {
    int nk = (K + 127) / 128;
    int num_sf_k = (nk + 3) / 4;

    CUtensorMap ta, tb, tsfa, tsfb, td;
    { // A: [M, K] fp8
        uint64_t g[2] = {(uint64_t)K, (uint64_t)M}; uint64_t s[1] = {(uint64_t)K};
        uint32_t b[2] = {128, 128}; uint32_t e[2] = {1, 1};
        chk(cuTensorMapEncodeTiled(&ta, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)A, g, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "ta");
    }
    { // B: [N, K] fp8
        uint64_t g[2] = {(uint64_t)K, (uint64_t)N}; uint64_t s[1] = {(uint64_t)K};
        uint32_t b[2] = {128, (uint32_t)BN}; uint32_t e[2] = {1, 1};
        chk(cuTensorMapEncodeTiled(&tb, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)B, g, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "tb");
    }
    { // SFA: [num_sf_k, M] uint32_t (M contiguous), load box = [128, 1]
        uint64_t g[2] = {(uint64_t)M, (uint64_t)num_sf_k};
        uint64_t s[1] = {(uint64_t)M * sizeof(uint32_t)};
        uint32_t b[2] = {128, 1}; uint32_t e[2] = {1, 1};
        chk(cuTensorMapEncodeTiled(&tsfa, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, (void*)sfa_packed, g, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "tsfa");
    }
    { // SFB: [num_sf_k, N] uint32_t (N contiguous), load box = [128, 1]
        uint64_t g[2] = {(uint64_t)N, (uint64_t)num_sf_k};
        uint64_t s[1] = {(uint64_t)N * sizeof(uint32_t)};
        uint32_t b[2] = {128, 1}; uint32_t e[2] = {1, 1};
        chk(cuTensorMapEncodeTiled(&tsfb, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, (void*)sfb_packed, g, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "tsfb");
    }
    { // D (output): [M, N] bf16
        uint64_t g[2] = {(uint64_t)N, (uint64_t)M};
        uint64_t s[1] = {(uint64_t)N * sizeof(__nv_bfloat16)};
        uint32_t b[2] = {64, 128}; uint32_t e[2] = {1, 1};
        chk(cuTensorMapEncodeTiled(&td, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)C, g, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "td");
    }

    int num_sms; cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    int total = ((M + 127) / 128) * ((N + BN - 1) / BN);
    int grid = std::min(total, num_sms);

    constexpr int NE = 2;
    int smem = NS * (128*128 + BN*128 + 128*4 + 128*4) + 128*128; // +CD stage
    smem = (smem + 7) & ~7;
    smem += (NS*3 + NE*2)*8 + 8;
    smem = (smem + 1023) & ~1023;

    auto k = fp8_gemm_bs<BN, NS>;
    if (smem > 48000) chk(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "sm");
    k<<<grid, 256, smem>>>(ta, tb, tsfa, tsfb, td, C, M, N, K, num_sms);
}

struct Config { const char* name; int M, K, N; };

int main() {
    cuInit(0);
    constexpr int BN = 128, NS = 6;
    Config configs[] = {
        {"q_b_proj_M512",512,1536,3072}, {"kv_b_proj_M512",512,512,4096}, {"o_proj_M512",512,2048,7168},
        {"q_b_proj_M1024",1024,1536,3072}, {"kv_b_proj_M1024",1024,512,4096}, {"o_proj_M1024",1024,2048,7168},
        {"q_b_proj_M2048",2048,1536,3072}, {"kv_b_proj_M2048",2048,512,4096}, {"o_proj_M2048",2048,2048,7168},
        {"q_b_proj_M4096",4096,1536,3072}, {"kv_b_proj_M4096",4096,512,4096}, {"o_proj_M4096",4096,2048,7168},
        {"q_b_proj_M8192",8192,1536,3072}, {"kv_b_proj_M8192",8192,512,4096}, {"o_proj_M8192",8192,2048,7168},
    };
    int ncfg = sizeof(configs) / sizeof(configs[0]);
    printf("KERNEL_RESULT {"); bool first = true;
    for (int ci = 0; ci < ncfg; ci++) {
        auto& c = configs[ci];
        int M = c.M, K = c.K, N = c.N;
        int nk = (K + 127) / 128, nb = (N + 127) / 128;
        int num_sf_k = (nk + 3) / 4;
        size_t a_sz = (size_t)M*K, b_sz = (size_t)N*K, c_sz = (size_t)M*N;
        size_t sa_sz = (size_t)M*nk, sb_sz = (size_t)nb*nk;

        // Host data
        std::vector<uint8_t> hA(a_sz), hB(b_sz);
        std::vector<float> hsa(sa_sz), hsb(sb_sz);
        srand(42 + ci);
        for (auto& v : hA) v = rand() % 256;
        for (auto& v : hB) v = rand() % 256;
        for (auto& v : hsa) v = 0.5f + (rand() % 100) / 200.0f;
        for (auto& v : hsb) v = 0.5f + (rand() % 100) / 200.0f;

        // Prepare UE8M0 packed scales
        // SFA: [M, nk] -> packed [num_sf_k, M]
        std::vector<uint32_t> sfa_packed(num_sf_k * M);
        prepare_sf(hsa.data(), sfa_packed.data(), M, nk);
        // SFB: expand [nb, nk] -> [N, nk], then pack [num_sf_k, N]
        std::vector<float> sb_exp(N * nk);
        expand_sb(hsb.data(), sb_exp.data(), N, nk);
        std::vector<uint32_t> sfb_packed(num_sf_k * N);
        prepare_sf(sb_exp.data(), sfb_packed.data(), N, nk);

        // Device alloc
        void *dA, *dB; uint32_t *dsfa, *dsfb; __nv_bfloat16 *dC;
        chk(cudaMalloc(&dA, a_sz), ""); chk(cudaMalloc(&dB, b_sz), "");
        chk(cudaMalloc(&dsfa, sfa_packed.size()*4), "");
        chk(cudaMalloc(&dsfb, sfb_packed.size()*4), "");
        chk(cudaMalloc(&dC, c_sz*2), "");
        chk(cudaMemcpy(dA, hA.data(), a_sz, cudaMemcpyHostToDevice), "");
        chk(cudaMemcpy(dB, hB.data(), b_sz, cudaMemcpyHostToDevice), "");
        chk(cudaMemcpy(dsfa, sfa_packed.data(), sfa_packed.size()*4, cudaMemcpyHostToDevice), "");
        chk(cudaMemcpy(dsfb, sfb_packed.data(), sfb_packed.size()*4, cudaMemcpyHostToDevice), "");
        chk(cudaMemset(dC, 0, c_sz*2), "");

        // Run + check
        run_bs<BN, NS>(dA, dB, dsfa, dsfb, dC, M, N, K);
        auto err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: LAUNCH ERROR: %s\n", c.name, cudaGetErrorString(err));
            if (!first) printf(", "); first = false;
            printf("\"%s\": 0.0", c.name);
            cudaFree(dA); cudaFree(dB); cudaFree(dsfa); cudaFree(dsfb); cudaFree(dC);
            continue;
        }

        // Warmup
        for (int i = 0; i < 20; i++) run_bs<BN, NS>(dA, dB, dsfa, dsfb, dC, M, N, K);
        chk(cudaDeviceSynchronize(), "");
        // Bench
        size_t fsz = 128*1024*1024; char* df; chk(cudaMalloc(&df, fsz), "");
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        std::vector<float> ts(100);
        for (int it = 0; it < 100; it++) {
            chk(cudaMemset(df, 0, fsz), "");
            cudaEventRecord(t0);
            run_bs<BN, NS>(dA, dB, dsfa, dsfb, dC, M, N, K);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms, t0, t1); ts[it] = ms;
        }
        std::sort(ts.begin(), ts.end()); float med = ts[50];
        double tflops = 2.0*M*N*K/(med/1000.0)/1e12;
        if (!first) printf(", "); first = false;
        printf("\"%s\": %.4f", c.name, tflops);
        fprintf(stderr, "%s: %.4f TFLOPS, %.1f us\n", c.name, tflops, med*1000);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(df); cudaFree(dA); cudaFree(dB); cudaFree(dsfa); cudaFree(dsfb); cudaFree(dC);
    }
    printf("}\n");
    return 0;
}
