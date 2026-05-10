// Grouped FP8 Block-Scale GEMM for DeepSeek V3 MoE DECODE (SM100a)
// Hardware block_scale MMA: tcgen05.mma.kind::mxf8f6f4.block_scale
// UE8M0 scales in TMEM, UTCCP copy from SMEM
// Based on v009 example architecture adapted for grouped GEMM
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
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred) : "r"(0xFFFFFFFF));
    return pred;
}
__device__ __forceinline__ void mb_init(int a, int c) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(c));
}
__device__ __forceinline__ void mb_wait(int a, int p) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LW:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DN;\n\t"
        "bra.uni LW;\n\t"
        "DN:\n\t"
        "}" :: "r"(a), "r"(p), "r"(0x989680));
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
__device__ __forceinline__ constexpr uint64_t denc(uint64_t x) { return (x & 0x3FFFFULL) >> 4ULL; }
__device__ __forceinline__ uint64_t mkdesc(int a) {
    return denc(a) | (denc(1024) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
}
__device__ __forceinline__ uint64_t mkdesc_sf(int a) {
    return denc(a) | (denc(128) << 32ULL) | (1ULL << 46ULL);
}
__device__ __forceinline__ uint32_t ld_shared_u32(const void* p) {
    uint32_t r; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"((uint32_t)__cvta_generic_to_shared(p))); return r;
}
__device__ __forceinline__ void st_shared_u32(void* p, uint32_t v) {
    asm volatile("st.shared.u32 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(p)), "r"(v));
}

template <int BN, int NS>
__global__ void __launch_bounds__(256, 1)
grouped_fp8_gemm_bs(
    const __grid_constant__ CUtensorMap ta,    // A: [K, M_total] fp8
    const __grid_constant__ CUtensorMap tb,    // B: [K, E*N] fp8
    const __grid_constant__ CUtensorMap tsfa,  // SFA: [M_total, num_sf_k] uint32 (packed UE8M0)
    const __grid_constant__ CUtensorMap tsfb,  // SFB: [E*N, num_sf_k] uint32 (packed UE8M0)
    const __grid_constant__ CUtensorMap td,    // D: [N, M_total] bf16 (for TMA store, N-major)
    const int* __restrict__ m_indices,
    int M_total, int N, int K, int E, int num_sms
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
    constexpr int BM = 128, BK = 128, UK = 32;
    constexpr int SF_PER_LOAD = 4; // 4 UE8M0 packed per uint32_t
    constexpr int NE = 2;
    const int tid = threadIdx.x, wid = tid / 32;
    const uint32_t lid = tid % 32;
    const int nk = (K + BK - 1) / BK;
    const int nn = (N + BN - 1) / BN;
    const int nm = (M_total + BM - 1) / BM;
    const int total = nm * nn;

    extern __shared__ __align__(1024) uint8_t sm[];
    constexpr int STORE_BN = 64;
    constexpr int NUM_TMA_ST = 1;
    constexpr int SCD_STAGE = BM * STORE_BN * 2;
    constexpr int SCD_TOT = SCD_STAGE * NUM_TMA_ST;
    constexpr int SA = BM * BK, SB = BN * BK;
    constexpr int SFA_SIZE = 128 * 4;
    constexpr int SFB_SIZE = BN * 4;
    auto sCD = [&](int s) -> uint8_t* { return sm + s * SCD_STAGE; };
    auto sA = [&](int s) -> uint8_t* { return sm + SCD_TOT + s * SA; };
    auto sB = [&](int s) -> uint8_t* { return sm + SCD_TOT + NS * SA + s * SB; };
    auto sSFA = [&](int s) -> uint32_t* { return (uint32_t*)(sm + SCD_TOT + NS * (SA + SB) + s * SFA_SIZE); };
    auto sSFB = [&](int s) -> uint32_t* { return (uint32_t*)(sm + SCD_TOT + NS * (SA + SB) + NS * SFA_SIZE + s * SFB_SIZE); };
    int bar_base = SCD_TOT + NS * (SA + SB + SFA_SIZE + SFB_SIZE);
    bar_base = (bar_base + 7) & ~7;
    auto bars = reinterpret_cast<uint64_t*>(sm + bar_base);
    int bf = __cvta_generic_to_shared(bars);
    int be = bf + NS * 8;
    int bsf = be + NS * 8;
    int btf = bsf + NS * 8;
    int bte = btf + NE * 8;
    auto tp = reinterpret_cast<uint32_t*>(bars + NS * 3 + NE * 2);

    constexpr int SF_BLOCK_M = 128;
    constexpr int SF_BLOCK_N = ((BN + 127) / 128) * 128;
    constexpr int TMEM_SFA_COLS = SF_BLOCK_M / 32;
    constexpr int TMEM_SFB_COLS = SF_BLOCK_N / 32;
    constexpr int TMEM_SFA = NE * BN;
    constexpr int TMEM_SFB = TMEM_SFA + TMEM_SFA_COLS;
    constexpr int TMEM_TOTAL = NE * BN + TMEM_SFA_COLS + TMEM_SFB_COLS;
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
            mb_init(bf + i * 8, 1);
            mb_init(be + i * 8, 1);
            mb_init(bsf + i * 8, 32);
        }
        for (int i = 0; i < NE; i++) {
            mb_init(btf + i * 8, 1);
            mb_init(bte + i * 8, 128);
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (wid == 2) {
        int a = __cvta_generic_to_shared(tp);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(TCA));
    }
    __syncthreads();
    const uint32_t taddr = *tp;
    const int num_blocks = gridDim.x;

    if (wid < 4) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");
    }

    constexpr uint32_t base_idesc = ((uint32_t)(BN / 8) << 17) | (1u << 23) | ((uint32_t)(BM / 128) << 27);

    // ====== WARP 0: TMA LOAD ======
    if (wid == 0 && elect_one_sync()) {
        int gki = 0;
        for (int iter = 0; ; iter++) {
            int bidx = iter * num_blocks + blockIdx.x;
            if (bidx >= total) break;
            int bm = bidx / nn, bn = bidx % nn;
            int m_start = bm * BM;
            int expert_id = (m_start < M_total) ? __ldg(m_indices + m_start) : 0;
            int om = m_start;
            int on = expert_id * N + bn * BN;

            for (int ki = 0; ki < nk; ki++, gki++) {
                int s = gki % NS;
                int ph = (gki / NS) & 1;
                mb_wait(be + s * 8, ph ^ 1);
                int mb = bf + s * 8;
                int as_ = __cvta_generic_to_shared(sA(s));
                int bs_ = __cvta_generic_to_shared(sB(s));
                tma_ld(as_, &ta, ki * BK, om, mb);
                tma_ld(bs_, &tb, ki * BK, on, mb);
                int tx = SA + SB;
                if (ki % SF_PER_LOAD == 0) {
                    int sfas_ = __cvta_generic_to_shared(sSFA(s));
                    int sfbs_ = __cvta_generic_to_shared(sSFB(s));
                    int sf_k = ki / SF_PER_LOAD;
                    tma_ld(sfas_, &tsfa, om, sf_k, mb);
                    tma_ld(sfbs_, &tsfb, on, sf_k, mb);
                    tx += SFA_SIZE + SFB_SIZE;
                }
                mb_arrive_tx(mb, tx);
            }
        }
    }
    // ====== WARP 2: UTCCP TRANSPOSE ======
    else if (wid == 2) {
        auto utccp_transpose = [&](uint32_t* ptr) {
            uint32_t v[4];
            #pragma unroll
            for (int i = 0; i < 4; i++)
                v[i] = ld_shared_u32(ptr + (i ^ (lid >> 3)) * 32 + lid);
            __syncwarp();
            #pragma unroll
            for (int i = 0; i < 4; i++)
                st_shared_u32(ptr + lid * 4 + (i ^ (lid >> 3)), v[i]);
        };

        int gki = 0;
        for (int iter = 0; ; iter++) {
            int bidx = iter * num_blocks + blockIdx.x;
            if (bidx >= total) break;
            for (int ki = 0; ki < nk; ki++, gki++) {
                int s = gki % NS;
                int ph = (gki / NS) & 1;
                mb_wait(bf + s * 8, ph);
                if (ki % SF_PER_LOAD == 0) {
                    utccp_transpose(sSFA(s));
                    for (int b = 0; b < SF_BLOCK_N; b += 128)
                        utccp_transpose(sSFB(s) + b);
                    asm volatile("fence.proxy.async.shared::cta;");
                }
                mb_arrive(bsf + s * 8);
            }
        }
    }
    // ====== WARP 1: MMA ISSUE ======
    else if (wid == 1 && elect_one_sync()) {
        int gki = 0;
        for (int iter = 0; ; iter++) {
            int bidx = iter * num_blocks + blockIdx.x;
            if (bidx >= total) break;
            int accum_idx = iter % NE;
            int accum_ph = (iter / NE) & 1;
            mb_wait(bte + accum_idx * 8, accum_ph ^ 1);

            for (int ki = 0; ki < nk; ki++, gki++) {
                int s = gki % NS;
                int ph = (gki / NS) & 1;
                mb_wait(bsf + s * 8, ph);
                asm volatile("tcgen05.fence::after_thread_sync;");

                if (ki % SF_PER_LOAD == 0) {
                    int sfas_ = __cvta_generic_to_shared(sSFA(s));
                    uint64_t sfa_desc = mkdesc_sf(sfas_);
                    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                        :: "r"(TMEM_SFA), "l"(sfa_desc));
                    for (int b = 0; b < SF_BLOCK_N / 128; b++) {
                        int sfbs_ = __cvta_generic_to_shared(sSFB(s) + b * 128);
                        uint64_t sfb_desc = mkdesc_sf(sfbs_);
                        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                            :: "r"(TMEM_SFB + b * 4), "l"(sfb_desc));
                    }
                }

                uint32_t sf_id = ki % SF_PER_LOAD;
                uint32_t idesc = base_idesc | (sf_id << 4) | (sf_id << 29);
                int as_ = __cvta_generic_to_shared(sA(s));
                int bs_ = __cvta_generic_to_shared(sB(s));
                uint32_t tc = taddr + accum_idx * BN;

                for (int k = 0; k < BK / UK; k++) {
                    uint64_t ad = mkdesc(as_ + k * UK);
                    uint64_t bd = mkdesc(bs_ + k * UK);
                    uint32_t en = (ki > 0 || k > 0) ? 1u : 0u;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.ne.b32 p, %4, 0;\n\t"
                        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n\t"
                        "}\n"
                        :: "r"(tc), "l"(ad), "l"(bd), "r"(idesc), "r"(en),
                           "r"(TMEM_SFA), "r"(TMEM_SFB));
                }

                asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    :: "r"(be + s * 8) : "memory");
                if (ki == nk - 1) {
                    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                        :: "r"(btf + accum_idx * 8) : "memory");
                }
            }
        }
    }
    // ====== WARPS 4-7: EPILOGUE (TMEM → SMEM → TMA STORE) ======
    else if (wid >= 4) {
        const int ew = wid - 4;
        uint32_t tma_st = 0;
        for (int iter = 0; ; iter++) {
            int bidx = iter * num_blocks + blockIdx.x;
            if (bidx >= total) break;
            int bm = bidx / nn, bn = bidx % nn;
            int m_start = bm * BM;
            int expert_id = (m_start < M_total) ? __ldg(m_indices + m_start) : 0;
            int om = m_start;
            int on = bn * BN; // Output N offset (not expert-shifted, output is [M_total, N])
            int accum_idx = iter % NE;
            int accum_ph = (iter / NE) & 1;
            mb_wait(btf + accum_idx * 8, accum_ph);
            asm volatile("tcgen05.fence::after_thread_sync;");
            constexpr int NUM_N_ST = BN / STORE_BN;
            #pragma unroll
            for (int si = 0; si < NUM_N_ST; si++) {
                if (ew == 0) asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(NUM_TMA_ST - 1) : "memory");
                asm volatile("bar.sync 0, 128;");
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    uint32_t row = lid, col = i ^ (row & 7u);
                    uint32_t tc = accum_idx * BN + si * STORE_BN + i * 8;
                    uint32_t so = ew * 32 * 128 + row * 128 + col * 16;
                    uint32_t v0,v1,v2,v3,v4,v5,v6,v7;
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                        :"=r"(v0),"=r"(v1),"=r"(v2),"=r"(v3),"=r"(v4),"=r"(v5),"=r"(v6),"=r"(v7):"r"(tc));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    uint32_t b0,b1,b2,b3;
                    asm("cvt.rn.bf16x2.f32 %0, %2, %1;":"=r"(b0):"r"(v0),"r"(v1));
                    asm("cvt.rn.bf16x2.f32 %0, %2, %1;":"=r"(b1):"r"(v2),"r"(v3));
                    asm("cvt.rn.bf16x2.f32 %0, %2, %1;":"=r"(b2):"r"(v4),"r"(v5));
                    asm("cvt.rn.bf16x2.f32 %0, %2, %1;":"=r"(b3):"r"(v6),"r"(v7));
                    uint32_t sa = __cvta_generic_to_shared(sCD(tma_st) + so);
                    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(sa),"r"(b0),"r"(b1),"r"(b2),"r"(b3):"memory");
                }
                if (si == NUM_N_ST - 1) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mb_arrive(bte + accum_idx * 8);
                }
                asm volatile("fence.proxy.async.shared::cta;");
                asm volatile("bar.sync 0, 128;");
                if (ew == 0 && elect_one_sync()) {
                    uint64_t dd = reinterpret_cast<uint64_t>(&td);
                    uint32_t sp = __cvta_generic_to_shared(sCD(tma_st));
                    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
                        ::"l"(dd),"r"(sp),"r"(on + si * STORE_BN),"r"(om):"memory");
                    asm volatile("cp.async.bulk.commit_group;");
                }
                tma_st = (tma_st + 1) % NUM_TMA_ST;
            }
        }
        if (ew == 0) asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(0) : "memory");
        if (ew == 1) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(TCA));
    }
#endif
}

// Host helpers
void chk(cudaError_t e, const char* m) { if (e != cudaSuccess) { fprintf(stderr, "CUDA(%s):%s\n", m, cudaGetErrorString(e)); exit(1); } }
void chk(CUresult e, const char* m) { if (e != CUDA_SUCCESS) { const char* s; cuGetErrorString(e, &s); fprintf(stderr, "CU(%s):%s\n", m, s); exit(1); } }

uint8_t float_to_ue8m0(float val) {
    if (val <= 0.0f) return 0;
    float p2 = exp2f(roundf(log2f(val)));
    uint32_t bits;
    memcpy(&bits, &p2, 4);
    return (bits >> 23) & 0xFF;
}

float ue8m0_to_float(uint8_t e) {
    if (e == 0) return 0.0f;
    uint32_t bits = (uint32_t)e << 23;
    float val;
    memcpy(&val, &bits, 4);
    return val;
}

// Pack float32 scales [dim, nk] → UE8M0 packed [ceil(nk/4), dim] column-major
void prepare_sf(const float* scales, uint32_t* packed, int dim, int nk) {
    int num_sf_k = (nk + 3) / 4;
    for (int d = 0; d < dim; d++) {
        for (int sk = 0; sk < num_sf_k; sk++) {
            uint32_t val = 0;
            for (int j = 0; j < 4; j++) {
                int ki = sk * 4 + j;
                uint8_t ue = (ki < nk) ? float_to_ue8m0(scales[d * nk + ki]) : 0;
                val |= ((uint32_t)ue) << (j * 8);
            }
            packed[sk * dim + d] = val;
        }
    }
}

template <int BN, int NS>
void launch_grouped_gemm_bs(
    const void* A, const void* B,
    const uint32_t* sfa_packed, const uint32_t* sfb_packed,
    __nv_bfloat16* output, const int* m_indices,
    int M_total, int N, int K, int E
) {
    int nk = (K + 127) / 128;
    int num_sf_k = (nk + 3) / 4;
    constexpr int BM = 128;

    CUtensorMap ta, tb, tsfa, tsfb, td_map;
    { // A: [K, M_total] fp8 - use L2 promotion for A data reuse across N-tiles
        uint64_t g[2] = {(uint64_t)K, (uint64_t)M_total}; uint64_t s[1] = {(uint64_t)K};
        uint32_t b[2] = {128, 128}; uint32_t e[2] = {1, 1};
        chk(cuTensorMapEncodeTiled(&ta, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)A, g, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "ta");
    }
    { // B: [K, E*N] fp8
        uint64_t g[2] = {(uint64_t)K, (uint64_t)E*N}; uint64_t s[1] = {(uint64_t)K};
        uint32_t b[2] = {128, (uint32_t)BN}; uint32_t e[2] = {1, 1};
        chk(cuTensorMapEncodeTiled(&tb, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)B, g, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "tb");
    }
    { // SFA: [M_total, num_sf_k] uint32 packed - L2 promote (reused across N-tiles)
        uint64_t g[2] = {(uint64_t)M_total, (uint64_t)num_sf_k};
        uint64_t s[1] = {(uint64_t)M_total * sizeof(uint32_t)};
        uint32_t b[2] = {128, 1}; uint32_t e[2] = {1, 1};
        chk(cuTensorMapEncodeTiled(&tsfa, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, (void*)sfa_packed, g, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "tsfa");
    }
    { // SFB: [E*N, num_sf_k] uint32 packed (expanded per-element)
        uint64_t g[2] = {(uint64_t)E*N, (uint64_t)num_sf_k};
        uint64_t s[1] = {(uint64_t)E*N * sizeof(uint32_t)};
        uint32_t b[2] = {(uint32_t)BN, 1}; uint32_t e[2] = {1, 1};
        chk(cuTensorMapEncodeTiled(&tsfb, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, (void*)sfb_packed, g, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "tsfb");
    }
    { // D: [N, M_total] bf16
        uint64_t g[2] = {(uint64_t)N, (uint64_t)M_total};
        uint64_t s[1] = {(uint64_t)N * sizeof(__nv_bfloat16)};
        uint32_t b[2] = {64, 128}; uint32_t e[2] = {1, 1};
        chk(cuTensorMapEncodeTiled(&td_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)output, g, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "td");
    }

    int num_sms; cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    int total_blocks = ((M_total + BM - 1) / BM) * ((N + BN - 1) / BN);
    int grid = std::min(total_blocks, num_sms);
    if (grid <= 0) grid = 1;

    constexpr int NE = 2;
    constexpr int NUM_TMA_ST_HOST = 1;
    int smem = NUM_TMA_ST_HOST*BM*64*2 + NS*(BM*128 + BN*128 + 128*4 + BN*4);
    smem = (smem + 7) & ~7;
    smem += (NS*3 + NE*2)*8 + 8;
    smem = (smem + 1023) & ~1023;

    auto k = grouped_fp8_gemm_bs<BN, NS>;
    if (smem > 48000) chk(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "sm");
    k<<<grid, 256, smem>>>(ta, tb, tsfa, tsfb, td_map, m_indices, M_total, N, K, E, num_sms);
}

// CPU reference using UE8M0-rounded scales
void cpu_ref_bs(const uint8_t* A, const uint8_t* B,
                const float* a_scale, const float* w_scale_expanded,
                float* C, const int* m_indices,
                int M_total, int N, int K, int E) {
    int nk = (K + 127) / 128;
    int BM = 128;
    for (int m = 0; m < M_total; m++) {
        int m_block_start = (m / BM) * BM;
        int expert_id = m_indices[m_block_start];
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int kk = 0; kk < nk; kk++) {
                float sfa = ue8m0_to_float(float_to_ue8m0(a_scale[m * nk + kk]));
                float sfb = ue8m0_to_float(float_to_ue8m0(w_scale_expanded[(long long)(expert_id * N + n) * nk + kk]));
                float dot = 0;
                for (int k = kk * 128; k < std::min((kk + 1) * 128, K); k++) {
                    float av = __half2float(__nv_cvt_fp8_to_halfraw(A[m * K + k], __NV_E4M3));
                    float bv = __half2float(__nv_cvt_fp8_to_halfraw(B[(long long)(expert_id * N + n) * K + k], __NV_E4M3));
                    dot += av * bv;
                }
                sum += dot * sfa * sfb;
            }
            C[m * N + n] = sum;
        }
    }
}

struct Cfg { const char* name; int M_per_expert, num_groups, K, N; };

int main() {
    cuInit(0);
    Cfg cfgs[] = {
        {"gate_up_M1",  1, 32, 7168, 4096}, {"gate_up_M4",  4, 32, 7168, 4096},
        {"gate_up_M8",  8, 32, 7168, 4096}, {"gate_up_M16",16, 32, 7168, 4096},
        {"down_M1",     1, 32, 2048, 7168}, {"down_M4",     4, 32, 2048, 7168},
        {"down_M8",     8, 32, 2048, 7168}, {"down_M16",   16, 32, 2048, 7168},
    };
    int ncfg = sizeof(cfgs)/sizeof(cfgs[0]);
    size_t flush_sz = 128*1024*1024; void* d_flush; chk(cudaMalloc(&d_flush, flush_sz), "flush");

    printf("KERNEL_RESULT {"); bool first = true;
    for (int ci = 0; ci < ncfg; ci++) {
        auto& c = cfgs[ci];
        int M_per_e = c.M_per_expert, E = c.num_groups, K = c.K, N = c.N;
        int M_total = M_per_e * E;
        int nk = (K+127)/128, nscale = (N+127)/128;
        int num_sf_k = (nk+3)/4;

        size_t A_sz = (size_t)M_total*K, B_sz = (size_t)E*N*K, C_sz = (size_t)M_total*N;
        void *dA, *dB; uint32_t *d_sfa_packed, *d_sfb_packed;
        __nv_bfloat16 *dC; int *d_mi;
        chk(cudaMalloc(&dA, A_sz), ""); chk(cudaMalloc(&dB, B_sz), "");
        chk(cudaMalloc(&d_sfa_packed, (size_t)num_sf_k*M_total*4), "");
        chk(cudaMalloc(&d_sfb_packed, (size_t)num_sf_k*E*N*4), "");
        chk(cudaMalloc(&dC, C_sz*2), ""); chk(cudaMalloc(&d_mi, M_total*4), "");

        std::vector<uint8_t> hA(A_sz), hB(B_sz);
        std::vector<float> h_ascale(M_total*nk), h_wscale(E*nscale*nk);
        std::vector<int> h_mi(M_total);
        srand(42+ci);
        for(auto&v:hA){v=rand()%254;if(v>=0x7F)v++;}
        for(auto&v:hB){v=rand()%254;if(v>=0x7F)v++;}
        for(auto&v:h_ascale)v=0.5f+(rand()%100)/200.0f;
        for(auto&v:h_wscale)v=0.5f+(rand()%100)/200.0f;
        for(int i=0;i<M_total;i++)h_mi[i]=i/M_per_e;

        // Expand weight scales from [E, nscale, nk] to [E*N, nk]
        std::vector<float> w_exp((size_t)E*N*nk);
        for(int e=0;e<E;e++) for(int n=0;n<N;n++) for(int k=0;k<nk;k++)
            w_exp[((long long)e*N+n)*nk+k] = h_wscale[e*nscale*nk+(n/128)*nk+k];

        // Prepare packed UE8M0 scales
        std::vector<uint32_t> sfa_packed(num_sf_k*M_total);
        prepare_sf(h_ascale.data(), sfa_packed.data(), M_total, nk);
        std::vector<uint32_t> sfb_packed((size_t)num_sf_k*E*N);
        prepare_sf(w_exp.data(), sfb_packed.data(), E*N, nk);

        chk(cudaMemcpy(dA, hA.data(), A_sz, cudaMemcpyHostToDevice), "");
        chk(cudaMemcpy(dB, hB.data(), B_sz, cudaMemcpyHostToDevice), "");
        chk(cudaMemcpy(d_sfa_packed, sfa_packed.data(), sfa_packed.size()*4, cudaMemcpyHostToDevice), "");
        chk(cudaMemcpy(d_sfb_packed, sfb_packed.data(), sfb_packed.size()*4, cudaMemcpyHostToDevice), "");
        chk(cudaMemcpy(d_mi, h_mi.data(), M_total*4, cudaMemcpyHostToDevice), "");
        chk(cudaMemset(dC, 0, C_sz*2), "");

        // Auto-select BN/NS based on problem shape
        // gate_up M1-M8 (K=7168, small M): BN=64/NS=8 for more tiles + deeper pipeline
        // gate_up M16 (K=7168, larger M): BN=128/NS=6 for better compute efficiency
        // down (K=2048): BN=128/NS=6 always
        auto run_kernel = [&]() {
            if (K > 4096 && M_per_e <= 8)
                launch_grouped_gemm_bs<64, 8>(dA, dB, d_sfa_packed, d_sfb_packed, dC, d_mi, M_total, N, K, E);
            else
                launch_grouped_gemm_bs<128, 6>(dA, dB, d_sfa_packed, d_sfb_packed, dC, d_mi, M_total, N, K, E);
        };
        run_kernel();
        auto err = cudaDeviceSynchronize();

        bool valid = false; float me = 999.0f;
        if (err == cudaSuccess) {
            int checkM = std::min(M_total, 32);
            std::vector<float> hr((size_t)checkM*N);
            cpu_ref_bs(hA.data(), hB.data(), h_ascale.data(), w_exp.data(),
                       hr.data(), h_mi.data(), checkM, N, K, E);
            std::vector<__nv_bfloat16> hC(C_sz);
            chk(cudaMemcpy(hC.data(), dC, C_sz*2, cudaMemcpyDeviceToHost), "");
            float max_abs=0; for(int i=0;i<checkM*N;i++){float r=hr[i];if(!isnan(r)&&fabsf(r)>max_abs)max_abs=fabsf(r);}
            float atol=max_abs*1e-3f; me=0; int nc=0;
            for(int i=0;i<checkM*N;i++){float g=__bfloat162float(hC[i]),r=hr[i];if(isnan(g)||isnan(r)){nc++;continue;}
                float d=fmaxf(fabsf(r),atol),e=fabsf(g-r)/d;if(e>me)me=e;}
            valid=(me<0.01f&&nc==0);
            fprintf(stderr,"%s: err=%.6f nan=%d %s\n",c.name,me,nc,valid?"PASS":"FAIL");
        } else { fprintf(stderr,"%s: LAUNCH: %s\n",c.name,cudaGetErrorString(err)); }

        double tflops = 0.0;
        if (valid) {
            for(int i=0;i<20;i++) run_kernel();
            chk(cudaDeviceSynchronize(),"");
            int NI=100; std::vector<float> times(NI);
            for(int it=0;it<NI;it++){
                chk(cudaMemset(d_flush,0,flush_sz),"");
                cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);
                cudaEventRecord(t0);
                run_kernel();
                cudaEventRecord(t1);cudaEventSynchronize(t1);
                float ms;cudaEventElapsedTime(&ms,t0,t1);times[it]=ms;
                cudaEventDestroy(t0);cudaEventDestroy(t1);
            }
            std::sort(times.begin(),times.end());
            float med=times[NI/2];
            tflops=2.0*M_total*N*K/(med/1000.0)/1e12;
            fprintf(stderr,"%s: %.4f TFLOPS, %.1f us\n",c.name,tflops,med*1000);
        }
        if(!first)printf(", ");first=false;
        if(valid)printf("\"%s\": %.4f",c.name,tflops);
        else{printf("\"%s\": 0.0",c.name);fprintf(stderr,"INVALID: %s\n",c.name);}
        cudaFree(dA);cudaFree(dB);cudaFree(d_sfa_packed);cudaFree(d_sfb_packed);cudaFree(dC);cudaFree(d_mi);
    }
    printf("}\nKERNEL_RESULT_REFERENCE {}\n");
    cudaFree(d_flush);
    return 0;
}
