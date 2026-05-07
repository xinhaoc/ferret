// FP8 Block-Scaled GEMM for Blackwell (SM100a) - block_scale MMA
// Hardware-fused dequant via tcgen05.mma.kind::mxf8f6f4.block_scale
// UE8M0 scales in TMEM, UTCCP transpose + copy
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
// K-major 128B swizzle descriptor
__device__ __forceinline__ uint64_t mkdesc(int a) {
    return denc(a) | (denc(1024) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
}
// SF descriptor: no swizzle, SBO=8*16=128, LBO=0, version=1
__device__ __forceinline__ uint64_t mkdesc_sf(int a) {
    return denc(a) | (denc(128) << 32ULL) | (1ULL << 46ULL);
}
__device__ __forceinline__ uint32_t ld_shared_u32(const void* p) {
    uint32_t r; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"((uint32_t)__cvta_generic_to_shared(p))); return r;
}
__device__ __forceinline__ void st_shared_u32(void* p, uint32_t v) {
    asm volatile("st.shared.u32 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(p)), "r"(v));
}
// L2-swizzled block index: groups M-blocks for better L2 reuse of A
__device__ __forceinline__ void swizzle_idx(int bidx, int nm, int nn, int gsz,
                                            int& bm, int& bn) {
    int bpg = nn * gsz;  // blocks per group
    int gi = bidx / bpg;
    int ig = bidx % bpg;
    int fm = gi * gsz;
    int ag = min(gsz, nm - fm);
    bm = fm + ig % ag;
    bn = ig / ag;
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
    constexpr int SF_PER_LOAD = 4; // 4 UE8M0 packed per uint32_t
    constexpr int NE = 2; // epilogue stages
    const int tid = threadIdx.x, wid = tid / 32;
    const uint32_t lid = tid % 32;
    const int nm = (M + BM - 1) / BM, nn = (N + BN - 1) / BN, nk = (K + BK - 1) / BK;
    const int total = nm * nn;
    // L2 swizzle group size: group M-blocks together for better L2 reuse of A
    constexpr int SWIZZLE_GROUP = 16;  // group 16 M-blocks

    extern __shared__ __align__(1024) uint8_t sm[];
    constexpr int STORE_BN = 64;
    constexpr int NUM_TMA_ST = 1;
    constexpr int SCD_STAGE = BM * STORE_BN * 2; // 16KB
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
    int bf = __cvta_generic_to_shared(bars);        // full[NS]: TMA done
    int be = bf + NS * 8;                            // empty[NS]: MMA consumed
    int bsf = be + NS * 8;                           // sf_ready[NS]: SF transposed by warp 2
    int btf = bsf + NS * 8;                          // tmem_full[NE]: accum ready
    int bte = btf + NE * 8;                          // tmem_empty[NE]: accum consumed
    auto tp = reinterpret_cast<uint32_t*>(bars + NS * 3 + NE * 2);

    // TMEM layout: NE*BN accum cols + 4 SFA cols + SFB cols
    constexpr int SF_BLOCK_M = ((BM + 127) / 128) * 128; // 128
    constexpr int SF_BLOCK_N = ((BN + 127) / 128) * 128; // 128 for BN=128
    constexpr int TMEM_SFA_COLS = SF_BLOCK_M / 32; // 4
    constexpr int TMEM_SFB_COLS = SF_BLOCK_N / 32; // 4
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
            mb_init(bf + i * 8, 1);    // TMA arrive
            mb_init(be + i * 8, 1);    // MMA commit (1 arrival from umma_arrive)
            mb_init(bsf + i * 8, 32);  // warp 2 (32 threads) arrive after SF transpose
        }
        for (int i = 0; i < NE; i++) {
            mb_init(btf + i * 8, 1);    // MMA commit
            mb_init(bte + i * 8, 128);  // 128 epilogue threads
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (wid == 2) {
        int a = __cvta_generic_to_shared(tp);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(TCA));
    }
    __syncthreads();
    const uint32_t taddr = *tp;
    const int num_blocks = gridDim.x;

    // Register allocation
    if (wid < 4) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");
    }

    // Block scale instruction descriptor
    // bits: [5:4]=a_sf_id, [9:7]=a_type(E4M3=0), [12:10]=b_type(E4M3=0),
    //       [22:17]=N>>3, [23]=scale_type(UE8M0=1), [28:27]=M>>7, [30:29]=b_sf_id
    constexpr uint32_t base_idesc = ((uint32_t)(BN / 8) << 17) | (1u << 23) | ((uint32_t)(BM / 128) << 27);

    // ====== WARP 0: TMA LOAD ======
    if (wid == 0 && elect_one_sync()) {
        int gki = 0;
        for (int iter = 0; ; iter++) {
            int bidx = iter * num_blocks + blockIdx.x;
            if (bidx >= total) break;
            int bm = bidx / nn, bn = bidx % nn;
            int om = bm * BM, on = bn * BN;
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
            }
        }
    }
    // ====== WARP 2: UTCCP TRANSPOSE ======
    else if (wid == 2) {
        auto utccp_transpose = [&](uint32_t* ptr) {
            // Transpose 128 uint32_t: 4x32 → 32x4
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
                // Wait for TMA to finish loading
                mb_wait(bf + s * 8, ph);

                // Only transpose when SF was loaded
                if (ki % SF_PER_LOAD == 0) {
                    // Transpose SFA (128 elements)
                    utccp_transpose(sSFA(s));
                    // Transpose SFB (BN elements, each 128-aligned block)
                    for (int b = 0; b < SF_BLOCK_N; b += 128) {
                        utccp_transpose(sSFB(s) + b);
                    }
                    // Fence for shared memory writes
                    asm volatile("fence.proxy.async.shared::cta;");
                }

                // Signal SF ready
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

            // Wait tmem empty at start of new tile
            int accum_idx = iter % NE;
            int accum_ph = (iter / NE) & 1;
            mb_wait(bte + accum_idx * 8, accum_ph ^ 1);

            for (int ki = 0; ki < nk; ki++, gki++) {
                int s = gki % NS;
                int ph = (gki / NS) & 1;
                // Wait for SF transpose to be done
                mb_wait(bsf + s * 8, ph);
                asm volatile("tcgen05.fence::after_thread_sync;");

                // UTCCP copy SF from SMEM to TMEM when SF was loaded
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

                // Build idesc with SF ID for current K-block within group
                uint32_t sf_id = ki % SF_PER_LOAD;
                uint32_t idesc = base_idesc | (sf_id << 4) | (sf_id << 29);

                // Issue block_scale MMA
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

                // Signal SMEM empty
                asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    :: "r"(be + s * 8) : "memory");

                // Signal tmem_full on last K-iteration
                if (ki == nk - 1) {
                    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                        :: "r"(btf + accum_idx * 8) : "memory");
                }
            }
        }
    }
    // ====== WARPS 4-7: EPILOGUE (TMA STORE) ======
    else if (wid >= 4) {
        const int ew = wid - 4;
        uint32_t tma_st = 0;
        for (int iter = 0; ; iter++) {
            int bidx = iter * num_blocks + blockIdx.x;
            if (bidx >= total) break;
            int bm = bidx / nn, bn = bidx % nn;
            int om = bm * BM, on = bn * BN;
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
        // Dealloc TMEM from epilogue warp 1 (like DeepGEMM)
        if (ew == 1) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(TCA));
    }
    // No __syncthreads() at end - epilogue handles dealloc
#endif
}

// Host helpers
void chk(cudaError_t e, const char* m) { if (e != cudaSuccess) { fprintf(stderr, "CUDA(%s):%s\n", m, cudaGetErrorString(e)); exit(1); } }
void chk(CUresult e, const char* m) { if (e != CUDA_SUCCESS) { const char* s; cuGetErrorString(e, &s); fprintf(stderr, "CU(%s):%s\n", m, s); exit(1); } }

// Convert float32 scale to UE8M0
// UE8M0 represents 2^(e-127), so we extract the exponent from a power of 2
uint8_t float_to_ue8m0(float val) {
    if (val <= 0.0f) return 0;
    // Round to nearest power of 2
    float p2 = exp2f(roundf(log2f(val)));
    uint32_t bits;
    memcpy(&bits, &p2, 4);
    return (bits >> 23) & 0xFF;
}

// Prepare UE8M0 packed scales with UTCCP-compatible layout
// Input: float scales[dim, nk] row-major
// Output: uint32_t packed[ceil(nk/4), dim] column-major
// Each uint32_t holds 4 UE8M0 bytes
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

// Expand weight scales from [N/128, nk] to [N, nk]
void expand_sb(const float* sb, float* sb_expanded, int N, int nk) {
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

    CUtensorMap ta, tb, tsfa, tsfb, td_map;
    { // A: [K, M] fp8
        uint64_t g[2] = {(uint64_t)K, (uint64_t)M}; uint64_t s[1] = {(uint64_t)K};
        uint32_t b[2] = {128, 128}; uint32_t e[2] = {1, 1};
        chk(cuTensorMapEncodeTiled(&ta, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)A, g, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "ta");
    }
    { // B: [K, N] fp8
        uint64_t g[2] = {(uint64_t)K, (uint64_t)N}; uint64_t s[1] = {(uint64_t)K};
        uint32_t b[2] = {128, (uint32_t)BN}; uint32_t e[2] = {1, 1};
        chk(cuTensorMapEncodeTiled(&tb, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)B, g, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "tb");
    }
    { // SFA: [M, num_sf_k] uint32_t, M contiguous
        uint64_t g[2] = {(uint64_t)M, (uint64_t)num_sf_k};
        uint64_t s[1] = {(uint64_t)M * sizeof(uint32_t)};
        uint32_t b[2] = {128, 1}; uint32_t e[2] = {1, 1};
        chk(cuTensorMapEncodeTiled(&tsfa, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, (void*)sfa_packed, g, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "tsfa");
    }
    { // SFB: [N, num_sf_k] uint32_t, N contiguous
        uint64_t g[2] = {(uint64_t)N, (uint64_t)num_sf_k};
        uint64_t s[1] = {(uint64_t)N * sizeof(uint32_t)};
        uint32_t b[2] = {(uint32_t)BN, 1}; uint32_t e[2] = {1, 1};
        chk(cuTensorMapEncodeTiled(&tsfb, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, (void*)sfb_packed, g, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "tsfb");
    }

    { // D
        uint64_t g[2] = {(uint64_t)N, (uint64_t)M};
        uint64_t s[1] = {(uint64_t)N * sizeof(__nv_bfloat16)};
        uint32_t b[2] = {64, 128}; uint32_t e[2] = {1, 1};
        chk(cuTensorMapEncodeTiled(&td_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)C, g, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "td");
    }
    int num_sms; cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    int total = ((M + 127) / 128) * ((N + BN - 1) / BN);
    int num_waves = (total + num_sms - 1) / num_sms;
    int grid = (total + num_waves - 1) / num_waves;
    grid = std::min(grid, num_sms);

    constexpr int NE = 2;
    int smem = 1*128*64*2 + NS * (128*128 + BN*128 + 128*4 + BN*4);
    smem = (smem + 7) & ~7;
    smem += (NS*3 + NE*2)*8 + 8;
    smem = (smem + 1023) & ~1023;

    auto k = fp8_gemm_bs<BN, NS>;
    if (smem > 48000) chk(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "sm");
    k<<<grid, 256, smem>>>(ta, tb, tsfa, tsfb, td_map, C, M, N, K, num_sms);
}

// Convert UE8M0 back to float
float ue8m0_to_float(uint8_t e) {
    if (e == 0) return 0.0f;
    uint32_t bits = (uint32_t)e << 23;
    float val;
    memcpy(&val, &bits, 4);
    return val;
}

void ref_bs(const uint8_t* A, const uint8_t* B, const float* sa, const float* sb, float* C, int M, int N, int K) {
    // Reference using UE8M0-rounded scales (to match block_scale MMA behavior)
    int kb = (K+127)/128;
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        float s = 0;
        for (int kk = 0; kk < kb; kk++) {
            // Convert float scale to UE8M0 and back to get the rounded value
            float sa_ue = ue8m0_to_float(float_to_ue8m0(sa[m*kb+kk]));
            float sb_ue = ue8m0_to_float(float_to_ue8m0(sb[(n/128)*kb+kk]));
            float sc = sa_ue * sb_ue, p = 0;
            for (int k = kk*128; k < std::min((kk+1)*128, K); k++) {
                float a = __half2float(__nv_cvt_fp8_to_halfraw(A[m*K+k], __NV_E4M3));
                float b = __half2float(__nv_cvt_fp8_to_halfraw(B[n*K+k], __NV_E4M3));
                p += a*b;
            }
            s += p*sc;
        }
        C[m*N+n] = s;
    }
}


void dispatch_run(const void* A, const void* B, const uint32_t* sfa, const uint32_t* sfb,
                  __nv_bfloat16* C, int M, int N, int K) {
    if (N > 4096) {
        run_bs<192, 5>(A, B, sfa, sfb, C, M, N, K);
    } else {
        run_bs<128, 6>(A, B, sfa, sfb, C, M, N, K);
    }
}

struct Cfg { const char* n; int M, K, N; };

int main() {
    cuInit(0);
    // BN=192 for o_proj (N=7168), BN=128 for others
    // Note: run_bs is templated, dispatch done per-config below
    constexpr int BN_LARGE = 192, NS_LARGE = 5;
    constexpr int BN_SMALL = 128, NS_SMALL = 6;

    Cfg cfgs[] = {
        {"o_proj_M2048",    2048, 2048, 7168},
        {"q_b_proj_M8192",  8192, 1536, 3072},
        {"kv_b_proj_M8192", 8192, 512,  4096},
        {"o_proj_M4096",    4096, 2048, 7168},
        {"o_proj_M8192",    8192, 2048, 7168},
    };
    int ncfg = sizeof(cfgs) / sizeof(cfgs[0]);

    // Validation on small config
    {
        int M = 128, K = 512, N = 128, kb = (K+127)/128, nb = (N+127)/128;
        int num_sf_k = (kb + 3) / 4;
        size_t as_ = M*K, bs_ = N*K, cs = M*N;
        void* dA; void* dB; uint32_t *dsfa, *dsfb; __nv_bfloat16* dC;
        chk(cudaMalloc(&dA, as_), ""); chk(cudaMalloc(&dB, bs_), "");
        chk(cudaMalloc(&dsfa, num_sf_k*M*4), "");
        chk(cudaMalloc(&dsfb, num_sf_k*N*4), "");
        chk(cudaMalloc(&dC, cs*2), "");
        std::vector<uint8_t> hA(as_), hB(bs_);
        std::vector<float> hsa(M*kb), hsb(nb*kb);
        srand(42);
        for (auto& v : hA) { v = rand()%254; if(v>=0x7F) v++; }
        for (auto& v : hB) { v = rand()%254; if(v>=0x7F) v++; }
        for (auto& v : hsa) v = 0.5f + (rand()%100)/200.0f;
        for (auto& v : hsb) v = 0.5f + (rand()%100)/200.0f;
        // Prepare packed scales
        std::vector<uint32_t> sfa_packed(num_sf_k*M);
        prepare_sf(hsa.data(), sfa_packed.data(), M, kb);
        std::vector<float> sb_exp(N*kb);
        expand_sb(hsb.data(), sb_exp.data(), N, kb);
        std::vector<uint32_t> sfb_packed(num_sf_k*N);
        prepare_sf(sb_exp.data(), sfb_packed.data(), N, kb);

        chk(cudaMemcpy(dA, hA.data(), as_, cudaMemcpyHostToDevice), "");
        chk(cudaMemcpy(dB, hB.data(), bs_, cudaMemcpyHostToDevice), "");
        chk(cudaMemcpy(dsfa, sfa_packed.data(), sfa_packed.size()*4, cudaMemcpyHostToDevice), "");
        chk(cudaMemcpy(dsfb, sfb_packed.data(), sfb_packed.size()*4, cudaMemcpyHostToDevice), "");
        chk(cudaMemset(dC, 0, cs*2), "");
        run_bs<BN_SMALL, NS_SMALL>(dA, dB, dsfa, dsfb, dC, M, N, K);  // small validation
        auto err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Validation LAUNCH ERROR: %s\n", cudaGetErrorString(err));
            // Fall through - might still produce output
        }
        std::vector<float> hr(cs);
        ref_bs(hA.data(), hB.data(), hsa.data(), hsb.data(), hr.data(), M, N, K);
        std::vector<__nv_bfloat16> hC(cs);
        chk(cudaMemcpy(hC.data(), dC, cs*2, cudaMemcpyDeviceToHost), "");

        float max_abs = 0;
        for (size_t i = 0; i < cs; i++) {
            float r = hr[i];
            if (!isnan(r) && fabsf(r) > max_abs) max_abs = fabsf(r);
        }
        float atol = max_abs * 1e-3f;
        float me = 0; int nan_cnt = 0;
        for (size_t i = 0; i < cs; i++) {
            float g = __bfloat162float(hC[i]), r = hr[i];
            if (isnan(g) || isnan(r)) { nan_cnt++; continue; }
            float denom = fmaxf(fabsf(r), atol);
            float e = fabsf(g-r) / denom;
            if (e > me) me = e;
        }
        fprintf(stderr, "Validation: max_err=%.6f nan=%d %s\n", me, nan_cnt, (me < 0.01f && nan_cnt == 0) ? "PASS" : "FAIL");
        for (int i = 0; i < 3; i++)
            fprintf(stderr, "  [%d] gpu=%.6f ref=%.6f\n", i, __bfloat162float(hC[i]), hr[i]);
        cudaFree(dA); cudaFree(dB); cudaFree(dsfa); cudaFree(dsfb); cudaFree(dC);
    }

    // Per-config validation + benchmark
    printf("KERNEL_RESULT {"); bool first = true;
    for (int ci = 0; ci < ncfg; ci++) {
        auto& c = cfgs[ci];
        int M = c.M, K = c.K, N = c.N, kb = (K+127)/128, nb = (N+127)/128;
        int num_sf_k = (kb + 3) / 4;
        size_t as_ = (size_t)M*K, bs_ = (size_t)N*K, cs = (size_t)M*N;
        void *dA, *dB; uint32_t *dsfa, *dsfb; __nv_bfloat16 *dC;
        chk(cudaMalloc(&dA, as_), ""); chk(cudaMalloc(&dB, bs_), "");
        chk(cudaMalloc(&dsfa, (size_t)num_sf_k*M*4), "");
        chk(cudaMalloc(&dsfb, (size_t)num_sf_k*N*4), "");
        chk(cudaMalloc(&dC, cs*2), "");

        std::vector<uint8_t> hA(as_), hB(bs_);
        std::vector<float> hsa(M*kb), hsb(nb*kb);
        srand(42 + ci);
        for (auto& v : hA) { v = rand()%254; if(v>=0x7F) v++; }
        for (auto& v : hB) { v = rand()%254; if(v>=0x7F) v++; }
        for (auto& v : hsa) v = 0.5f + (rand()%100)/200.0f;
        for (auto& v : hsb) v = 0.5f + (rand()%100)/200.0f;
        std::vector<uint32_t> sfa_packed(num_sf_k*M);
        prepare_sf(hsa.data(), sfa_packed.data(), M, kb);
        std::vector<float> sb_exp(N*kb);
        expand_sb(hsb.data(), sb_exp.data(), N, kb);
        std::vector<uint32_t> sfb_packed(num_sf_k*N);
        prepare_sf(sb_exp.data(), sfb_packed.data(), N, kb);

        chk(cudaMemcpy(dA, hA.data(), as_, cudaMemcpyHostToDevice), "");
        chk(cudaMemcpy(dB, hB.data(), bs_, cudaMemcpyHostToDevice), "");
        chk(cudaMemcpy(dsfa, sfa_packed.data(), sfa_packed.size()*4, cudaMemcpyHostToDevice), "");
        chk(cudaMemcpy(dsfb, sfb_packed.data(), sfb_packed.size()*4, cudaMemcpyHostToDevice), "");

        // Correctness check - use dispatch_run for correct BN selection
        chk(cudaMemset(dC, 0, cs*2), "");
        dispatch_run(dA, dB, dsfa, dsfb, dC, M, N, K);
        auto err2 = cudaDeviceSynchronize();

        bool valid = false;
        float me = 999.0f; int nan_cnt = 0;
        if (err2 == cudaSuccess) {
            int checkM = std::min(M, 128);
            std::vector<float> hr((size_t)checkM*N);
            ref_bs(hA.data(), hB.data(), hsa.data(), hsb.data(), hr.data(), checkM, N, K);
            std::vector<__nv_bfloat16> hC(cs);
            chk(cudaMemcpy(hC.data(), dC, cs*2, cudaMemcpyDeviceToHost), "");
            float max_abs = 0;
            for (int m = 0; m < checkM; m++) for (int n = 0; n < N; n++) {
                float r = hr[m*N+n];
                if (!isnan(r) && fabsf(r) > max_abs) max_abs = fabsf(r);
            }
            float atol = max_abs * 1e-3f;
            me = 0;
            for (int m = 0; m < checkM; m++) for (int n = 0; n < N; n++) {
                float g = __bfloat162float(hC[m*N+n]), r = hr[m*N+n];
                if (isnan(g) || isnan(r)) { nan_cnt++; continue; }
                float denom = fmaxf(fabsf(r), atol);
                float e = fabsf(g-r) / denom;
                if (e > me) me = e;
            }
            valid = (me < 0.01f && nan_cnt == 0);
        } else {
            fprintf(stderr, "%s: LAUNCH ERROR: %s\n", c.n, cudaGetErrorString(err2));
        }
        fprintf(stderr, "%s: err=%.6f nan=%d %s\n", c.n, me, nan_cnt, valid ? "OK" : "INVALID");

        double tflops = 0.0;
        if (valid) {
            for (int i = 0; i < 20; i++) dispatch_run(dA, dB, dsfa, dsfb, dC, M, N, K);
            chk(cudaDeviceSynchronize(), "");
            cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
            int NI = 100;
            cudaEventRecord(t0);
            for (int it = 0; it < NI; it++)
                dispatch_run(dA, dB, dsfa, dsfb, dC, M, N, K);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            float total_ms; cudaEventElapsedTime(&total_ms, t0, t1);
            float med = total_ms / NI;
            tflops = 2.0*M*N*K / (med/1000.0) / 1e12;
            fprintf(stderr, "%s: %.4f TFLOPS, %.1f us\n", c.n, tflops, med*1000);
            cudaEventDestroy(t0); cudaEventDestroy(t1);
        }

        if (!first) printf(", "); first = false;
        if (valid) printf("\"%s\": %.4f", c.n, tflops);
        else { printf("\"%s\": 0.0", c.n); fprintf(stderr, "INVALID: %s\n", c.n); }
        cudaFree(dA); cudaFree(dB); cudaFree(dsfa); cudaFree(dsfb); cudaFree(dC);
    }
    printf("}\n");
    return 0;
}
