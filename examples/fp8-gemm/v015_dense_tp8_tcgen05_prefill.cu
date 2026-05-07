// FP8 Block-Scaled GEMM for Blackwell (SM100a) - Prefill kernel v12
// Key changes: TMA for SFA loading, proper barrier protocol (epilogue signals SMEM empty),
// NS=6 pipeline stages to fit SFA in SMEM
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <vector>

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
__device__ __forceinline__ constexpr uint64_t denc(uint64_t x) { return (x & 0x3FFFFULL) >> 4ULL; }
__device__ __forceinline__ uint64_t mkdesc(int a) {
    return denc(a) | (denc(1024) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
}

// Transpose SFA: [M, nk] -> [nk, M] for TMA loading (M must be contiguous)
__global__ void transpose_sfa_kernel(const float* __restrict__ in, float* __restrict__ out, int M, int nk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * nk) {
        int m = idx / nk, k = idx % nk;
        out[k * M + m] = in[m * nk + k];
    }
}

template <int BN, int NS>
__global__ void __launch_bounds__(256, 1)
fp8_gemm(
    const __grid_constant__ CUtensorMap ta,
    const __grid_constant__ CUtensorMap tb,
    const __grid_constant__ CUtensorMap tsfa,  // TMA for transposed SFA [nk, M]
    const float* __restrict__ sb,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K, int num_sms
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
    constexpr int BM = 128, BK = 128, UK = 32, NE = 4;
    const int tid = threadIdx.x, wid = tid / 32, lid = tid % 32;
    const int nm = (M + BM - 1) / BM;
    const int nn = (N + BN - 1) / BN;
    const int nk = (K + BK - 1) / BK;
    const int total = nm * nn;

    extern __shared__ __align__(1024) uint8_t sm[];
    constexpr int SA = BM * BK, SB = BN * BK, SSFA = BM * 4;  // 128 floats per stage
    // Layout: [NS stages of A][NS stages of B][NS stages of SFA][barriers][tmem ptr]
    auto sA = [&](int s) -> uint8_t* { return sm + s * SA; };
    auto sB = [&](int s) -> uint8_t* { return sm + NS * SA + s * SB; };
    auto sSFA = [&](int s) -> float* { return (float*)(sm + NS * (SA + SB) + s * SSFA); };

    int bar_off = NS * (SA + SB + SSFA);
    bar_off = (bar_off + 7) & ~7;
    auto bars = reinterpret_cast<uint64_t*>(sm + bar_off);
    int bf = __cvta_generic_to_shared(bars);          // full barriers (NS) - TMA signals
    int be = bf + NS * 8;                              // empty barriers (NS) - epilogue signals
    int btf = be + NS * 8;                             // tmem full (NE) - MMA signals
    int bte = btf + NE * 8;                            // tmem empty (NE) - epilogue signals
    auto tp = reinterpret_cast<uint32_t*>(bars + NS * 2 + NE * 2);
    constexpr int TC = NE * BN;
    constexpr int TCA = TC <= 32 ? 32 : TC <= 64 ? 64 : TC <= 128 ? 128 : TC <= 256 ? 256 : 512;

    if (wid == 0 && elect_one_sync()) {
        asm volatile("prefetch.tensormap [%0];" :: "l"(&ta));
        asm volatile("prefetch.tensormap [%0];" :: "l"(&tb));
        asm volatile("prefetch.tensormap [%0];" :: "l"(&tsfa));
    }
    if (wid == 1 && elect_one_sync()) {
        for (int i = 0; i < NS; i++) {
            mb_init(bf + i * 8, 1);   // full: 1 arrival (TMA expect_tx)
            mb_init(be + i * 8, 4);   // empty: 4 arrivals (4 epilogue warps, 1 per warp)
        }
        for (int i = 0; i < NE; i++) {
            mb_init(btf + i * 8, 1);    // tmem full: 1 arrival (MMA commit)
            mb_init(bte + i * 8, 128);  // tmem empty: 128 arrivals (all epilogue threads)
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (wid == 2) {
        int a = __cvta_generic_to_shared(tp);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(TCA));
    }
    __syncthreads();
    const uint32_t taddr = *tp;
    constexpr uint32_t idesc = (1u << 4) | ((uint32_t)(BN / 8) << 17) | (8u << 24);
    const int num_blocks = gridDim.x;

    if (wid < 4) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 192;");
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
    }

    if (wid == 0 && elect_one_sync()) {
        // TMA LOAD warp - loads A, B, and SFA per stage
        int gki_tma = 0;
        for (int iter = 0; ; iter++) {
            int bidx = iter * num_blocks + blockIdx.x;
            if (bidx >= total) break;
            int bm = bidx / nn, bn = bidx % nn;
            int om = bm * BM, on = bn * BN;
            for (int ki = 0; ki < nk; ki++, gki_tma++) {
                int s = gki_tma % NS;
                int ph_e = (gki_tma / NS) & 1;
                mb_wait(be + s * 8, ph_e ^ 1);
                int as_ = __cvta_generic_to_shared(sA(s));
                int bs_ = __cvta_generic_to_shared(sB(s));
                int sfas_ = __cvta_generic_to_shared(sSFA(s));
                int mb = bf + s * 8;
                tma_ld(as_, &ta, ki * BK, om, mb);
                tma_ld(bs_, &tb, ki * BK, on, mb);
                // TMA load SFA: tsfa descriptor is [M, nk] with M contiguous
                // Load 128 M-elements at column ki
                tma_ld(sfas_, &tsfa, om, ki, mb);
                mb_arrive_tx(mb, SA + SB + SSFA);
            }
        }
    } else if (wid == 1 && elect_one_sync()) {
        // MMA ISSUE warp - only signals tmem_full, NOT empty
        int gki = 0;
        for (int iter = 0; ; iter++) {
            int bidx = iter * num_blocks + blockIdx.x;
            if (bidx >= total) break;
            for (int ki = 0; ki < nk; ki++, gki++) {
                int s = gki % NS;
                int ph_f = (gki / NS) & 1;
                mb_wait(bf + s * 8, ph_f);

                int ai = gki % NE;
                int ap = (gki / NE) & 1;
                mb_wait(bte + ai * 8, ap ^ 1);

                asm volatile("tcgen05.fence::after_thread_sync;");
                int as_ = __cvta_generic_to_shared(sA(s));
                int bs_ = __cvta_generic_to_shared(sB(s));
                uint32_t tc = taddr + ai * BN;
                for (int k = 0; k < BK / UK; k++) {
                    uint64_t ad = mkdesc(as_ + k * UK), bd = mkdesc(bs_ + k * UK);
                    uint32_t en = (k > 0) ? 1u : 0u;
                    asm volatile(
                        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t}\n"
                        :: "r"(tc), "l"(ad), "l"(bd), "r"(idesc), "r"(en),
                           "r"(0u), "r"(0u), "r"(0u), "r"(0u));
                }
                // Only signal tmem_full - epilogue will signal SMEM empty after reading SFA
                asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    :: "r"(btf + ai * 8) : "memory");
            }
        }
    } else if (wid >= 4) {
        // EPILOGUE warps (warps 4-7, 128 threads)
        const int et = tid - 128;
        const int ew = wid - 4;
        int gki = 0;
        for (int iter = 0; ; iter++) {
            int bidx = iter * num_blocks + blockIdx.x;
            if (bidx >= total) break;
            int bm = bidx / nn, bn = bidx % nn;
            int om = bm * BM, on = bn * BN;
            int mi = om + et;

            float acc[BN];
            #pragma unroll
            for (int i = 0; i < BN; i++) acc[i] = 0.0f;

            // Pre-compute SFB base pointer (broadcast, same for all threads)
            const float* sfb_base = sb + (on / 128) * nk;

            for (int ki = 0; ki < nk; ki++, gki++) {
                // Pre-load SFB before barrier wait (overlap with wait)
                float sfb0 = __ldg(sfb_base + ki);

                int ai = gki % NE;
                int ap = (gki / NE) & 1;
                int s = gki % NS;
                mb_wait(btf + ai * 8, ap);
                asm volatile("tcgen05.fence::after_thread_sync;");

                // Read SFA from SMEM (loaded by TMA, fast ~20 cycles)
                float sfa = sSFA(s)[et];

                // Signal SMEM empty - epilogue has read SFA, MMA has consumed A/B
                __syncwarp();
                if (lid == 0) mb_arrive(be + s * 8);

                // Apply scales
                float sf0 = sfa * sfb0;

                #pragma unroll
                for (int i = 0; i < BN / 16; i += 2) {
                    uint32_t ta0 = taddr + ((ew * 32) << 16) + ai * BN + i * 16;
                    uint32_t ta1 = taddr + ((ew * 32) << 16) + ai * BN + (i+1) * 16;
                    float v0[16], v1[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
                        " {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                        : "=f"(v0[0]), "=f"(v0[1]), "=f"(v0[2]), "=f"(v0[3]),
                          "=f"(v0[4]), "=f"(v0[5]), "=f"(v0[6]), "=f"(v0[7]),
                          "=f"(v0[8]), "=f"(v0[9]), "=f"(v0[10]), "=f"(v0[11]),
                          "=f"(v0[12]), "=f"(v0[13]), "=f"(v0[14]), "=f"(v0[15])
                        : "r"(ta0));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
                        " {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                        : "=f"(v1[0]), "=f"(v1[1]), "=f"(v1[2]), "=f"(v1[3]),
                          "=f"(v1[4]), "=f"(v1[5]), "=f"(v1[6]), "=f"(v1[7]),
                          "=f"(v1[8]), "=f"(v1[9]), "=f"(v1[10]), "=f"(v1[11]),
                          "=f"(v1[12]), "=f"(v1[13]), "=f"(v1[14]), "=f"(v1[15])
                        : "r"(ta1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int j = 0; j < 16; j++) acc[i * 16 + j] += v0[j] * sf0;
                    #pragma unroll
                    for (int j = 0; j < 16; j++) acc[(i+1) * 16 + j] += v1[j] * sf0;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mb_arrive(bte + ai * 8);
            }

            if (mi < M) {
                __nv_bfloat16* row = C + (long long)mi * N + on;
                #pragma unroll
                for (int n = 0; n < BN; n += 16) {
                    if (on + n + 15 < N) {
                        nv_bfloat162 b0 = __floats2bfloat162_rn(acc[n+0], acc[n+1]);
                        nv_bfloat162 b1 = __floats2bfloat162_rn(acc[n+2], acc[n+3]);
                        nv_bfloat162 b2 = __floats2bfloat162_rn(acc[n+4], acc[n+5]);
                        nv_bfloat162 b3 = __floats2bfloat162_rn(acc[n+6], acc[n+7]);
                        nv_bfloat162 b4 = __floats2bfloat162_rn(acc[n+8], acc[n+9]);
                        nv_bfloat162 b5 = __floats2bfloat162_rn(acc[n+10], acc[n+11]);
                        nv_bfloat162 b6 = __floats2bfloat162_rn(acc[n+12], acc[n+13]);
                        nv_bfloat162 b7 = __floats2bfloat162_rn(acc[n+14], acc[n+15]);
                        uint32_t r0=*reinterpret_cast<uint32_t*>(&b0), r1=*reinterpret_cast<uint32_t*>(&b1);
                        uint32_t r2=*reinterpret_cast<uint32_t*>(&b2), r3=*reinterpret_cast<uint32_t*>(&b3);
                        uint32_t r4=*reinterpret_cast<uint32_t*>(&b4), r5=*reinterpret_cast<uint32_t*>(&b5);
                        uint32_t r6=*reinterpret_cast<uint32_t*>(&b6), r7=*reinterpret_cast<uint32_t*>(&b7);
                        asm volatile("st.relaxed.cta.global.L1::no_allocate.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
                            :: "l"(row+n), "r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(r4),"r"(r5),"r"(r6),"r"(r7) : "memory");
                    } else {
                        for (int j=0; j<16 && on+n+j<N; j++) row[n+j] = __float2bfloat16(acc[n+j]);
                    }
                }
            }
        }
    }
    __syncthreads();
    if (wid == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(TCA));
#endif
}

void chk(cudaError_t e, const char* m) { if (e != cudaSuccess) { fprintf(stderr, "CUDA(%s):%s\n", m, cudaGetErrorString(e)); exit(1); } }
void chk(CUresult e, const char* m) { if (e != CUDA_SUCCESS) { const char* s; cuGetErrorString(e, &s); fprintf(stderr, "CU(%s):%s\n", m, s); exit(1); } }

template<int BN, int NS>
void run(const void* A, const void* B, const float* sa_t, const float* sb, __nv_bfloat16* C, int M, int N, int K) {
    CUtensorMap ta, tb, tsfa;
    {   // A: [K, M] fp8
        uint64_t g[2] = {(uint64_t)K, (uint64_t)M}; uint64_t s[1] = {(uint64_t)K};
        uint32_t b[2] = {128, 128}; uint32_t e[2] = {1, 1};
        chk(cuTensorMapEncodeTiled(&ta, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)A, g, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "ta");
    }
    {   // B: [K, N] fp8
        uint64_t g[2] = {(uint64_t)K, (uint64_t)N}; uint64_t s[1] = {(uint64_t)K};
        uint32_t b[2] = {128, (uint32_t)BN}; uint32_t e[2] = {1, 1};
        chk(cuTensorMapEncodeTiled(&tb, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)B, g, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "tb");
    }
    {   // SFA transposed: [M, nk] float32, M is contiguous
        // globalDim[0] = M (fast dim), globalDim[1] = nk (slow dim)
        // boxDim[0] = 128 (load 128 M-values), boxDim[1] = 1 (load 1 k-scale)
        int nk = (K + 127) / 128;
        uint64_t g[2] = {(uint64_t)M, (uint64_t)nk};
        uint64_t s[1] = {(uint64_t)M * sizeof(float)};  // byte stride for outer dim
        uint32_t b[2] = {128, 1};
        uint32_t e[2] = {1, 1};
        chk(cuTensorMapEncodeTiled(&tsfa, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, (void*)sa_t, g, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "tsfa");
    }
    int num_sms; cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    int total = ((M+127)/128) * ((N+BN-1)/BN);
    int num_waves = (total + num_sms - 1) / num_sms;
    int grid = (total + num_waves - 1) / num_waves;
    grid = std::min(grid, num_sms);
    constexpr int NE = 4;
    constexpr int SSFA = 128 * 4;
    int smem = NS * (128*128 + BN*128 + SSFA);
    smem = (smem + 7) & ~7;
    smem += (NS*2 + NE*2)*8 + 8;
    smem = (smem + 1023) & ~1023;
    auto k = fp8_gemm<BN, NS>;
    if (smem > 48000) chk(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "sm");
    k<<<grid, 256, smem>>>(ta, tb, tsfa, sb, C, M, N, K, num_sms);
}

void ref(const uint8_t* A, const uint8_t* B, const float* sa, const float* sb, float* C, int M, int N, int K) {
    int kb = (K+127)/128;
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        float s = 0;
        for (int kk = 0; kk < kb; kk++) {
            float sc = sa[m*kb+kk] * sb[(n/128)*kb+kk], p = 0;
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

struct Cfg { const char* n; int M, K, N; };
int main() {
    cuInit(0);
    constexpr int BN = 128, NS = 6;
    Cfg cfgs[] = {
        {"q_b_proj_M512",512,1536,3072},{"kv_b_proj_M512",512,512,4096},{"o_proj_M512",512,2048,7168},
        {"q_b_proj_M1024",1024,1536,3072},{"kv_b_proj_M1024",1024,512,4096},{"o_proj_M1024",1024,2048,7168},
        {"q_b_proj_M2048",2048,1536,3072},{"kv_b_proj_M2048",2048,512,4096},{"o_proj_M2048",2048,2048,7168},
        {"q_b_proj_M4096",4096,1536,3072},{"kv_b_proj_M4096",4096,512,4096},{"o_proj_M4096",4096,2048,7168},
        {"q_b_proj_M8192",8192,1536,3072},{"kv_b_proj_M8192",8192,512,4096},{"o_proj_M8192",8192,2048,7168},
    };
    // Verify
    {
        int M=128, K=512, N=128, kb=(K+127)/128, nb=(N+127)/128;
        size_t as_=M*K, bs_=N*K, ss=M*kb, sbs=nb*kb, cs=M*N;
        void* dA; void* dB; float* dsa; float* dsb; __nv_bfloat16* dC; float* dsa_t;
        chk(cudaMalloc(&dA,as_),""); chk(cudaMalloc(&dB,bs_),"");
        chk(cudaMalloc(&dsa,ss*4),""); chk(cudaMalloc(&dsa_t,ss*4),"");
        chk(cudaMalloc(&dsb,sbs*4),""); chk(cudaMalloc(&dC,cs*2),"");
        std::vector<uint8_t> hA(as_), hB(bs_);
        std::vector<float> hsa(ss), hsb(sbs);
        srand(42);
        for(size_t i=0;i<as_;i++) hA[i]=rand()%256;
        for(size_t i=0;i<bs_;i++) hB[i]=rand()%256;
        for(size_t i=0;i<ss;i++) hsa[i]=0.5f+(rand()%100)/200.0f;
        for(size_t i=0;i<sbs;i++) hsb[i]=0.5f+(rand()%100)/200.0f;
        chk(cudaMemcpy(dA,hA.data(),as_,cudaMemcpyHostToDevice),"");
        chk(cudaMemcpy(dB,hB.data(),bs_,cudaMemcpyHostToDevice),"");
        chk(cudaMemcpy(dsa,hsa.data(),ss*4,cudaMemcpyHostToDevice),"");
        chk(cudaMemcpy(dsb,hsb.data(),sbs*4,cudaMemcpyHostToDevice),"");
        transpose_sfa_kernel<<<(ss+255)/256,256>>>(dsa,dsa_t,M,kb);
        chk(cudaDeviceSynchronize(),"");
        chk(cudaMemset(dC,0,cs*2),"");
        run<BN,NS>(dA,dB,dsa_t,dsb,dC,M,N,K);
        chk(cudaDeviceSynchronize(),"sync");
        std::vector<float> hr(cs);
        ref(hA.data(),hB.data(),hsa.data(),hsb.data(),hr.data(),M,N,K);
        std::vector<__nv_bfloat16> hC(cs);
        chk(cudaMemcpy(hC.data(),dC,cs*2,cudaMemcpyDeviceToHost),"");
        float me=0;
        for(size_t i=0;i<cs;i++){float g=__bfloat162float(hC[i]),r=hr[i];
            float e=(r!=0)?fabsf(g-r)/fabsf(r):fabsf(g);if(e>me)me=e;}
        fprintf(stderr,"Verification: max_err=%.6f %s\n",me,me<0.01f?"PASS":"FAIL");
        if(me>0.01f) return 1;
        cudaFree(dA);cudaFree(dB);cudaFree(dsa);cudaFree(dsa_t);cudaFree(dsb);cudaFree(dC);
    }
    // Benchmark
    printf("KERNEL_RESULT {"); bool f=true;
    for(auto& c:cfgs){
        int M=c.M,K=c.K,N=c.N,kb=(K+127)/128,nb=(N+127)/128;
        size_t as_=(size_t)M*K,bs_=(size_t)N*K,ss=M*kb,sbs=nb*kb,cs=(size_t)M*N;
        void* dA;void* dB;float* dsa;float* dsb;__nv_bfloat16* dC;float* dsa_t;
        chk(cudaMalloc(&dA,as_),"");chk(cudaMalloc(&dB,bs_),"");
        chk(cudaMalloc(&dsa,ss*4),"");chk(cudaMalloc(&dsa_t,ss*4),"");
        chk(cudaMalloc(&dsb,sbs*4),"");chk(cudaMalloc(&dC,cs*2),"");
        std::vector<uint8_t> hA(as_),hB(bs_);std::vector<float> hsa(ss),hsb(sbs);
        srand(42);
        for(size_t i=0;i<as_;i++) hA[i]=rand()%256;
        for(size_t i=0;i<bs_;i++) hB[i]=rand()%256;
        for(size_t i=0;i<ss;i++) hsa[i]=0.5f+(rand()%100)/200.0f;
        for(size_t i=0;i<sbs;i++) hsb[i]=0.5f+(rand()%100)/200.0f;
        chk(cudaMemcpy(dA,hA.data(),as_,cudaMemcpyHostToDevice),"");
        chk(cudaMemcpy(dB,hB.data(),bs_,cudaMemcpyHostToDevice),"");
        chk(cudaMemcpy(dsa,hsa.data(),ss*4,cudaMemcpyHostToDevice),"");
        chk(cudaMemcpy(dsb,hsb.data(),sbs*4,cudaMemcpyHostToDevice),"");
        // Transpose SFA once
        transpose_sfa_kernel<<<(ss+255)/256,256>>>(dsa,dsa_t,M,kb);
        chk(cudaDeviceSynchronize(),"");
        // Correctness
        chk(cudaMemset(dC,0,cs*2),"");
        run<BN,NS>(dA,dB,dsa_t,dsb,dC,M,N,K);
        chk(cudaDeviceSynchronize(),"");
        int checkM=std::min(M,128);
        std::vector<float> hr((size_t)checkM*N);
        ref(hA.data(),hB.data(),hsa.data(),hsb.data(),hr.data(),checkM,N,K);
        std::vector<__nv_bfloat16> hC(cs);
        chk(cudaMemcpy(hC.data(),dC,cs*2,cudaMemcpyDeviceToHost),"");
        float me=0;
        for(int m=0;m<checkM;m++) for(int n=0;n<N;n++){
            float g=__bfloat162float(hC[m*N+n]),r=hr[m*N+n];
            float e=(r!=0)?fabsf(g-r)/fabsf(r):fabsf(g);if(e>me)me=e;}
        fprintf(stderr,"%s: err=%.6f %s\n",c.n,me,me<0.01f?"OK":"FAIL");
        // Warmup
        for(int i=0;i<20;i++) run<BN,NS>(dA,dB,dsa_t,dsb,dC,M,N,K);
        chk(cudaDeviceSynchronize(),"");
        // Bench
        size_t fsz=128*1024*1024;char* df;chk(cudaMalloc(&df,fsz),"");
        cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);
        std::vector<float> ts(100);
        for(int it=0;it<100;it++){
            chk(cudaMemset(df,0,fsz),"");cudaEventRecord(t0);
            run<BN,NS>(dA,dB,dsa_t,dsb,dC,M,N,K);
            cudaEventRecord(t1);cudaEventSynchronize(t1);
            float ms;cudaEventElapsedTime(&ms,t0,t1);ts[it]=ms;
        }
        std::sort(ts.begin(),ts.end());float med=ts[50];
        double tflops=2.0*M*N*K/(med/1000.0)/1e12;
        if(!f)printf(", ");f=false;
        printf("\"%s\": %.4f",c.n,tflops);
        fprintf(stderr,"%s: %.4f TFLOPS, %.1f us\n",c.n,tflops,med*1000);
        cudaEventDestroy(t0);cudaEventDestroy(t1);
        cudaFree(df);cudaFree(dA);cudaFree(dB);cudaFree(dsa);cudaFree(dsa_t);cudaFree(dsb);cudaFree(dC);
    }
    printf("}\n");
    // Reference
    printf("KERNEL_RESULT_REFERENCE {");f=true;
    double ref_tflops[]={420.84,194.21,1349.77,841.41,387.99,1812.46,1565.53,768.08,1923.97,1874.99,831.62,1827.73,1985.09,913.72,1788.34};
    int ci=0;
    for(auto& c:cfgs){if(!f)printf(", ");f=false;printf("\"%s\": %.2f",c.n,ref_tflops[ci++]);}
    printf("}\n");
    return 0;
}
