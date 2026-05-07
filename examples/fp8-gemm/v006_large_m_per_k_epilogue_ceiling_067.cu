// FP8 Block-Scaled GEMM for Blackwell (SM100a) - Large M prefill v3
// TMA-based SFA loading to fix uncoalesced global accesses
// cta_group::1, persistent kernel, warp-specialized
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <string.h>

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

__device__ __forceinline__ constexpr uint64_t denc(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}
__device__ __forceinline__ uint64_t mkdesc(int a) {
    return denc(a) | (denc(1024) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
}

// Transpose SFA: [M, nk] -> [nk, M] for TMA loading
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
    const __grid_constant__ CUtensorMap tsfa,
    const float* __restrict__ sb,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K, int num_sms
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
    constexpr int BM=128, BK=128, UK=32, NE=4;
    const int tid=threadIdx.x, wid=tid/32, lid=tid%32;
    const int nn=(N+BN-1)/BN, nk=(K+BK-1)/BK;
    const int total=((M+BM-1)/BM)*nn;

    extern __shared__ __align__(1024) uint8_t sm[];
    constexpr int SA=BM*BK, SB=BN*BK, SSFA=BM*4;
    auto sA=[&](int s)->uint8_t*{return sm+s*SA;};
    auto sB=[&](int s)->uint8_t*{return sm+NS*SA+s*SB;};
    auto sSFA=[&](int s)->float*{return (float*)(sm+NS*(SA+SB)+s*SSFA);};

    int bar_off=NS*(SA+SB+SSFA);
    bar_off=(bar_off+7)&~7;
    auto bars=reinterpret_cast<uint64_t*>(sm+bar_off);
    int bf=__cvta_generic_to_shared(bars);
    int be=bf+NS*8, btf=be+NS*8, bte=btf+NE*8;
    auto tp=reinterpret_cast<uint32_t*>(bars+NS*2+NE*2);
    constexpr int TC=NE*BN;
    constexpr int TCA=TC<=32?32:TC<=64?64:TC<=128?128:TC<=256?256:512;

    if(wid==0&&elect_one_sync()){
        asm volatile("prefetch.tensormap [%0];"::  "l"(&ta));
        asm volatile("prefetch.tensormap [%0];"::  "l"(&tb));
        asm volatile("prefetch.tensormap [%0];"::  "l"(&tsfa));
    }
    if(wid==1&&elect_one_sync()){
        for(int i=0;i<NS;i++){mb_init(bf+i*8,1);mb_init(be+i*8,4);}  // be: 4 arrivals from 4 epilogue warps
        for(int i=0;i<NE;i++){mb_init(btf+i*8,1);mb_init(bte+i*8,128);}
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if(wid==2){
        int a=__cvta_generic_to_shared(tp);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(a),"r"(TCA));
    }
    __syncthreads();
    const uint32_t taddr=*tp;
    constexpr uint32_t idesc=(1u<<4)|((uint32_t)(BN/8)<<17)|(8u<<24);

    // Persistent scheduling
    if(wid==0&&elect_one_sync()){
        // TMA LOAD warp - loads A, B, and SFA
        int ph=0;
        for(int iter=0;;iter++){
            int bidx=iter*num_sms+blockIdx.x;
            if(bidx>=total) break;
            int bm=bidx/nn, bn=bidx%nn;
            int om=bm*BM, on=bn*BN;
            for(int ki=0;ki<nk;ki++){
                int s=ki%NS;
                mb_wait(be+s*8, ph^1);
                if(s==NS-1) ph^=1;
                int as_=__cvta_generic_to_shared(sA(s));
                int bs_=__cvta_generic_to_shared(sB(s));
                int sfas_=__cvta_generic_to_shared(sSFA(s));
                int mb=bf+s*8;
                tma_ld(as_,&ta,ki*BK,om,mb);
                tma_ld(bs_,&tb,ki*BK,on,mb);
                tma_ld(sfas_,&tsfa,om,ki,mb);
                mb_arrive_tx(mb,SA+SB+SSFA);
            }
        }
    } else if(wid==1&&elect_one_sync()){
        // MMA ISSUE warp
        int ph=0;
        int gki=0;
        for(int iter=0;;iter++){
            int bidx=iter*num_sms+blockIdx.x;
            if(bidx>=total) break;
            for(int ki=0;ki<nk;ki++,gki++){
                int ai=gki%NE;
                int ap=(gki/NE)&1;
                mb_wait(bte+ai*8, ap^1);

                int s=ki%NS;
                mb_wait(bf+s*8, ph);
                if(s==NS-1) ph^=1;

                asm volatile("tcgen05.fence::after_thread_sync;");
                int as_=__cvta_generic_to_shared(sA(s));
                int bs_=__cvta_generic_to_shared(sB(s));
                uint32_t tc=taddr+ai*BN;
                for(int k=0;k<BK/UK;k++){
                    uint64_t ad=mkdesc(as_+k*UK), bd=mkdesc(bs_+k*UK);
                    uint32_t en=(k>0)?1u:0u;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.ne.b32 p, %4, 0;\n\t"
                        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"
                        "}\n"
                        ::"r"(tc),"l"(ad),"l"(bd),"r"(idesc),"r"(en),"r"(0u),"r"(0u),"r"(0u),"r"(0u));
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                // Only signal TMEM full - epilogue signals SMEM empty
                asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    ::"r"(btf+ai*8):"memory");
            }
        }
    } else if(wid>=4){
        // EPILOGUE warps (4-7)
        const int et=tid-128, ew=wid-4;
        int gki=0;
        for(int iter=0;;iter++){
            int bidx=iter*num_sms+blockIdx.x;
            if(bidx>=total) break;
            int bm=bidx/nn, bn=bidx%nn;
            int om=bm*BM, on=bn*BN;
            int mi=om+et;

            float acc[BN];
            #pragma unroll
            for(int i=0;i<BN;i++) acc[i]=0.0f;

            for(int ki=0;ki<nk;ki++,gki++){
                // Pre-load SFB from global (broadcast, efficient)
                float sfb0=__ldg(sb+(on/128)*nk+ki);
                float sfb1=0.0f;
                if(BN>128) sfb1=__ldg(sb+((on+128)/128)*nk+ki);

                int ai=gki%NE;
                int ap=(gki/NE)&1;
                int s=ki%NS;
                mb_wait(btf+ai*8, ap);
                asm volatile("tcgen05.fence::after_thread_sync;");

                // Read SFA from SMEM (loaded by TMA, coalesced)
                float sfa=sSFA(s)[et];

                // Signal SMEM empty - epilogue has read SFA
                if(lid==0) mb_arrive(be+s*8);

                float sf0=sfa*sfb0;
                uint32_t tbase=taddr+((ew*32)<<16)+ai*BN;
                // Issue all TMEM loads first, then wait and process
                {
                    float v[BN];
                    #pragma unroll
                    for(int i=0;i<BN/16;i++){
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                            :"=f"(v[i*16+0]),"=f"(v[i*16+1]),"=f"(v[i*16+2]),"=f"(v[i*16+3]),
                             "=f"(v[i*16+4]),"=f"(v[i*16+5]),"=f"(v[i*16+6]),"=f"(v[i*16+7]),
                             "=f"(v[i*16+8]),"=f"(v[i*16+9]),"=f"(v[i*16+10]),"=f"(v[i*16+11]),
                             "=f"(v[i*16+12]),"=f"(v[i*16+13]),"=f"(v[i*16+14]),"=f"(v[i*16+15])
                            :"r"(tbase+i*16));
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for(int i=0;i<BN;i++) acc[i]+=v[i]*sf0;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mb_arrive(bte+ai*8);
            }

            // Write output
            if(mi<M){
                __nv_bfloat16* row=C+(long long)mi*N+on;
                #pragma unroll
                for(int n=0;n<BN;n+=16){
                    if(on+n+15<N){
                        nv_bfloat162 b0=__floats2bfloat162_rn(acc[n+0],acc[n+1]);
                        nv_bfloat162 b1=__floats2bfloat162_rn(acc[n+2],acc[n+3]);
                        nv_bfloat162 b2=__floats2bfloat162_rn(acc[n+4],acc[n+5]);
                        nv_bfloat162 b3=__floats2bfloat162_rn(acc[n+6],acc[n+7]);
                        nv_bfloat162 b4=__floats2bfloat162_rn(acc[n+8],acc[n+9]);
                        nv_bfloat162 b5=__floats2bfloat162_rn(acc[n+10],acc[n+11]);
                        nv_bfloat162 b6=__floats2bfloat162_rn(acc[n+12],acc[n+13]);
                        nv_bfloat162 b7=__floats2bfloat162_rn(acc[n+14],acc[n+15]);
                        uint32_t r0=*reinterpret_cast<uint32_t*>(&b0);
                        uint32_t r1=*reinterpret_cast<uint32_t*>(&b1);
                        uint32_t r2=*reinterpret_cast<uint32_t*>(&b2);
                        uint32_t r3=*reinterpret_cast<uint32_t*>(&b3);
                        uint32_t r4=*reinterpret_cast<uint32_t*>(&b4);
                        uint32_t r5=*reinterpret_cast<uint32_t*>(&b5);
                        uint32_t r6=*reinterpret_cast<uint32_t*>(&b6);
                        uint32_t r7=*reinterpret_cast<uint32_t*>(&b7);
                        asm volatile(
                            "st.relaxed.cta.global.L1::no_allocate.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
                            :: "l"(row+n), "r"(r0),"r"(r1),"r"(r2),"r"(r3),
                               "r"(r4),"r"(r5),"r"(r6),"r"(r7) : "memory");
                    } else {
                        for(int j=0;j<16&&on+n+j<N;j++) row[n+j]=__float2bfloat16(acc[n+j]);
                    }
                }
            }
        }
    }

    __syncthreads();
    if(wid==0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(taddr),"r"(TCA));
#endif
}

void chk(cudaError_t e,const char*m){if(e!=cudaSuccess){fprintf(stderr,"CUDA(%s):%s\n",m,cudaGetErrorString(e));exit(1);}}
void chk(CUresult e,const char*m){if(e!=CUDA_SUCCESS){const char*s;cuGetErrorString(e,&s);fprintf(stderr,"CU(%s):%s\n",m,s);exit(1);}}

template<int BN,int NS>
void run(const void*A,const void*B,const float*sa_t,const float*sb,__nv_bfloat16*C,int M,int N,int K){
    CUtensorMap ta,tb,tsfa;
    {uint64_t g[2]={(uint64_t)K,(uint64_t)M};uint64_t s[1]={(uint64_t)K};uint32_t b[2]={128,128};uint32_t e[2]={1,1};
     chk(cuTensorMapEncodeTiled(&ta,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,(void*)A,g,s,b,e,
         CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),"ta");}
    {uint64_t g[2]={(uint64_t)K,(uint64_t)N};uint64_t s[1]={(uint64_t)K};uint32_t b[2]={128,(uint32_t)BN};uint32_t e[2]={1,1};
     chk(cuTensorMapEncodeTiled(&tb,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,(void*)B,g,s,b,e,
         CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),"tb");}
    {int nk=(K+127)/128;
     uint64_t g[2]={(uint64_t)M,(uint64_t)nk};
     uint64_t s[1]={(uint64_t)M*sizeof(float)};
     uint32_t b[2]={128,1};uint32_t e[2]={1,1};
     chk(cuTensorMapEncodeTiled(&tsfa,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,(void*)sa_t,g,s,b,e,
         CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
         CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),"tsfa");}
    int num_sms; cudaDeviceGetAttribute(&num_sms,cudaDevAttrMultiProcessorCount,0);
    int total=((M+127)/128)*((N+BN-1)/BN);
    int grid = std::min(total, num_sms);
    constexpr int NE=4, SSFA=128*4;
    int smem=NS*(128*128+BN*128+SSFA);
    smem=(smem+7)&~7;
    smem+=(NS*2+NE*2)*8+8;
    smem=(smem+1023)&~1023;
    auto k=fp8_gemm<BN,NS>;
    if(smem>48000) chk(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,smem),"sm");
    k<<<grid,256,smem>>>(ta,tb,tsfa,sb,C,M,N,K,num_sms);
}

void ref(const uint8_t*A,const uint8_t*B,const float*sa,const float*sb,float*C,int M,int N,int K){
    int kb=(K+127)/128;
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){
        float s=0; for(int kk=0;kk<kb;kk++){
            float sc=sa[m*kb+kk]*sb[(n/128)*kb+kk],p=0;
            for(int k=kk*128;k<std::min((kk+1)*128,K);k++){
                float a=__half2float(__nv_cvt_fp8_to_halfraw(A[m*K+k],__NV_E4M3));
                float b=__half2float(__nv_cvt_fp8_to_halfraw(B[n*K+k],__NV_E4M3));
                p+=a*b;}
            s+=p*sc;}
        C[m*N+n]=s;}
}

struct Cfg{const char*n;int M,K,N;};

int main(){
    cuInit(0);
    constexpr int BN=128, NS=5;

    Cfg cfgs[]={
        {"o_proj_M2048",    2048, 2048, 7168},
        {"q_b_proj_M8192",  8192, 1536, 3072},
        {"kv_b_proj_M8192", 8192, 512,  4096},
        {"o_proj_M4096",    4096, 2048, 7168},
        {"o_proj_M8192",    8192, 2048, 7168},
    };
    const int ncfg=5;

    bool all_valid=true;
    for(int ci=0;ci<ncfg;ci++){
        auto& c=cfgs[ci];
        int M=c.M, K=c.K, N=c.N;
        int kb=(K+127)/128, nb=(N+127)/128;
        void*dA,*dB; float*dsa,*dsb,*dsa_t; __nv_bfloat16*dC;
        chk(cudaMalloc(&dA,(size_t)M*K),"");chk(cudaMalloc(&dB,(size_t)N*K),"");
        chk(cudaMalloc(&dsa,M*kb*4),"");chk(cudaMalloc(&dsa_t,M*kb*4),"");
        chk(cudaMalloc(&dsb,nb*kb*4),"");
        chk(cudaMalloc(&dC,(size_t)M*N*2),"");
        std::vector<uint8_t>hA(M*K),hB(N*K);
        std::vector<float>hsa(M*kb),hsb(nb*kb);
        srand(42+ci);
        for(auto&v:hA){v=rand()%254; if(v>=0x7F)v++;}
        for(auto&v:hB){v=rand()%254; if(v>=0x7F)v++;}
        for(auto&v:hsa)v=0.5f+(rand()%100)/200.0f;
        for(auto&v:hsb)v=0.5f+(rand()%100)/200.0f;
        chk(cudaMemcpy(dA,hA.data(),M*K,cudaMemcpyHostToDevice),"");
        chk(cudaMemcpy(dB,hB.data(),N*K,cudaMemcpyHostToDevice),"");
        chk(cudaMemcpy(dsa,hsa.data(),M*kb*4,cudaMemcpyHostToDevice),"");
        chk(cudaMemcpy(dsb,hsb.data(),nb*kb*4,cudaMemcpyHostToDevice),"");
        transpose_sfa_kernel<<<(M*kb+255)/256,256>>>(dsa,dsa_t,M,kb);
        chk(cudaDeviceSynchronize(),"");
        chk(cudaMemset(dC,0,(size_t)M*N*2),"");

        run<BN,NS>(dA,dB,dsa_t,dsb,dC,M,N,K);
        chk(cudaDeviceSynchronize(),"sync");

        int checkM=std::min(M,128);
        std::vector<float>hr((size_t)checkM*N);
        ref(hA.data(),hB.data(),hsa.data(),hsb.data(),hr.data(),checkM,N,K);
        std::vector<__nv_bfloat16>hC((size_t)M*N);
        chk(cudaMemcpy(hC.data(),dC,(size_t)M*N*2,cudaMemcpyDeviceToHost),"");
        float max_abs=0;
        for(int i=0;i<checkM*N;i++){
            float r=hr[i];
            if(!isnan(r)&&fabsf(r)>max_abs) max_abs=fabsf(r);
        }
        float atol=max_abs*1e-3f;
        float me=0; int nan_cnt=0, bad_cnt=0;
        for(int i=0;i<checkM*N;i++){
            float g=__bfloat162float(hC[i]),r=hr[i];
            if(isnan(g)||isnan(r)){nan_cnt++;continue;}
            float denom=fmaxf(fabsf(r),atol);
            float e=fabsf(g-r)/denom;
            if(e>me)me=e;
            if(e>0.01f) bad_cnt++;
        }
        bool valid=(me<0.02f && nan_cnt==0);
        fprintf(stderr,"%s: max_err=%.6f nan=%d bad=%d %s\n",c.n,me,nan_cnt,bad_cnt,valid?"OK":"INVALID");
        if(!valid) all_valid=false;
        cudaFree(dA);cudaFree(dB);cudaFree(dsa);cudaFree(dsa_t);cudaFree(dsb);cudaFree(dC);
    }

    if(!all_valid){
        printf("KERNEL_RESULT {");
        bool first=true;
        for(int ci=0;ci<ncfg;ci++){
            if(!first)printf(", ");first=false;
            printf("\"%s\": 0.0",cfgs[ci].n);
        }
        printf("}\n");
        return 1;
    }

    // Benchmark
    size_t flush_sz=128*1024*1024;
    char*d_flush; chk(cudaMalloc(&d_flush,flush_sz),"");

    printf("KERNEL_RESULT {");
    bool first=true;
    for(int ci=0;ci<ncfg;ci++){
        auto&c=cfgs[ci];
        int M=c.M,K=c.K,N=c.N;
        int kb=(K+127)/128,nb=(N+127)/128;
        void*dA,*dB;float*dsa,*dsb,*dsa_t;__nv_bfloat16*dC;
        chk(cudaMalloc(&dA,(size_t)M*K),"");chk(cudaMalloc(&dB,(size_t)N*K),"");
        chk(cudaMalloc(&dsa,M*kb*4),"");chk(cudaMalloc(&dsa_t,M*kb*4),"");
        chk(cudaMalloc(&dsb,nb*kb*4),"");
        chk(cudaMalloc(&dC,(size_t)M*N*2),"");

        {std::vector<uint8_t>h(M*K);for(auto&v:h)v=rand()%256;
         chk(cudaMemcpy(dA,h.data(),M*K,cudaMemcpyHostToDevice),"");}
        {std::vector<uint8_t>h(N*K);for(auto&v:h)v=rand()%256;
         chk(cudaMemcpy(dB,h.data(),N*K,cudaMemcpyHostToDevice),"");}
        {std::vector<float>h(M*kb,1.0f);
         chk(cudaMemcpy(dsa,h.data(),M*kb*4,cudaMemcpyHostToDevice),"");}
        {std::vector<float>h(nb*kb,1.0f);
         chk(cudaMemcpy(dsb,h.data(),nb*kb*4,cudaMemcpyHostToDevice),"");}
        transpose_sfa_kernel<<<(M*kb+255)/256,256>>>(dsa,dsa_t,M,kb);
        chk(cudaDeviceSynchronize(),"");

        for(int i=0;i<20;i++) run<BN,NS>(dA,dB,dsa_t,dsb,dC,M,N,K);
        chk(cudaDeviceSynchronize(),"");

        cudaEvent_t t0,t1;
        cudaEventCreate(&t0);cudaEventCreate(&t1);
        std::vector<float>ts(100);
        for(int it=0;it<100;it++){
            chk(cudaMemset(d_flush,0,flush_sz),"");
            cudaEventRecord(t0);
            run<BN,NS>(dA,dB,dsa_t,dsb,dC,M,N,K);
            cudaEventRecord(t1);cudaEventSynchronize(t1);
            float ms;cudaEventElapsedTime(&ms,t0,t1);ts[it]=ms;
        }
        std::sort(ts.begin(),ts.end());
        float med=ts[50];
        double tflops=2.0*M*N*K/(med/1000.0)/1e12;

        if(!first)printf(", ");first=false;
        printf("\"%s\": %.4f",c.n,tflops);
        fprintf(stderr,"%s: %.4f TFLOPS, %.1f us\n",c.n,tflops,med*1000);

        cudaEventDestroy(t0);cudaEventDestroy(t1);
        cudaFree(dA);cudaFree(dB);cudaFree(dsa);cudaFree(dsa_t);cudaFree(dsb);cudaFree(dC);
    }
    printf("}\n");

    cudaFree(d_flush);

    // Run baseline reference
    fflush(stdout);
    (void)system("cd /home/xinhaoc/repos/ferret && python3 baselines/fp8-gemm/baseline_prefill_large_m.py 2>/dev/null | grep TFLOPS > /tmp/ref_out.txt");
    FILE* rf = fopen("/tmp/ref_out.txt","r");
    if(rf){
        double ref_vals[5] = {0};
        const char* ref_names[5] = {"o_proj_M2048","q_b_proj_M8192","kv_b_proj_M8192","o_proj_M4096","o_proj_M8192"};
        char line[256];
        while(fgets(line,sizeof(line),rf)){
            for(int i=0;i<5;i++){
                if(strstr(line,ref_names[i])){
                    char* p=strstr(line,":");
                    if(p) ref_vals[i]=atof(p+1);
                }
            }
        }
        fclose(rf);
        printf("KERNEL_RESULT_REFERENCE {");
        for(int i=0;i<5;i++){
            if(i)printf(", ");
            printf("\"%s\": %.2f",ref_names[i],ref_vals[i]);
        }
        printf("}\n");
    }
    return 0;
}
