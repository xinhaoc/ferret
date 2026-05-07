// FP8 Block-Scaled GEMM for Blackwell (SM100a) - Prefill (large M)
// Based on v006_dense_tp8_tcgen05.cu architecture
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

template <int BN, int NS>
__global__ void __launch_bounds__(256, 1)
fp8_gemm(
    const __grid_constant__ CUtensorMap ta,
    const __grid_constant__ CUtensorMap tb,
    const float* __restrict__ sa,
    const float* __restrict__ sb,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K, int num_sms
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
    constexpr int BM=128, BK=128, UK=32, NE=4;
    const int tid=threadIdx.x, wid=tid/32;
    const int nn=(N+BN-1)/BN, nk=(K+BK-1)/BK;
    const int total=((M+BM-1)/BM)*nn;

    extern __shared__ __align__(1024) uint8_t sm[];
    constexpr int SA=BM*BK, SB=BN*BK;
    auto sA=[&](int s)->uint8_t*{return sm+s*SA;};
    auto sB=[&](int s)->uint8_t*{return sm+NS*SA+s*SB;};

    auto bars=reinterpret_cast<uint64_t*>(sm+NS*(SA+SB));
    int bf=__cvta_generic_to_shared(bars);
    int be=bf+NS*8, btf=be+NS*8, bte=btf+NE*8;
    auto tp=reinterpret_cast<uint32_t*>(bars+NS*2+NE*2);
    constexpr int TC=NE*BN;
    constexpr int TCA=TC<=32?32:TC<=64?64:TC<=128?128:TC<=256?256:512;

    if(wid==0&&elect_one_sync()){
        asm volatile("prefetch.tensormap [%0];"::"l"(&ta));
        asm volatile("prefetch.tensormap [%0];"::"l"(&tb));
    }
    if(wid==1&&elect_one_sync()){
        for(int i=0;i<NS;i++){mb_init(bf+i*8,1);mb_init(be+i*8,1);}
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
        // TMA LOAD warp
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
                int mb=bf+s*8;
                tma_ld(as_,&ta,ki*BK,om,mb);
                tma_ld(bs_,&tb,ki*BK,on,mb);
                mb_arrive_tx(mb,SA+SB);
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
                int s=ki%NS;
                mb_wait(bf+s*8, ph);
                if(s==NS-1) ph^=1;

                int ai=gki%NE;
                int ap=(gki/NE)&1;
                mb_wait(bte+ai*8, ap^1);

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
                asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    ::"r"(be+s*8):"memory");
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
                float sfa=(mi<M)?__ldg(sa+mi*nk+ki):0.0f;
                float sfb0=__ldg(sb+(on/128)*nk+ki);
                float sfb1=0.0f;
                if(BN>128) sfb1=__ldg(sb+((on+128)/128)*nk+ki);

                int ai=gki%NE;
                int ap=(gki/NE)&1;
                mb_wait(btf+ai*8, ap);
                asm volatile("tcgen05.fence::after_thread_sync;");

                float sf0=sfa*sfb0, sf1=sfa*sfb1;
                #pragma unroll
                for(int i=0;i<BN/16;i++){
                    uint32_t ta_=taddr+((ew*32)<<16)+ai*BN+i*16;
                    float v[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                        :"=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]),
                         "=f"(v[4]),"=f"(v[5]),"=f"(v[6]),"=f"(v[7]),
                         "=f"(v[8]),"=f"(v[9]),"=f"(v[10]),"=f"(v[11]),
                         "=f"(v[12]),"=f"(v[13]),"=f"(v[14]),"=f"(v[15])
                        :"r"(ta_));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float sf=(BN<=128||i*16<128)?sf0:sf1;
                    #pragma unroll
                    for(int j=0;j<16;j++) acc[i*16+j]+=v[j]*sf;
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
void run(const void*A,const void*B,const float*sa,const float*sb,__nv_bfloat16*C,int M,int N,int K){
    CUtensorMap ta,tb;
    {uint64_t g[2]={(uint64_t)K,(uint64_t)M};uint64_t s[1]={(uint64_t)K};uint32_t b[2]={128,128};uint32_t e[2]={1,1};
     chk(cuTensorMapEncodeTiled(&ta,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,(void*)A,g,s,b,e,
         CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),"ta");}
    {uint64_t g[2]={(uint64_t)K,(uint64_t)N};uint64_t s[1]={(uint64_t)K};uint32_t b[2]={128,(uint32_t)BN};uint32_t e[2]={1,1};
     chk(cuTensorMapEncodeTiled(&tb,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,(void*)B,g,s,b,e,
         CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),"tb");}
    int num_sms; cudaDeviceGetAttribute(&num_sms,cudaDevAttrMultiProcessorCount,0);
    int total=((M+127)/128)*((N+BN-1)/BN);
    int num_waves = (total + num_sms - 1) / num_sms;
    int grid = (total + num_waves - 1) / num_waves;
    grid = std::min(grid, num_sms);
    int sm_sz=NS*(128*128+BN*128)+(NS*2+4)*8+8;
    sm_sz=(sm_sz+1023)&~1023;
    auto k=fp8_gemm<BN,NS>;
    if(sm_sz>48000) chk(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,sm_sz),"sm");
    k<<<grid,256,sm_sz>>>(ta,tb,sa,sb,C,M,N,K,num_sms);
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

    // All 15 prefill configs
    Cfg cfgs[]={
        {"q_b_proj_M512",   512,  1536, 3072},
        {"kv_b_proj_M512",  512,  512,  4096},
        {"o_proj_M512",     512,  2048, 7168},
        {"q_b_proj_M1024",  1024, 1536, 3072},
        {"kv_b_proj_M1024", 1024, 512,  4096},
        {"o_proj_M1024",    1024, 2048, 7168},
        {"q_b_proj_M2048",  2048, 1536, 3072},
        {"kv_b_proj_M2048", 2048, 512,  4096},
        {"o_proj_M2048",    2048, 2048, 7168},
        {"q_b_proj_M4096",  4096, 1536, 3072},
        {"kv_b_proj_M4096", 4096, 512,  4096},
        {"o_proj_M4096",    4096, 2048, 7168},
        {"q_b_proj_M8192",  8192, 1536, 3072},
        {"kv_b_proj_M8192", 8192, 512,  4096},
        {"o_proj_M8192",    8192, 2048, 7168},
    };

    // Validate correctness on smallest config
    {
        auto& c = cfgs[0]; // q_b_proj_M512
        int M=c.M, K=c.K, N=c.N;
        int kb=(K+127)/128, nb=(N+127)/128;
        void*dA,*dB; float*dsa,*dsb; __nv_bfloat16*dC;
        chk(cudaMalloc(&dA,(size_t)M*K),"");chk(cudaMalloc(&dB,(size_t)N*K),"");
        chk(cudaMalloc(&dsa,M*kb*4),"");chk(cudaMalloc(&dsb,nb*kb*4),"");
        chk(cudaMalloc(&dC,(size_t)M*N*2),"");
        std::vector<uint8_t>hA(M*K),hB(N*K);
        std::vector<float>hsa(M*kb),hsb(nb*kb);
        srand(42);
        // Generate valid FP8 E4M3 values (avoid NaN: 0x7F and 0xFF)
        for(auto&v:hA){v=rand()%254; if(v>=0x7F)v++;}  // skip 0x7F
        for(auto&v:hB){v=rand()%254; if(v>=0x7F)v++;}  // skip 0x7F
        for(auto&v:hsa)v=0.5f+(rand()%100)/200.0f;
        for(auto&v:hsb)v=0.5f+(rand()%100)/200.0f;
        chk(cudaMemcpy(dA,hA.data(),M*K,cudaMemcpyHostToDevice),"");
        chk(cudaMemcpy(dB,hB.data(),N*K,cudaMemcpyHostToDevice),"");
        chk(cudaMemcpy(dsa,hsa.data(),M*kb*4,cudaMemcpyHostToDevice),"");
        chk(cudaMemcpy(dsb,hsb.data(),nb*kb*4,cudaMemcpyHostToDevice),"");
        chk(cudaMemset(dC,0,(size_t)M*N*2),"");

        run<128,4>(dA,dB,dsa,dsb,dC,M,N,K);
        chk(cudaDeviceSynchronize(),"sync");

        std::vector<float>hr(M*N);
        ref(hA.data(),hB.data(),hsa.data(),hsb.data(),hr.data(),M,N,K);
        std::vector<__nv_bfloat16>hC(M*N);
        chk(cudaMemcpy(hC.data(),dC,(size_t)M*N*2,cudaMemcpyDeviceToHost),"");

        // Compute max absolute value for relative error threshold
        float max_abs=0;
        for(int i=0;i<M*N;i++){
            float r=hr[i];
            if(!isnan(r)&&fabsf(r)>max_abs) max_abs=fabsf(r);
        }
        float atol=max_abs*1e-3f; // absolute tolerance based on scale
        float me=0; int worst_idx=0; int nan_cnt=0; int bad_cnt=0;
        for(int i=0;i<M*N;i++){
            float g=__bfloat162float(hC[i]),r=hr[i];
            if(isnan(g)||isnan(r)){nan_cnt++;continue;}
            float denom=fmaxf(fabsf(r),atol);
            float e=fabsf(g-r)/denom;
            if(e>me){me=e;worst_idx=i;}
            if(e>0.01f) bad_cnt++;
        }
        if(nan_cnt>0) fprintf(stderr,"  WARNING: %d NaN values found\n",nan_cnt);
        for(int i=0;i<3;i++){
            fprintf(stderr,"  [%d] gpu=%.4f ref=%.4f\n",i,__bfloat162float(hC[i]),hr[i]);
        }
        fprintf(stderr,"Validation %s: max_rel_err=%.6f bad_elements=%d/%d max_abs=%.1f\n",
                c.n,me,bad_cnt,M*N,max_abs);
        if(me>0.05f){
            fprintf(stderr,"INVALID: max_rel_err %.4f > 0.05 for %s\n",me,c.n);
            fprintf(stderr,"  got=%.6f ref=%.6f\n",__bfloat162float(hC[worst_idx]),hr[worst_idx]);
        }
        cudaFree(dA);cudaFree(dB);cudaFree(dsa);cudaFree(dsb);cudaFree(dC);
    }

    // Benchmark all configs
    size_t flush_sz=128*1024*1024;
    char*d_flush; chk(cudaMalloc(&d_flush,flush_sz),"");

    printf("KERNEL_RESULT {");
    bool first=true;
    for(auto&c:cfgs){
        int M=c.M,K=c.K,N=c.N;
        int kb=(K+127)/128,nb=(N+127)/128;
        void*dA,*dB;float*dsa,*dsb;__nv_bfloat16*dC;
        chk(cudaMalloc(&dA,(size_t)M*K),"");chk(cudaMalloc(&dB,(size_t)N*K),"");
        chk(cudaMalloc(&dsa,M*kb*4),"");chk(cudaMalloc(&dsb,nb*kb*4),"");
        chk(cudaMalloc(&dC,(size_t)M*N*2),"");

        // Random init
        {std::vector<uint8_t>h(M*K);for(auto&v:h)v=rand()%256;
         chk(cudaMemcpy(dA,h.data(),M*K,cudaMemcpyHostToDevice),"");}
        {std::vector<uint8_t>h(N*K);for(auto&v:h)v=rand()%256;
         chk(cudaMemcpy(dB,h.data(),N*K,cudaMemcpyHostToDevice),"");}
        {std::vector<float>h(M*kb,1.0f);
         chk(cudaMemcpy(dsa,h.data(),M*kb*4,cudaMemcpyHostToDevice),"");}
        {std::vector<float>h(nb*kb,1.0f);
         chk(cudaMemcpy(dsb,h.data(),nb*kb*4,cudaMemcpyHostToDevice),"");}

        // Warmup
        for(int i=0;i<20;i++) run<128,4>(dA,dB,dsa,dsb,dC,M,N,K);
        chk(cudaDeviceSynchronize(),"");

        // Benchmark
        cudaEvent_t t0,t1;
        cudaEventCreate(&t0);cudaEventCreate(&t1);
        std::vector<float>ts(100);
        for(int it=0;it<100;it++){
            chk(cudaMemset(d_flush,0,flush_sz),"");
            cudaEventRecord(t0);
            run<128,4>(dA,dB,dsa,dsb,dC,M,N,K);
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
        cudaFree(dA);cudaFree(dB);cudaFree(dsa);cudaFree(dsb);cudaFree(dC);
    }
    printf("}\n");

    cudaFree(d_flush);
    return 0;
}
