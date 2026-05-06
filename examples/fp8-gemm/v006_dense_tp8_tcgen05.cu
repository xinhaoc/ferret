// FP8 Block-Scaled GEMM for Blackwell (SM100a) - Persistent kernel v3
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <vector>

static constexpr int WARP_SIZE = 32;

__device__ __forceinline__ uint32_t elect_one_sync() {
    uint32_t pred = 0;
    asm volatile("{\n\t.reg .pred %%px;\n\telect.sync _|%%px, %1;\n\t@%%px mov.s32 %0, 1;\n\t}"
        : "+r"(pred) : "r"(0xFFFFFFFF));
    return pred;
}

__device__ __forceinline__ void mb_init(int a, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(c)); }

__device__ __forceinline__ void mb_wait(int a, int p) {
    asm volatile("{\n\t.reg .pred P1;\n\tLW:\n\tmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t@P1 bra.uni DN;\n\tbra.uni LW;\n\tDN:\n\t}" :: "r"(a), "r"(p), "r"(0x989680));
}

__device__ __forceinline__ void mb_arrive(int a) { asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(a) : "memory"); }
__device__ __forceinline__ void mb_arrive_tx(int a, int s) { asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(s) : "memory"); }

__device__ __forceinline__ void tma_ld(int d, const void *t, int x, int y, int m) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" :: "r"(d), "l"(t), "r"(x), "r"(y), "r"(m) : "memory");
}

__device__ __forceinline__ constexpr uint64_t denc(uint64_t x) { return (x & 0x3FFFFULL) >> 4ULL; }
__device__ __forceinline__ uint64_t mkdesc(int a) { return denc(a) | (denc(1024) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL); }

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
    constexpr int BM=128, BK=128, UK=32, NE=2;
    const int tid=threadIdx.x, wid=tid/32, lid=tid%32;
    const int nn=(N+BN-1)/BN, nk=(K+BK-1)/BK;
    const int total=((M+BM-1)/BM)*nn;

    extern __shared__ __align__(1024) uint8_t sm[];
    constexpr int SA=BM*BK, SB=BN*BK;
    auto sA=[&](int s){return sm+s*SA;};
    auto sB=[&](int s){return sm+NS*SA+s*SB;};

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
        for(int i=0;i<NS;i++){mb_init(bf+i*8,1);mb_init(be+i*8,1);} // empty signaled by MMA commit
        for(int i=0;i<NE;i++){mb_init(btf+i*8,1);mb_init(bte+i*8,128);}
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if(wid==2){
        int a=__cvta_generic_to_shared(tp);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(a),"r"(TCA));
    }
    __syncthreads();
    const uint32_t taddr=*tp;
    constexpr uint32_t idesc=(1u<<4)|((uint32_t)(BN/8)<<17)|(8u<<24);

    // Persistent scheduling - each role maintains its own phase
    if(wid==0&&elect_one_sync()){
        // TMA LOAD
        int ph=0; // phase for empty barriers
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
        // MMA ISSUE
        int ph=0; // phase for full barriers
        int gki=0; // global K iteration counter
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
                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t}\n"
                        ::"r"(tc),"l"(ad),"l"(bd),"r"(idesc),"r"(en),"r"(0u),"r"(0u),"r"(0u),"r"(0u));
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                // Signal both: empty (TMA can reuse SMEM) + tmem_full (epilogue can read TMEM)
                asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    ::"r"(be+s*8):"memory");
                asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    ::"r"(btf+ai*8):"memory");
            }
        }
    } else if(wid>=4){
        // EPILOGUE
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
                int s=ki%NS;
                float sfa=(mi<M)?__ldg(sa+mi*nk+ki):0.0f;
                float sfb0=__ldg(sb+(on/128)*nk+ki);
                float sfb1=0.0f;
                if(BN>128) sfb1=__ldg(sb+((on+128)/128)*nk+ki);

                int ai=gki%NE;
                int ap=(gki/NE)&1;
                mb_wait(btf+ai*8, ap);
                asm volatile("tcgen05.fence::after_thread_sync;");
                // No empty barrier arrive needed - MMA commit signals it

                float sf0=sfa*sfb0, sf1=sfa*sfb1;
                #pragma unroll
                for(int i=0;i<BN/16;i++){
                    uint32_t ta_=taddr+((ew*32)<<16)+ai*BN+i*16;
                    float v[16];
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                        :"=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]),"=f"(v[4]),"=f"(v[5]),"=f"(v[6]),"=f"(v[7]),
                         "=f"(v[8]),"=f"(v[9]),"=f"(v[10]),"=f"(v[11]),"=f"(v[12]),"=f"(v[13]),"=f"(v[14]),"=f"(v[15])
                        :"r"(ta_));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float sf=(BN<=128||i*16<128)?sf0:sf1;
                    #pragma unroll
                    for(int j=0;j<16;j++) acc[i*16+j]+=v[j]*sf;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mb_arrive(bte+ai*8);
            }

            // Write output with v8.b32 stores and L1::no_allocate
            if(mi<M){
                __nv_bfloat16* row=C+mi*N+on;
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
                        asm volatile("st.relaxed.cta.global.L1::no_allocate.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
                            :: "l"(row+n), "r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(r4),"r"(r5),"r"(r6),"r"(r7) : "memory");
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
     chk(cuTensorMapEncodeTiled(&ta,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,(void*)A,g,s,b,e,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),"ta");}
    {uint64_t g[2]={(uint64_t)K,(uint64_t)N};uint64_t s[1]={(uint64_t)K};uint32_t b[2]={128,(uint32_t)BN};uint32_t e[2]={1,1};
     chk(cuTensorMapEncodeTiled(&tb,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,(void*)B,g,s,b,e,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),"tb");}
    int num_sms; cudaDeviceGetAttribute(&num_sms,cudaDevAttrMultiProcessorCount,0);
    int total=((M+127)/128)*((N+BN-1)/BN);
    // Minimize grid size: use ceil(total/num_waves) SMs for balanced workload
    int num_waves = (total + num_sms - 1) / num_sms;
    int grid = (total + num_waves - 1) / num_waves;
    grid = std::min(grid, num_sms);
    int sm=NS*(128*128+BN*128)+(NS*2+4)*8+8;
    sm=(sm+1023)&~1023;
    auto k=fp8_gemm<BN,NS>;
    if(sm>48000) chk(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,sm),"sm");
    k<<<grid,256,sm>>>(ta,tb,sa,sb,C,M,N,K,num_sms);
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
    Cfg cfgs[]={
        {"q_b_proj_M1",1,1536,3072},{"kv_b_proj_M1",1,512,4096},{"o_proj_M1",1,2048,7168},
        {"q_b_proj_M16",16,1536,3072},{"kv_b_proj_M16",16,512,4096},{"o_proj_M16",16,2048,7168},
        {"q_b_proj_M512",512,1536,3072},{"o_proj_M512",512,2048,7168},
    };
    printf("KERNEL_RESULT {"); bool f=true;
    for(auto&c:cfgs){
        int M=c.M,K=c.K,N=c.N;
        int kb=(K+127)/128,nb=(N+127)/128;
        size_t as=M*K,bs=N*K,ss=M*kb,sbs=nb*kb,cs=M*N;
        void*dA,*dB;float*dsa,*dsb;__nv_bfloat16*dC;
        chk(cudaMalloc(&dA,as),"");chk(cudaMalloc(&dB,bs),"");
        chk(cudaMalloc(&dsa,ss*4),"");chk(cudaMalloc(&dsb,sbs*4),"");
        chk(cudaMalloc(&dC,cs*2),"");
        std::vector<uint8_t>hA(as),hB(bs);std::vector<float>hsa(ss),hsb(sbs);
        srand(42);
        for(size_t i=0;i<as;i++)hA[i]=rand()%256;
        for(size_t i=0;i<bs;i++)hB[i]=rand()%256;
        for(size_t i=0;i<ss;i++)hsa[i]=0.5f+(rand()%100)/200.0f;
        for(size_t i=0;i<sbs;i++)hsb[i]=0.5f+(rand()%100)/200.0f;
        chk(cudaMemcpy(dA,hA.data(),as,cudaMemcpyHostToDevice),"");
        chk(cudaMemcpy(dB,hB.data(),bs,cudaMemcpyHostToDevice),"");
        chk(cudaMemcpy(dsa,hsa.data(),ss*4,cudaMemcpyHostToDevice),"");
        chk(cudaMemcpy(dsb,hsb.data(),sbs*4,cudaMemcpyHostToDevice),"");

        chk(cudaMemset(dC,0,cs*2),"");
        run<128,4>(dA,dB,dsa,dsb,dC,M,N,K);
        chk(cudaDeviceSynchronize(),"sync");

        std::vector<float>hr(cs);ref(hA.data(),hB.data(),hsa.data(),hsb.data(),hr.data(),M,N,K);
        std::vector<__nv_bfloat16>hC(cs);
        chk(cudaMemcpy(hC.data(),dC,cs*2,cudaMemcpyDeviceToHost),"");
        float me=0;
        for(size_t i=0;i<cs;i++){float g=__bfloat162float(hC[i]),r=hr[i];
            float e=(r!=0)?fabsf(g-r)/fabsf(r):fabsf(g);if(e>me)me=e;}
        if(me>0.01f)fprintf(stderr,"WARN:%s err=%.4f\n",c.n,me);

        for(int i=0;i<20;i++)run<128,4>(dA,dB,dsa,dsb,dC,M,N,K);
        chk(cudaDeviceSynchronize(),"");
        size_t fsz=128*1024*1024;char*df;chk(cudaMalloc(&df,fsz),"");
        cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);
        std::vector<float>ts(100);
        for(int it=0;it<100;it++){
            chk(cudaMemset(df,0,fsz),"");cudaEventRecord(t0);
            run<128,4>(dA,dB,dsa,dsb,dC,M,N,K);
            cudaEventRecord(t1);cudaEventSynchronize(t1);
            float ms;cudaEventElapsedTime(&ms,t0,t1);ts[it]=ms;}
        std::sort(ts.begin(),ts.end());float med=ts[50];
        double tflops=2.0*M*N*K/(med/1000.0)/1e12;
        if(!f)printf(", ");f=false;
        printf("\"%s\": %.4f",c.n,tflops);
        fprintf(stderr,"%s: %.4f TFLOPS, %.1f us, err=%.6f\n",c.n,tflops,med*1000,me);
        cudaEventDestroy(t0);cudaEventDestroy(t1);
        cudaFree(df);cudaFree(dA);cudaFree(dB);cudaFree(dsa);cudaFree(dsb);cudaFree(dC);
    }
    printf("}\n");return 0;
}
