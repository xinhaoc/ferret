// MLA Prefill — Unabsorbed, TMA KV loads + mma.sync compute
// TMA loads K/V with 128B swizzle into [BN,64] blocks. Compute reads from these blocks.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <vector>

using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;

static constexpr int D_QK_NOPE=128, D_QK_ROPE=64, D_QK=192, D_V=128;
static constexpr int BM=64, BN=128, NT=128, MK=16;
static constexpr int HALF_N=BN/16/2; // 4
static constexpr int NMDK=D_QK_NOPE/MK; // 8
static constexpr int NMRK=D_QK_ROPE/MK; // 4
static constexpr int NMDV=D_V/16;        // 8

// SMEM: Q_nope + Q_pe + 5 TMA blocks of [BN,64] with 128B swizzle
static constexpr int Q_NOPE_SZ = BM*D_QK_NOPE*2;  // 16KB
static constexpr int Q_PE_SZ   = BM*D_QK_ROPE*2;   // 8KB
static constexpr int TMA_BLK   = BN*64*2;           // 16KB per [BN,64] block
// Layout: Q_nope | Q_pe | KN0 | KN1 | KP | V0 | V1 | mbar
static constexpr int KN0_OFF = Q_NOPE_SZ + Q_PE_SZ;
static constexpr int KN1_OFF = KN0_OFF + TMA_BLK;
static constexpr int KP_OFF  = KN1_OFF + TMA_BLK;
static constexpr int V0_OFF  = KP_OFF + TMA_BLK;
static constexpr int V1_OFF  = V0_OFF + TMA_BLK;
static constexpr int MBAR_OFF= V1_OFF + TMA_BLK;
static constexpr int SMEM_SZ = MBAR_OFF + 16;

// Swizzle for 128B: row stride S bytes, 16B chunks XOR'd by (r%8)
template<int S> __device__ __forceinline__ int swz(int r, int c){
    if constexpr(S>=128) c^=(r%8)/(128/S>1?128/S:1);
    return r*S+c*16;
}
__device__ __forceinline__ void ldm4(uint32_t r[4],int a){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
        :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a));
}
__device__ __forceinline__ void ldm4t(uint32_t r[4],int a){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];\n"
        :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a));
}
__device__ __forceinline__ void hmma(const uint32_t A[4],const uint32_t B[2],float C[4]){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        :"+f"(C[0]),"+f"(C[1]),"+f"(C[2]),"+f"(C[3])
        :"r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),"r"(B[0]),"r"(B[1]));
}
__device__ __forceinline__ void hmma0(const uint32_t A[4],const uint32_t B[2],float C[4]){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        :"=f"(C[0]),"=f"(C[1]),"=f"(C[2]),"=f"(C[3])
        :"r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),"r"(B[0]),"r"(B[1]),
         "f"(0.f),"f"(0.f),"f"(0.f),"f"(0.f));
}
__device__ __forceinline__ void hmma16(const uint32_t A[4],const uint32_t B[4],float C[8]){hmma(A,&B[0],&C[0]);hmma(A,&B[2],&C[4]);}
__device__ __forceinline__ void hmma16_0(const uint32_t A[4],const uint32_t B[4],float C[8]){hmma0(A,&B[0],&C[0]);hmma0(A,&B[2],&C[4]);}
__device__ __forceinline__ void rowsum(float*d,uint32_t*s){
    asm volatile("{mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,_,%1,_},{%2,%3,%4,%5},{%6,%7},{%0,0.,%1,0.};}\n"
        :"+f"(d[0]),"+f"(d[1]):"r"(s[0]),"r"(s[1]),"r"(s[2]),"r"(s[3]),
         "r"(1065369472u),"r"(1065369472u));
}
__device__ __forceinline__ void cpa(int d,const void*s){asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(d),"l"(s));}
__device__ __forceinline__ void cpa_p(int d,const void*s,bool p){
    asm volatile("{.reg .pred q; setp.ne.b32 q,%2,0; @!q st.shared.v4.u32 [%0],{0,0,0,0}; @q cp.async.cg.shared.global [%0],[%1],16;}\n"
        ::"r"(d),"l"(s),"r"((int)p));}
__device__ __forceinline__ void cpa_commit(){asm volatile("cp.async.commit_group;\n");}
template<int N> __device__ __forceinline__ void cpa_wait(){asm volatile("cp.async.wait_group %0;\n"::"n"(N));}
__device__ __forceinline__ float sxor(float v,int m){return __shfl_xor_sync(0xffffffff,v,m);}
__device__ __forceinline__ float ex2(float x){float r;asm volatile("ex2.approx.ftz.f32 %0,%1;\n":"=f"(r):"f"(x));return r;}
__device__ __forceinline__ uint32_t f2b(float a,float b){bf16_2 v=__float22bfloat162_rn(make_float2(a,b));return *(uint32_t*)&v;}
__host__ __device__ __forceinline__ int cdiv(int a,int b){return(a+b-1)/b;}

__device__ __forceinline__ void tma3d(const CUtensorMap*d,int sa,int mb,int c0,int c1,int c2){
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4}],[%5];"
        ::"r"(sa),"l"((uint64_t)d),"r"(c0),"r"(c1),"r"(c2),"r"(mb):"memory");}
__device__ __forceinline__ void mbar_init(int a,int c){asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(a),"r"(c));}
__device__ __forceinline__ void mbar_tx(int a,int b){asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(a),"r"(b));}
__device__ __forceinline__ void mbar_wait(int a,int p){
    asm volatile("{.reg .pred P;\nW: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P,[%0],%1,0x989680;\n@P bra D;\n bra W;\nD:}"::"r"(a),"r"(p));}

__global__ __launch_bounds__(NT,2)
void mla_tma_kernel(
    const __grid_constant__ CUtensorMap k_tma,
    const __grid_constant__ CUtensorMap v_tma,
    const bf16*__restrict__ Qn,const bf16*__restrict__ Qp,
    bf16*__restrict__ O,const int S,const int H,const float sml2)
{
    const int head=blockIdx.x,qb=cdiv(S,BM)-1-blockIdx.y,qs=qb*BM;
    const int bat=blockIdx.z,tid=threadIdx.x,wid=tid/32,lid=tid%32;
    const long long bqn=(long long)bat*S*H*D_QK_NOPE,bqp=(long long)bat*S*H*D_QK_ROPE;
    const long long bo=(long long)bat*S*H*D_V;
    extern __shared__ __align__(128) uint8_t sm[];
    int sb=__cvta_generic_to_shared(sm);
    int qn_s=sb,qp_s=sb+Q_NOPE_SZ;
    int kn0=sb+KN0_OFF,kn1=sb+KN1_OFF,kps=sb+KP_OFF,v0s=sb+V0_OFF,v1s=sb+V1_OFF;
    int mbs=sb+MBAR_OFF;
    if(tid==0){mbar_init(mbs,1);asm volatile("fence.mbarrier_init.release.cluster;");}
    __syncthreads();
    // Load Q via cp.async
    {
        constexpr int SN=D_QK_NOPE*2,SP=D_QK_ROPE*2;
        for(int i=tid;i<BM*(D_QK_NOPE/8);i+=NT){int r=i/(D_QK_NOPE/8),c=i%(D_QK_NOPE/8),qi=qs+r;
            int a=qn_s+swz<SN>(r,c);
            if(qi<S)cpa(a,Qn+bqn+(long long)qi*H*D_QK_NOPE+(long long)head*D_QK_NOPE+c*8);
            else asm volatile("st.shared.v4.u32 [%0],{0,0,0,0};\n"::"r"(a));}
        for(int i=tid;i<BM*(D_QK_ROPE/8);i+=NT){int r=i/(D_QK_ROPE/8),c=i%(D_QK_ROPE/8),qi=qs+r;
            int a=qp_s+swz<SP>(r,c);
            if(qi<S)cpa(a,Qp+bqp+(long long)qi*H*D_QK_ROPE+(long long)head*D_QK_ROPE+c*8);
            else asm volatile("st.shared.v4.u32 [%0],{0,0,0,0};\n"::"r"(a));}
        cpa_commit();cpa_wait<0>();__syncthreads();
    }
    constexpr int SNB=D_QK_NOPE*2,SPB=D_QK_ROPE*2,S128=128;
    int qnl=qn_s+swz<SNB>(wid*16+(lid%16),lid/16);
    int qpl=qp_s+swz<(D_QK_ROPE*2)>(wid*16+(lid%16),lid/16);
    const int kr=(lid%8)+(lid/16)*8,kc=(lid%16)/8;

    float of[NMDV][8];
    #pragma unroll
    for(int i=0;i<NMDV;i++)for(int j=0;j<8;j++)of[i][j]=0.f;
    float ms[2]={-INFINITY,-INFINITY},ds[2]={1.f,1.f};
    float sf[HALF_N][8];
    int kvend=min(S,qs+BM),nt=cdiv(kvend,BN);
    int mph=0;

    auto tld=[&](int kvb){if(tid==0){
        mbar_tx(mbs,5*TMA_BLK);
        tma3d(&k_tma,kn0,mbs,0,kvb,0); tma3d(&k_tma,kn1,mbs,0,kvb,1);
        tma3d(&k_tma,kps,mbs,0,kvb,2);
        tma3d(&v_tma,v0s,mbs,0,kvb,0); tma3d(&v_tma,v1s,mbs,0,kvb,1);
    }};
    if(nt>0)tld(0);

    // K ldmatrix addresses: each block is [BN,64] with 128B row stride
    // kn0 block: mk=0..3 (first 64 nope elements)
    // kn1 block: mk=4..7 (next 64 nope elements)
    // kps block: mk=0..3 (64 rope elements)

    #pragma unroll 1
    for(int t=0;t<nt;t++){
        int kvb=t*BN;
        mbar_wait(mbs,mph);mph^=1;
        __syncthreads();

        #pragma unroll
        for(int half=0;half<2;half++){
            int noff=half*HALF_N;
            // QK
            {
                // K addresses with 128B stride per block
                int kpl=kps+swz<S128>(kr,kc);
                int kn0l=kn0+swz<S128>(kr,kc);
                int kn1l=kn1+swz<S128>(kr,kc);
                #pragma unroll
                for(int mk=0;mk<NMRK;mk++){
                    {uint32_t qr[4];ldm4(qr,qpl^(mk*32));
                     #pragma unroll
                     for(int nl=0;nl<HALF_N;nl++){uint32_t k2[4];
                        ldm4(k2,(kpl+(noff+nl)*16*S128)^(mk*32));
                        if(mk==0)hmma16_0(qr,k2,sf[nl]);else hmma16(qr,k2,sf[nl]);}}
                    {uint32_t qr[4];ldm4(qr,qnl^(mk*32));
                     #pragma unroll
                     for(int nl=0;nl<HALF_N;nl++){uint32_t k2[4];
                        ldm4(k2,(kn0l+(noff+nl)*16*S128)^(mk*32));
                        hmma16(qr,k2,sf[nl]);}}
                }
                // mk=4..7: nope only, from kn1 block (but mk offset within block is mk-4)
                #pragma unroll
                for(int mk=NMRK;mk<NMDK;mk++){
                    uint32_t qr[4];ldm4(qr,qnl^(mk*32));
                    #pragma unroll
                    for(int nl=0;nl<HALF_N;nl++){uint32_t k2[4];
                        ldm4(k2,(kn1l+(noff+nl)*16*S128)^((mk-4)*32));
                        hmma16(qr,k2,sf[nl]);}
                }
            }
            // Causal mask
            if(kvb+BN>qs){int qrb=qs+wid*16;
                #pragma unroll
                for(int nl=0;nl<HALF_N;nl++){
                    #pragma unroll
                    for(int ri=0;ri<8;ri++){
                        int rit=((ri&2)==0)?(lid/4):(lid/4+8);
                        int kvc=2*(lid%4)+((ri&4)?8:0)+(ri&1);
                        int qp=qrb+rit,kvp=kvb+(noff+nl)*16+kvc;
                        if(!((kvp<=qp)&&(kvp<S)))sf[nl][ri]=-INFINITY;}}}
            // Softmax
            {float mp[2]={ms[0],ms[1]};
                #pragma unroll
                for(int j=0;j<2;j++){
                    #pragma unroll
                    for(int nl=0;nl<HALF_N;nl++){float lm=fmaxf(fmaxf(sf[nl][j*2],sf[nl][j*2+1]),fmaxf(sf[nl][j*2+4],sf[nl][j*2+5]));ms[j]=fmaxf(ms[j],lm);}
                    ms[j]=fmaxf(ms[j],sxor(ms[j],0x2));ms[j]=fmaxf(ms[j],sxor(ms[j],0x1));
                    float nms=-(ms[j]*sml2);float sc=ex2(__fmaf_rn(mp[j],sml2,nms));ds[j]*=sc;
                    #pragma unroll
                    for(int md=0;md<NMDV;md++){of[md][j*2+0]*=sc;of[md][j*2+1]*=sc;of[md][j*2+4]*=sc;of[md][j*2+5]*=sc;}
                    #pragma unroll
                    for(int nl=0;nl<HALF_N;nl++){
                        sf[nl][j*2+0]=ex2(__fmaf_rn(sf[nl][j*2+0],sml2,nms));sf[nl][j*2+1]=ex2(__fmaf_rn(sf[nl][j*2+1],sml2,nms));
                        sf[nl][j*2+4]=ex2(__fmaf_rn(sf[nl][j*2+4],sml2,nms));sf[nl][j*2+5]=ex2(__fmaf_rn(sf[nl][j*2+5],sml2,nms));}}}
            // bf16 + rowsum
            uint32_t pf[HALF_N][4];
            #pragma unroll
            for(int nl=0;nl<HALF_N;nl++){
                #pragma unroll
                for(int i=0;i<4;i++)pf[nl][i]=f2b(sf[nl][i*2],sf[nl][i*2+1]);
                rowsum(ds,pf[nl]);}
            // PV: V stored as 2 blocks [BN,64] with 128B stride
            {int vr0=lid%16,vcb=lid/16;
                #pragma unroll
                for(int mkv=0;mkv<HALF_N;mkv++){
                    #pragma unroll
                    for(int md=0;md<NMDV;md++){
                        uint32_t vf[4];
                        // md=0..3: V block 0 (cols 0-63), md=4..7: V block 1 (cols 64-127)
                        int vs_base=(md<4)?v0s:v1s;
                        int md_local=(md<4)?md:(md-4);
                        ldm4t(vf,vs_base+swz<S128>(vr0+(noff+mkv)*16,vcb+md_local*2));
                        hmma16(pf[mkv],vf,of[md]);}}}
        }
        __syncthreads();
        if(t+1<nt)tld((t+1)*BN);
    }
    // Normalize
    {float dr[2];
        #pragma unroll
        for(int j=0;j<2;j++){if(ms[j]!=-INFINITY)asm volatile("rcp.approx.ftz.f32 %0,%1;":"=f"(dr[j]):"f"(ds[j]));else dr[j]=0.f;}
        #pragma unroll
        for(int md=0;md<NMDV;md++)
            #pragma unroll
            for(int ri=0;ri<8;ri++)of[md][ri]*=dr[(ri%4)/2];}
    // Write O
    {int g=lid/4,t2=lid%4;
        #pragma unroll
        for(int md=0;md<NMDV;md++){int db=md*16,qp=qs+wid*16+g;
            if(qp<S){long long off=bo+(long long)qp*H*D_V+(long long)head*D_V+db;
                *(bf16_2*)&O[off+2*t2]=__float22bfloat162_rn(make_float2(of[md][0],of[md][1]));
                *(bf16_2*)&O[off+2*t2+8]=__float22bfloat162_rn(make_float2(of[md][4],of[md][5]));}
            qp=qs+wid*16+g+8;
            if(qp<S){long long off=bo+(long long)qp*H*D_V+(long long)head*D_V+db;
                *(bf16_2*)&O[off+2*t2]=__float22bfloat162_rn(make_float2(of[md][2],of[md][3]));
                *(bf16_2*)&O[off+2*t2+8]=__float22bfloat162_rn(make_float2(of[md][6],of[md][7]));}}}
}

void launch_mla_prefill(const bf16*Qn,const bf16*Qp,const bf16*K,const bf16*V,
    bf16*O,int B,int S,int H,float sm,cudaStream_t st=0){
    float sml2=sm*1.44269504089f;
    CUtensorMap k_tma,v_tma;
    {uint64_t gD[3]={64,(uint64_t)S,(uint64_t)(D_QK/64)};
     uint64_t gS[2]={(uint64_t)(D_QK*sizeof(bf16)),64*sizeof(bf16)};
     uint32_t bD[3]={64,(uint32_t)BN,1}; uint32_t eS[3]={1,1,1};
     cuTensorMapEncodeTiled(&k_tma,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)K,gD,gS,bD,eS,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);}
    {uint64_t gD[3]={64,(uint64_t)S,(uint64_t)(D_V/64)};
     uint64_t gS[2]={(uint64_t)(D_V*sizeof(bf16)),64*sizeof(bf16)};
     uint32_t bD[3]={64,(uint32_t)BN,1}; uint32_t eS[3]={1,1,1};
     cuTensorMapEncodeTiled(&v_tma,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)V,gD,gS,bD,eS,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);}
    dim3 g(H,cdiv(S,BM),B);
    auto k=mla_tma_kernel;
    cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_SZ);
    k<<<g,NT,SMEM_SZ,st>>>(k_tma,v_tma,Qn,Qp,O,S,H,sml2);
}
bool load_bin(const char*p,void*d,size_t b){FILE*f=fopen(p,"rb");if(!f)return false;size_t r=fread(d,1,b,f);fclose(f);return r==b;}
int main(int argc,char**argv){
    cuInit(0);
    int B=1,H=16;int SL[]={1024,2048,4096};int nc=3;
    float sm=1.0f/sqrtf(192.0f);
    bool val=false;int vS=64;
    for(int i=1;i<argc;i++){if(strcmp(argv[i],"--validate")==0)val=true;
        else if(strcmp(argv[i],"--S")==0&&i+1<argc)vS=atoi(argv[++i]);}
    if(val){int S=vS;
        printf("=== Validation: B=%d S=%d H=%d ===\n",B,S,H);
        bf16*dQn,*dQp,*dK,*dV,*dO;
        size_t qns=(size_t)B*S*H*D_QK_NOPE*2,qps=(size_t)B*S*H*D_QK_ROPE*2;
        size_t ks=(size_t)B*S*D_QK*2,vs=(size_t)B*S*D_V*2,os=(size_t)B*S*H*D_V*2;
        cudaMalloc(&dQn,qns);cudaMalloc(&dQp,qps);cudaMalloc(&dK,ks);cudaMalloc(&dV,vs);cudaMalloc(&dO,os);
        bf16*hQn=(bf16*)malloc(qns),*hQp=(bf16*)malloc(qps),*hK=(bf16*)malloc(ks);
        bf16*hV=(bf16*)malloc(vs),*hO=(bf16*)malloc(os),*hRef=(bf16*)malloc(os);
        load_bin("/tmp/q_nope.bin",hQn,qns);load_bin("/tmp/q_pe.bin",hQp,qps);
        load_bin("/tmp/k.bin",hK,ks);load_bin("/tmp/v.bin",hV,vs);load_bin("/tmp/o_ref.bin",hRef,os);
        cudaMemcpy(dQn,hQn,qns,cudaMemcpyHostToDevice);cudaMemcpy(dQp,hQp,qps,cudaMemcpyHostToDevice);
        cudaMemcpy(dK,hK,ks,cudaMemcpyHostToDevice);cudaMemcpy(dV,hV,vs,cudaMemcpyHostToDevice);
        cudaMemset(dO,0,os);
        launch_mla_prefill(dQn,dQp,dK,dV,dO,B,S,H,sm);
        cudaDeviceSynchronize();
        cudaError_t err=cudaGetLastError();
        if(err!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(err));return 1;}
        cudaMemcpy(hO,dO,os,cudaMemcpyDeviceToHost);
        float me=0;for(size_t i=0;i<(size_t)B*S*H*D_V;i++){float a=__bfloat162float(hO[i]),b=__bfloat162float(hRef[i]);me=fmaxf(me,fabsf(a-b));}
        printf("max_abs_err=%.6f\n",me);
        printf("K O[0,0,0,:8]: ");for(int i=0;i<8;i++)printf("%.4f ",__bfloat162float(hO[i]));
        printf("\nR O[0,0,0,:8]: ");for(int i=0;i<8;i++)printf("%.4f ",__bfloat162float(hRef[i]));printf("\n");
        free(hQn);free(hQp);free(hK);free(hV);free(hO);free(hRef);
        cudaFree(dQn);cudaFree(dQp);cudaFree(dK);cudaFree(dV);cudaFree(dO);
        return(me<0.01f)?0:1;}
    printf("=== MLA Prefill TMA ===\n");
    int SM=4096;bf16*dQn,*dQp,*dK,*dV,*dO;
    size_t qns=(size_t)B*SM*H*D_QK_NOPE*2,qps=(size_t)B*SM*H*D_QK_ROPE*2;
    size_t ks=(size_t)B*SM*D_QK*2,vs=(size_t)B*SM*D_V*2,os=(size_t)B*SM*H*D_V*2;
    cudaMalloc(&dQn,qns);cudaMalloc(&dQp,qps);cudaMalloc(&dK,ks);cudaMalloc(&dV,vs);cudaMalloc(&dO,os);
    cudaMemset(dQn,1,qns);cudaMemset(dQp,1,qps);cudaMemset(dK,1,ks);cudaMemset(dV,1,vs);
    cudaEvent_t st,en;cudaEventCreate(&st);cudaEventCreate(&en);
    printf("KERNEL_RESULT {");
    for(int ci=0;ci<nc;ci++){int S=SL[ci];
        for(int i=0;i<20;i++)launch_mla_prefill(dQn,dQp,dK,dV,dO,B,S,H,sm);
        cudaDeviceSynchronize();
        cudaError_t err=cudaGetLastError();
        if(err!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(err));return 1;}
        int NI=100;std::vector<float>bt;
        for(int bi=0;bi<5;bi++){cudaEventRecord(st);
            for(int i=0;i<NI;i++)launch_mla_prefill(dQn,dQp,dK,dV,dO,B,S,H,sm);
            cudaEventRecord(en);cudaEventSynchronize(en);
            float ms;cudaEventElapsedTime(&ms,st,en);bt.push_back(ms/NI);}
        std::sort(bt.begin(),bt.end());float ms=bt[bt.size()/2];
        double fl=2.0*(double)B*H*S*S*(576+512);double tf=fl/(ms/1000.0)/1e12;
        printf("\"S%d\": %.2f",S,tf);if(ci<nc-1)printf(", ");
        fprintf(stderr,"S%d: %.2f TFLOPS, %.1f us\n",S,tf,ms*1000.0);}
    printf("}\n");
    cudaEventDestroy(st);cudaEventDestroy(en);
    cudaFree(dQn);cudaFree(dQp);cudaFree(dK);cudaFree(dV);cudaFree(dO);
    return 0;
}
