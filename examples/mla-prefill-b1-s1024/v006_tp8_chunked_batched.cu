// MLA Chunked Prefill — Batched version with 4D TMA for K/V
// Q covers positions [q_start, q_start + chunk_size), KV covers [0, kv_len)
// Causal: position q attends to kv positions <= q
// Batch dimension via blockIdx.z and 4D TMA
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
static constexpr int BN=128, MK=16;
static constexpr int HALF_N=BN/16/2; // 4
static constexpr int NMDK=D_QK_NOPE/MK; // 8
static constexpr int NMRK=D_QK_ROPE/MK; // 4
static constexpr int NMDV=D_V/16;        // 8
static constexpr int TMA_BLK = BN*64*2;

static constexpr int BM=64, NT=128;
static constexpr int Q_NOPE_SZ = BM*D_QK_NOPE*2;
static constexpr int Q_PE_SZ   = BM*D_QK_ROPE*2;
static constexpr int KN0_OFF = Q_NOPE_SZ + Q_PE_SZ;
static constexpr int KN1_OFF = KN0_OFF + TMA_BLK;
static constexpr int KP_OFF  = KN1_OFF + TMA_BLK;
static constexpr int V0_OFF  = KP_OFF + TMA_BLK;
static constexpr int V1_OFF  = V0_OFF + TMA_BLK;
// 2 mbarriers for K/V split pipeline
static constexpr int MBK_OFF = V1_OFF + TMA_BLK;
static constexpr int MBV_OFF = MBK_OFF + 16;
static constexpr int SMEM_SZ = MBV_OFF + 16;

// BM32 
static constexpr int BM32=32, NT32=64;
static constexpr int Q_NOPE_SZ32 = BM32*D_QK_NOPE*2;
static constexpr int Q_PE_SZ32   = BM32*D_QK_ROPE*2;
static constexpr int KN0_OFF32 = Q_NOPE_SZ32 + Q_PE_SZ32;
static constexpr int KN1_OFF32 = KN0_OFF32 + TMA_BLK;
static constexpr int KP_OFF32  = KN1_OFF32 + TMA_BLK;
static constexpr int V0_OFF32  = KP_OFF32 + TMA_BLK;
static constexpr int V1_OFF32  = V0_OFF32 + TMA_BLK;
static constexpr int MBK_OFF32 = V1_OFF32 + TMA_BLK;
static constexpr int MBV_OFF32 = MBK_OFF32 + 16;
static constexpr int SMEM_SZ32 = MBV_OFF32 + 16;
// BM32 double-buffered
static constexpr int KV_BUF_SZ = 5*TMA_BLK;
static constexpr int DB_KV0_OFF32 = Q_NOPE_SZ32 + Q_PE_SZ32;
static constexpr int DB_KV1_OFF32 = DB_KV0_OFF32 + KV_BUF_SZ;
static constexpr int DB_MBAR_OFF32= DB_KV1_OFF32 + KV_BUF_SZ;
static constexpr int DB_SMEM_SZ32 = DB_MBAR_OFF32 + 32;

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
__device__ __forceinline__ void cpa_commit(){asm volatile("cp.async.commit_group;\n");}
template<int N> __device__ __forceinline__ void cpa_wait(){asm volatile("cp.async.wait_group %0;\n"::"n"(N));}
__device__ __forceinline__ float sxor(float v,int m){return __shfl_xor_sync(0xffffffff,v,m);}
__device__ __forceinline__ float ex2(float x){float r;asm volatile("ex2.approx.ftz.f32 %0,%1;\n":"=f"(r):"f"(x));return r;}
__device__ __forceinline__ uint32_t f2b(float a,float b){bf16_2 v=__float22bfloat162_rn(make_float2(a,b));return *(uint32_t*)&v;}
__host__ __device__ __forceinline__ int cdiv(int a,int b){return(a+b-1)/b;}

__device__ __forceinline__ void tma3d(const CUtensorMap*d,int sa,int mb,int c0,int c1,int c2){
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4}],[%5];"
        ::"r"(sa),"l"((uint64_t)d),"r"(c0),"r"(c1),"r"(c2),"r"(mb):"memory");}

__device__ __forceinline__ void tma4d(const CUtensorMap*d,int sa,int mb,int c0,int c1,int c2,int c3){
    asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4,%5}],[%6];"
        ::"r"(sa),"l"((uint64_t)d),"r"(c0),"r"(c1),"r"(c2),"r"(c3),"r"(mb):"memory");}

__device__ __forceinline__ void mbar_init(int a,int c){asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(a),"r"(c));}
__device__ __forceinline__ void mbar_tx(int a,int b){asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(a),"r"(b));}
__device__ __forceinline__ void mbar_wait(int a,int p){
    asm volatile("{.reg .pred P;\nW: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P,[%0],%1,0x989680;\n@P bra D;\n bra W;\nD:}"::"r"(a),"r"(p));}

// Inline QK compute for one half  
__device__ __forceinline__ void do_qk_half(
    float sf[][8], int noff, int qpl, int qnl, int kps, int kn0, int kn1,
    int kr, int kc, int lid)
{
    constexpr int S128=128;
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
    #pragma unroll
    for(int mk=NMRK;mk<NMDK;mk++){
        uint32_t qr[4];ldm4(qr,qnl^(mk*32));
        #pragma unroll
        for(int nl=0;nl<HALF_N;nl++){uint32_t k2[4];
            ldm4(k2,(kn1l+(noff+nl)*16*S128)^((mk-4)*32));
            hmma16(qr,k2,sf[nl]);}
    }
}

// Mask + softmax + convert to bf16 + rowsum (no V SMEM access)
__device__ __forceinline__ void do_mask_softmax(
    float sf[][8], int noff, int kvb, int q_start, int qs,
    int kv_len, int wid, int lid, float sml2,
    float ms[2], float ds[2], float of[][8])
{
    if(kvb+BN>q_start+qs){int qrb=q_start+qs+wid*16;
        #pragma unroll
        for(int nl=0;nl<HALF_N;nl++){
            #pragma unroll
            for(int ri=0;ri<8;ri++){
                int rit=((ri&2)==0)?(lid/4):(lid/4+8);
                int kvc=2*(lid%4)+((ri&4)?8:0)+(ri&1);
                int qp=qrb+rit,kvp=kvb+(noff+nl)*16+kvc;
                if(!((kvp<=qp)&&(kvp<kv_len)))sf[nl][ri]=-INFINITY;}}}
    float mp[2]={ms[0],ms[1]};
    #pragma unroll
    for(int j=0;j<2;j++){
        #pragma unroll
        for(int nl=0;nl<HALF_N;nl++){float lm=fmaxf(fmaxf(sf[nl][j*2],sf[nl][j*2+1]),fmaxf(sf[nl][j*2+4],sf[nl][j*2+5]));ms[j]=fmaxf(ms[j],lm);}
        ms[j]=fmaxf(ms[j],sxor(ms[j],0x2));ms[j]=fmaxf(ms[j],sxor(ms[j],0x1));
        float nms=-(ms[j]*sml2);
        if(ms[j]!=mp[j]){float sc=ex2(__fmaf_rn(mp[j],sml2,nms));ds[j]*=sc;
        #pragma unroll
        for(int md=0;md<NMDV;md++){of[md][j*2+0]*=sc;of[md][j*2+1]*=sc;of[md][j*2+4]*=sc;of[md][j*2+5]*=sc;}}
        #pragma unroll
        for(int nl=0;nl<HALF_N;nl++){
            sf[nl][j*2+0]=ex2(__fmaf_rn(sf[nl][j*2+0],sml2,nms));sf[nl][j*2+1]=ex2(__fmaf_rn(sf[nl][j*2+1],sml2,nms));
            sf[nl][j*2+4]=ex2(__fmaf_rn(sf[nl][j*2+4],sml2,nms));sf[nl][j*2+5]=ex2(__fmaf_rn(sf[nl][j*2+5],sml2,nms));}}
}

// PV compute for one half (reads V SMEM)
__device__ __forceinline__ void do_pv_half(
    float sf[][8], int noff, int v0s, int v1s, int lid,
    float ds[2], float of[][8])
{
    constexpr int S128=128;
    uint32_t pf[HALF_N][4];
    #pragma unroll
    for(int nl=0;nl<HALF_N;nl++){
        #pragma unroll
        for(int i=0;i<4;i++)pf[nl][i]=f2b(sf[nl][i*2],sf[nl][i*2+1]);
        rowsum(ds,pf[nl]);}
    int vr0=lid%16,vcb=lid/16;
    #pragma unroll
    for(int mkv=0;mkv<HALF_N;mkv++){
        #pragma unroll
        for(int md=0;md<NMDV;md++){
            uint32_t vf[4];
            int vs_base=(md<4)?v0s:v1s;
            int md_local=(md<4)?md:(md-4);
            ldm4t(vf,vs_base+swz<S128>(vr0+(noff+mkv)*16,vcb+md_local*2));
            hmma16(pf[mkv],vf,of[md]);}}
}

// Write O to global memory
__device__ __forceinline__ void write_o(float of[][8], bf16* O, long long bo,
    int qs, int q_len, int H, int head, int wid, int lid)
{
    int g=lid/4,t2=lid%4;
    #pragma unroll
    for(int md=0;md<NMDV;md++){int db=md*16,qp=qs+wid*16+g;
        if(qp<q_len){long long off=bo+(long long)qp*H*D_V+(long long)head*D_V+db;
            *(bf16_2*)&O[off+2*t2]=__float22bfloat162_rn(make_float2(of[md][0],of[md][1]));
            *(bf16_2*)&O[off+2*t2+8]=__float22bfloat162_rn(make_float2(of[md][4],of[md][5]));}
        qp=qs+wid*16+g+8;
        if(qp<q_len){long long off=bo+(long long)qp*H*D_V+(long long)head*D_V+db;
            *(bf16_2*)&O[off+2*t2]=__float22bfloat162_rn(make_float2(of[md][2],of[md][3]));
            *(bf16_2*)&O[off+2*t2+8]=__float22bfloat162_rn(make_float2(of[md][6],of[md][7]));}}
}

__device__ __forceinline__ void finalize_o(float of[][8], float ms[2], float ds[2]){
    float dr[2];
    #pragma unroll
    for(int j=0;j<2;j++){if(ms[j]!=-INFINITY)asm volatile("rcp.approx.ftz.f32 %0,%1;":"=f"(dr[j]):"f"(ds[j]));else dr[j]=0.f;}
    #pragma unroll
    for(int md=0;md<NMDV;md++)
        #pragma unroll
        for(int ri=0;ri<8;ri++)of[md][ri]*=dr[(ri%4)/2];
}

// Persistent kernel: stride-based tile scheduler eliminates wave quantization waste
__global__ __launch_bounds__(NT,2)
void mla_persistent_kernel(
    const __grid_constant__ CUtensorMap k_tma,
    const __grid_constant__ CUtensorMap v_tma,
    const bf16*__restrict__ Qn,const bf16*__restrict__ Qp,
    bf16*__restrict__ O,const int q_len,const int kv_len,
    const int q_start,const int H,const float sml2,
    const int nqb, const int total_work)
{
    const int tid=threadIdx.x,wid=tid/32,lid=tid%32;
    extern __shared__ __align__(128) uint8_t sm[];
    int sb=__cvta_generic_to_shared(sm);
    int qn_s=sb,qp_s=sb+Q_NOPE_SZ;
    int kn0=sb+KN0_OFF,kn1=sb+KN1_OFF,kps=sb+KP_OFF,v0s=sb+V0_OFF,v1s=sb+V1_OFF;
    int mbk=sb+MBK_OFF, mbv=sb+MBV_OFF;
    const int stride = gridDim.x;
    const int kr=(lid%8)+(lid/16)*8,kc=(lid%16)/8;
    for(int work_id = blockIdx.x; work_id < total_work; work_id += stride) {
        int head = work_id % H;
        int qb_rev = (work_id / H) % nqb;
        int bat = work_id / (H * nqb);
        int qb = nqb - 1 - qb_rev;
        int qs = qb * BM;
        const long long bqn=(long long)bat*q_len*H*D_QK_NOPE,bqp=(long long)bat*q_len*H*D_QK_ROPE;
        const long long bo=(long long)bat*q_len*H*D_V;
        if(tid==0){mbar_init(mbk,1);mbar_init(mbv,1);asm volatile("fence.mbarrier_init.release.cluster;");}
        __syncthreads();
        {
            constexpr int SN=D_QK_NOPE*2,SP=D_QK_ROPE*2;
            for(int i=tid;i<BM*(D_QK_NOPE/8);i+=NT){int r=i/(D_QK_NOPE/8),c=i%(D_QK_NOPE/8),qi=qs+r;
                int a=qn_s+swz<SN>(r,c);
                if(qi<q_len)cpa(a,Qn+bqn+(long long)qi*H*D_QK_NOPE+(long long)head*D_QK_NOPE+c*8);
                else asm volatile("st.shared.v4.u32 [%0],{0,0,0,0};\n"::"r"(a));}
            for(int i=tid;i<BM*(D_QK_ROPE/8);i+=NT){int r=i/(D_QK_ROPE/8),c=i%(D_QK_ROPE/8),qi=qs+r;
                int a=qp_s+swz<SP>(r,c);
                if(qi<q_len)cpa(a,Qp+bqp+(long long)qi*H*D_QK_ROPE+(long long)head*D_QK_ROPE+c*8);
                else asm volatile("st.shared.v4.u32 [%0],{0,0,0,0};\n"::"r"(a));}
            cpa_commit();cpa_wait<0>();__syncthreads();
        }
        int qnl=qn_s+swz<(D_QK_NOPE*2)>(wid*16+(lid%16),lid/16);
        int qpl=qp_s+swz<(D_QK_ROPE*2)>(wid*16+(lid%16),lid/16);
        float of[NMDV][8];
        #pragma unroll
        for(int i=0;i<NMDV;i++)for(int j=0;j<8;j++)of[i][j]=0.f;
        float ms[2]={-INFINITY,-INFINITY},ds[2]={1.f,1.f};
        float sf0[HALF_N][8], sf1[HALF_N][8];
        int kvend=min(kv_len,q_start+qs+BM),nt=cdiv(kvend,BN);
        int mphk=0, mphv=0;
        auto tld_k=[&](int kvb){if(tid==0){
            mbar_tx(mbk,3*TMA_BLK);
            tma4d(&k_tma,kn0,mbk,0,kvb,0,bat); tma4d(&k_tma,kn1,mbk,0,kvb,1,bat);
            tma4d(&k_tma,kps,mbk,0,kvb,2,bat);
        }};
        auto tld_v=[&](int kvb){if(tid==0){
            mbar_tx(mbv,2*TMA_BLK);
            tma4d(&v_tma,v0s,mbv,0,kvb,0,bat); tma4d(&v_tma,v1s,mbv,0,kvb,1,bat);
        }};
        if(nt>0){tld_k(0);tld_v(0);}
        #pragma unroll 1
        for(int t=0;t<nt;t++){
            int kvb=t*BN;
            mbar_wait(mbk,mphk);mphk^=1;
            do_qk_half(sf0, 0, qpl, qnl, kps, kn0, kn1, kr, kc, lid);
            do_qk_half(sf1, HALF_N, qpl, qnl, kps, kn0, kn1, kr, kc, lid);
            __syncthreads();
            if(t+1<nt)tld_k((t+1)*BN);
            do_mask_softmax(sf0, 0, kvb, q_start, qs, kv_len, wid, lid, sml2, ms, ds, of);
            mbar_wait(mbv,mphv);mphv^=1;
            do_pv_half(sf0, 0, v0s, v1s, lid, ds, of);
            do_mask_softmax(sf1, HALF_N, kvb, q_start, qs, kv_len, wid, lid, sml2, ms, ds, of);
            do_pv_half(sf1, HALF_N, v0s, v1s, lid, ds, of);
            __syncthreads();
            if(t+1<nt)tld_v((t+1)*BN);
        }
        finalize_o(of, ms, ds);
        write_o(of, O, bo, qs, q_len, H, head, wid, lid);
        __syncthreads();
    }
}

// Main kernel with K/V split pipeline - 4D TMA for batched
__global__ __launch_bounds__(NT,2)
void mla_chunked_kernel(
    const __grid_constant__ CUtensorMap k_tma,
    const __grid_constant__ CUtensorMap v_tma,
    const bf16*__restrict__ Qn,const bf16*__restrict__ Qp,
    bf16*__restrict__ O,const int q_len,const int kv_len,
    const int q_start,const int H,const float sml2)
{
    const int head=blockIdx.x,qb=cdiv(q_len,BM)-1-blockIdx.y,qs=qb*BM;
    const int bat=blockIdx.z,tid=threadIdx.x,wid=tid/32,lid=tid%32;
    const long long bqn=(long long)bat*q_len*H*D_QK_NOPE,bqp=(long long)bat*q_len*H*D_QK_ROPE;
    const long long bo=(long long)bat*q_len*H*D_V;
    extern __shared__ __align__(128) uint8_t sm[];
    int sb=__cvta_generic_to_shared(sm);
    int qn_s=sb,qp_s=sb+Q_NOPE_SZ;
    int kn0=sb+KN0_OFF,kn1=sb+KN1_OFF,kps=sb+KP_OFF,v0s=sb+V0_OFF,v1s=sb+V1_OFF;
    int mbk=sb+MBK_OFF, mbv=sb+MBV_OFF;
    if(tid==0){mbar_init(mbk,1);mbar_init(mbv,1);asm volatile("fence.mbarrier_init.release.cluster;");}
    __syncthreads();
    {
        constexpr int SN=D_QK_NOPE*2,SP=D_QK_ROPE*2;
        for(int i=tid;i<BM*(D_QK_NOPE/8);i+=NT){int r=i/(D_QK_NOPE/8),c=i%(D_QK_NOPE/8),qi=qs+r;
            int a=qn_s+swz<SN>(r,c);
            if(qi<q_len)cpa(a,Qn+bqn+(long long)qi*H*D_QK_NOPE+(long long)head*D_QK_NOPE+c*8);
            else asm volatile("st.shared.v4.u32 [%0],{0,0,0,0};\n"::"r"(a));}
        for(int i=tid;i<BM*(D_QK_ROPE/8);i+=NT){int r=i/(D_QK_ROPE/8),c=i%(D_QK_ROPE/8),qi=qs+r;
            int a=qp_s+swz<SP>(r,c);
            if(qi<q_len)cpa(a,Qp+bqp+(long long)qi*H*D_QK_ROPE+(long long)head*D_QK_ROPE+c*8);
            else asm volatile("st.shared.v4.u32 [%0],{0,0,0,0};\n"::"r"(a));}
        cpa_commit();cpa_wait<0>();__syncthreads();
    }
    int qnl=qn_s+swz<(D_QK_NOPE*2)>(wid*16+(lid%16),lid/16);
    int qpl=qp_s+swz<(D_QK_ROPE*2)>(wid*16+(lid%16),lid/16);
    const int kr=(lid%8)+(lid/16)*8,kc=(lid%16)/8;

    float of[NMDV][8];
    #pragma unroll
    for(int i=0;i<NMDV;i++)for(int j=0;j<8;j++)of[i][j]=0.f;
    float ms[2]={-INFINITY,-INFINITY},ds[2]={1.f,1.f};
    float sf0[HALF_N][8], sf1[HALF_N][8];
    int kvend=min(kv_len,q_start+qs+BM),nt=cdiv(kvend,BN);
    int mphk=0, mphv=0;

    auto tld_k=[&](int kvb){if(tid==0){
        mbar_tx(mbk,3*TMA_BLK);
        tma4d(&k_tma,kn0,mbk,0,kvb,0,bat); tma4d(&k_tma,kn1,mbk,0,kvb,1,bat);
        tma4d(&k_tma,kps,mbk,0,kvb,2,bat);
    }};
    auto tld_v=[&](int kvb){if(tid==0){
        mbar_tx(mbv,2*TMA_BLK);
        tma4d(&v_tma,v0s,mbv,0,kvb,0,bat); tma4d(&v_tma,v1s,mbv,0,kvb,1,bat);
    }};
    if(nt>0){tld_k(0);tld_v(0);}

    #pragma unroll 1
    for(int t=0;t<nt;t++){
        int kvb=t*BN;
        // 1. Wait K, compute QK for both halves
        mbar_wait(mbk,mphk);mphk^=1;
        do_qk_half(sf0, 0, qpl, qnl, kps, kn0, kn1, kr, kc, lid);
        do_qk_half(sf1, HALF_N, qpl, qnl, kps, kn0, kn1, kr, kc, lid);
        // 2. K done, issue next K (overlaps with softmax + PV)
        __syncthreads();
        if(t+1<nt)tld_k((t+1)*BN);
        // 3. Mask + softmax for half0 (no SMEM needed, overlaps with K TMA)
        do_mask_softmax(sf0, 0, kvb, q_start, qs, kv_len, wid, lid, sml2, ms, ds, of);
        // 4. Wait V, compute PV for half0
        mbar_wait(mbv,mphv);mphv^=1;
        do_pv_half(sf0, 0, v0s, v1s, lid, ds, of);
        // 5. Mask + softmax + PV for half1 (V already in SMEM)
        do_mask_softmax(sf1, HALF_N, kvb, q_start, qs, kv_len, wid, lid, sml2, ms, ds, of);
        do_pv_half(sf1, HALF_N, v0s, v1s, lid, ds, of);
        // 6. V done, issue next V
        __syncthreads();
        if(t+1<nt)tld_v((t+1)*BN);
    }
    finalize_o(of, ms, ds);
    write_o(of, O, bo, qs, q_len, H, head, wid, lid);
}

// Interleaved kernel for high parallelism (BS16+): processes each half sequentially
// to reduce register pressure at the cost of K/V overlap
__global__ __launch_bounds__(NT,2)
void mla_chunked_kernel_interleaved(
    const __grid_constant__ CUtensorMap k_tma,
    const __grid_constant__ CUtensorMap v_tma,
    const bf16*__restrict__ Qn,const bf16*__restrict__ Qp,
    bf16*__restrict__ O,const int q_len,const int kv_len,
    const int q_start,const int H,const float sml2)
{
    const int head=blockIdx.x,qb=cdiv(q_len,BM)-1-blockIdx.y,qs=qb*BM;
    const int bat=blockIdx.z,tid=threadIdx.x,wid=tid/32,lid=tid%32;
    const long long bqn=(long long)bat*q_len*H*D_QK_NOPE,bqp=(long long)bat*q_len*H*D_QK_ROPE;
    const long long bo=(long long)bat*q_len*H*D_V;
    extern __shared__ __align__(128) uint8_t sm[];
    int sb=__cvta_generic_to_shared(sm);
    int qn_s=sb,qp_s=sb+Q_NOPE_SZ;
    int kn0=sb+KN0_OFF,kn1=sb+KN1_OFF,kps=sb+KP_OFF,v0s=sb+V0_OFF,v1s=sb+V1_OFF;
    int mbk=sb+MBK_OFF, mbv=sb+MBV_OFF;
    if(tid==0){mbar_init(mbk,1);mbar_init(mbv,1);asm volatile("fence.mbarrier_init.release.cluster;");}
    __syncthreads();
    {
        constexpr int SN=D_QK_NOPE*2,SP=D_QK_ROPE*2;
        for(int i=tid;i<BM*(D_QK_NOPE/8);i+=NT){int r=i/(D_QK_NOPE/8),c=i%(D_QK_NOPE/8),qi=qs+r;
            int a=qn_s+swz<SN>(r,c);
            if(qi<q_len)cpa(a,Qn+bqn+(long long)qi*H*D_QK_NOPE+(long long)head*D_QK_NOPE+c*8);
            else asm volatile("st.shared.v4.u32 [%0],{0,0,0,0};\n"::"r"(a));}
        for(int i=tid;i<BM*(D_QK_ROPE/8);i+=NT){int r=i/(D_QK_ROPE/8),c=i%(D_QK_ROPE/8),qi=qs+r;
            int a=qp_s+swz<SP>(r,c);
            if(qi<q_len)cpa(a,Qp+bqp+(long long)qi*H*D_QK_ROPE+(long long)head*D_QK_ROPE+c*8);
            else asm volatile("st.shared.v4.u32 [%0],{0,0,0,0};\n"::"r"(a));}
        cpa_commit();cpa_wait<0>();__syncthreads();
    }
    int qnl=qn_s+swz<(D_QK_NOPE*2)>(wid*16+(lid%16),lid/16);
    int qpl=qp_s+swz<(D_QK_ROPE*2)>(wid*16+(lid%16),lid/16);
    const int kr=(lid%8)+(lid/16)*8,kc=(lid%16)/8;
    float of[NMDV][8];
    #pragma unroll
    for(int i=0;i<NMDV;i++)for(int j=0;j<8;j++)of[i][j]=0.f;
    float ms[2]={-INFINITY,-INFINITY},ds[2]={1.f,1.f};
    float sf[HALF_N][8];
    int kvend=min(kv_len,q_start+qs+BM),nt=cdiv(kvend,BN);
    int mphk=0, mphv=0;
    auto tld_k=[&](int kvb){if(tid==0){
        mbar_tx(mbk,3*TMA_BLK);
        tma4d(&k_tma,kn0,mbk,0,kvb,0,bat); tma4d(&k_tma,kn1,mbk,0,kvb,1,bat);
        tma4d(&k_tma,kps,mbk,0,kvb,2,bat);
    }};
    auto tld_v=[&](int kvb){if(tid==0){
        mbar_tx(mbv,2*TMA_BLK);
        tma4d(&v_tma,v0s,mbv,0,kvb,0,bat); tma4d(&v_tma,v1s,mbv,0,kvb,1,bat);
    }};
    if(nt>0){tld_k(0);tld_v(0);}
    #pragma unroll 1
    for(int t=0;t<nt;t++){
        int kvb=t*BN;
        mbar_wait(mbk,mphk);mphk^=1;
        do_qk_half(sf, 0, qpl, qnl, kps, kn0, kn1, kr, kc, lid);
        do_mask_softmax(sf, 0, kvb, q_start, qs, kv_len, wid, lid, sml2, ms, ds, of);
        mbar_wait(mbv,mphv);mphv^=1;
        do_pv_half(sf, 0, v0s, v1s, lid, ds, of);
        do_qk_half(sf, HALF_N, qpl, qnl, kps, kn0, kn1, kr, kc, lid);
        __syncthreads();
        if(t+1<nt)tld_k((t+1)*BN);
        do_mask_softmax(sf, HALF_N, kvb, q_start, qs, kv_len, wid, lid, sml2, ms, ds, of);
        do_pv_half(sf, HALF_N, v0s, v1s, lid, ds, of);
        __syncthreads();
        if(t+1<nt)tld_v((t+1)*BN);
    }
    finalize_o(of, ms, ds);
    write_o(of, O, bo, qs, q_len, H, head, wid, lid);
}

// BM32 K/V split kernel - 4D TMA
__global__ __launch_bounds__(NT32,2)
void mla_chunked_kernel_bm32(
    const __grid_constant__ CUtensorMap k_tma,
    const __grid_constant__ CUtensorMap v_tma,
    const bf16*__restrict__ Qn,const bf16*__restrict__ Qp,
    bf16*__restrict__ O,const int q_len,const int kv_len,
    const int q_start,const int H,const float sml2)
{
    const int head=blockIdx.x,qb=cdiv(q_len,BM32)-1-blockIdx.y,qs=qb*BM32;
    const int bat=blockIdx.z,tid=threadIdx.x,wid=tid/32,lid=tid%32;
    const long long bqn=(long long)bat*q_len*H*D_QK_NOPE,bqp=(long long)bat*q_len*H*D_QK_ROPE;
    const long long bo=(long long)bat*q_len*H*D_V;
    extern __shared__ __align__(128) uint8_t sm[];
    int sb=__cvta_generic_to_shared(sm);
    int qn_s=sb,qp_s=sb+Q_NOPE_SZ32;
    int kn0=sb+KN0_OFF32,kn1=sb+KN1_OFF32,kps=sb+KP_OFF32,v0s=sb+V0_OFF32,v1s=sb+V1_OFF32;
    int mbk=sb+MBK_OFF32, mbv=sb+MBV_OFF32;
    if(tid==0){mbar_init(mbk,1);mbar_init(mbv,1);asm volatile("fence.mbarrier_init.release.cluster;");}
    __syncthreads();
    {
        constexpr int SN=D_QK_NOPE*2,SP=D_QK_ROPE*2;
        for(int i=tid;i<BM32*(D_QK_NOPE/8);i+=NT32){int r=i/(D_QK_NOPE/8),c=i%(D_QK_NOPE/8),qi=qs+r;
            int a=qn_s+swz<SN>(r,c);
            if(qi<q_len)cpa(a,Qn+bqn+(long long)qi*H*D_QK_NOPE+(long long)head*D_QK_NOPE+c*8);
            else asm volatile("st.shared.v4.u32 [%0],{0,0,0,0};\n"::"r"(a));}
        for(int i=tid;i<BM32*(D_QK_ROPE/8);i+=NT32){int r=i/(D_QK_ROPE/8),c=i%(D_QK_ROPE/8),qi=qs+r;
            int a=qp_s+swz<SP>(r,c);
            if(qi<q_len)cpa(a,Qp+bqp+(long long)qi*H*D_QK_ROPE+(long long)head*D_QK_ROPE+c*8);
            else asm volatile("st.shared.v4.u32 [%0],{0,0,0,0};\n"::"r"(a));}
        cpa_commit();cpa_wait<0>();__syncthreads();
    }
    int qnl=qn_s+swz<(D_QK_NOPE*2)>(wid*16+(lid%16),lid/16);
    int qpl=qp_s+swz<(D_QK_ROPE*2)>(wid*16+(lid%16),lid/16);
    const int kr=(lid%8)+(lid/16)*8,kc=(lid%16)/8;
    float of[NMDV][8];
    #pragma unroll
    for(int i=0;i<NMDV;i++)for(int j=0;j<8;j++)of[i][j]=0.f;
    float ms[2]={-INFINITY,-INFINITY},ds[2]={1.f,1.f};
    float sf[HALF_N][8];
    uint32_t pf0[HALF_N][4], pf1[HALF_N][4];
    int kvend=min(kv_len,q_start+qs+BM32),nt=cdiv(kvend,BN);
    int mphk=0, mphv=0;
    auto tld_k=[&](int kvb){if(tid==0){
        mbar_tx(mbk,3*TMA_BLK);
        tma4d(&k_tma,kn0,mbk,0,kvb,0,bat); tma4d(&k_tma,kn1,mbk,0,kvb,1,bat);
        tma4d(&k_tma,kps,mbk,0,kvb,2,bat);
    }};
    auto tld_v=[&](int kvb){if(tid==0){
        mbar_tx(mbv,2*TMA_BLK);
        tma4d(&v_tma,v0s,mbv,0,kvb,0,bat); tma4d(&v_tma,v1s,mbv,0,kvb,1,bat);
    }};
    if(nt>0){tld_k(0);tld_v(0);}
    #pragma unroll 1
    for(int t=0;t<nt;t++){
        int kvb=t*BN;
        mbar_wait(mbk,mphk);mphk^=1;
        // Half 0
        do_qk_half(sf, 0, qpl, qnl, kps, kn0, kn1, kr, kc, lid);
        do_mask_softmax(sf, 0, kvb, q_start, qs, kv_len, wid, lid, sml2, ms, ds, of);
        #pragma unroll
        for(int nl=0;nl<HALF_N;nl++){
            #pragma unroll
            for(int i=0;i<4;i++)pf0[nl][i]=f2b(sf[nl][i*2],sf[nl][i*2+1]);
            rowsum(ds,pf0[nl]);}
        // Half 1
        do_qk_half(sf, HALF_N, qpl, qnl, kps, kn0, kn1, kr, kc, lid);
        do_mask_softmax(sf, HALF_N, kvb, q_start, qs, kv_len, wid, lid, sml2, ms, ds, of);
        #pragma unroll
        for(int nl=0;nl<HALF_N;nl++){
            #pragma unroll
            for(int i=0;i<4;i++)pf1[nl][i]=f2b(sf[nl][i*2],sf[nl][i*2+1]);
            rowsum(ds,pf1[nl]);}
        // K done, issue next K
        __syncthreads();
        if(t+1<nt)tld_k((t+1)*BN);
        // Wait V, do PV
        mbar_wait(mbv,mphv);mphv^=1;
        {int vr0=lid%16,vcb=lid/16;
            constexpr int S128=128;
            #pragma unroll
            for(int mkv=0;mkv<HALF_N;mkv++){
                #pragma unroll
                for(int md=0;md<NMDV;md++){
                    uint32_t vf[4];
                    int vs_base=(md<4)?v0s:v1s;
                    int md_local=(md<4)?md:(md-4);
                    ldm4t(vf,vs_base+swz<S128>(vr0+(0*HALF_N+mkv)*16,vcb+md_local*2));
                    hmma16(pf0[mkv],vf,of[md]);}}
            #pragma unroll
            for(int mkv=0;mkv<HALF_N;mkv++){
                #pragma unroll
                for(int md=0;md<NMDV;md++){
                    uint32_t vf[4];
                    int vs_base=(md<4)?v0s:v1s;
                    int md_local=(md<4)?md:(md-4);
                    ldm4t(vf,vs_base+swz<S128>(vr0+(1*HALF_N+mkv)*16,vcb+md_local*2));
                    hmma16(pf1[mkv],vf,of[md]);}}}
        __syncthreads();
        if(t+1<nt)tld_v((t+1)*BN);
    }
    finalize_o(of, ms, ds);
    write_o(of, O, bo, qs, q_len, H, head, wid, lid);
}

// BM32 double-buffered TMA variant - 4D TMA
__global__ __launch_bounds__(NT32,1)
void mla_chunked_kernel_bm32_db(
    const __grid_constant__ CUtensorMap k_tma,
    const __grid_constant__ CUtensorMap v_tma,
    const bf16*__restrict__ Qn,const bf16*__restrict__ Qp,
    bf16*__restrict__ O,const int q_len,const int kv_len,
    const int q_start,const int H,const float sml2)
{
    const int head=blockIdx.x,qb=cdiv(q_len,BM32)-1-blockIdx.y,qs=qb*BM32;
    const int bat=blockIdx.z,tid=threadIdx.x,wid=tid/32,lid=tid%32;
    const long long bqn=(long long)bat*q_len*H*D_QK_NOPE,bqp=(long long)bat*q_len*H*D_QK_ROPE;
    const long long bo=(long long)bat*q_len*H*D_V;
    extern __shared__ __align__(128) uint8_t sm[];
    int sb=__cvta_generic_to_shared(sm);
    int qn_s=sb,qp_s=sb+Q_NOPE_SZ32;
    int kv_buf0=sb+DB_KV0_OFF32, kv_buf1=sb+DB_KV1_OFF32;
    int mbs0=sb+DB_MBAR_OFF32, mbs1=sb+DB_MBAR_OFF32+16;
    if(tid==0){mbar_init(mbs0,1);mbar_init(mbs1,1);asm volatile("fence.mbarrier_init.release.cluster;");}
    __syncthreads();
    {
        constexpr int SN=D_QK_NOPE*2,SP=D_QK_ROPE*2;
        for(int i=tid;i<BM32*(D_QK_NOPE/8);i+=NT32){int r=i/(D_QK_NOPE/8),c=i%(D_QK_NOPE/8),qi=qs+r;
            int a=qn_s+swz<SN>(r,c);
            if(qi<q_len)cpa(a,Qn+bqn+(long long)qi*H*D_QK_NOPE+(long long)head*D_QK_NOPE+c*8);
            else asm volatile("st.shared.v4.u32 [%0],{0,0,0,0};\n"::"r"(a));}
        for(int i=tid;i<BM32*(D_QK_ROPE/8);i+=NT32){int r=i/(D_QK_ROPE/8),c=i%(D_QK_ROPE/8),qi=qs+r;
            int a=qp_s+swz<SP>(r,c);
            if(qi<q_len)cpa(a,Qp+bqp+(long long)qi*H*D_QK_ROPE+(long long)head*D_QK_ROPE+c*8);
            else asm volatile("st.shared.v4.u32 [%0],{0,0,0,0};\n"::"r"(a));}
        cpa_commit();cpa_wait<0>();__syncthreads();
    }
    int qnl=qn_s+swz<(D_QK_NOPE*2)>(wid*16+(lid%16),lid/16);
    int qpl=qp_s+swz<(D_QK_ROPE*2)>(wid*16+(lid%16),lid/16);
    const int kr=(lid%8)+(lid/16)*8,kc=(lid%16)/8;
    float of[NMDV][8];
    #pragma unroll
    for(int i=0;i<NMDV;i++)for(int j=0;j<8;j++)of[i][j]=0.f;
    float ms[2]={-INFINITY,-INFINITY},ds[2]={1.f,1.f};
    float sf[HALF_N][8];
    int kvend=min(kv_len,q_start+qs+BM32),nt=cdiv(kvend,BN);
    int mph0=0,mph1=0;
    auto tld_b=[&](int kvb,int base,int mb){if(tid==0){
        mbar_tx(mb,5*TMA_BLK);
        tma4d(&k_tma,base+0*TMA_BLK,mb,0,kvb,0,bat);
        tma4d(&k_tma,base+1*TMA_BLK,mb,0,kvb,1,bat);
        tma4d(&k_tma,base+2*TMA_BLK,mb,0,kvb,2,bat);
        tma4d(&v_tma,base+3*TMA_BLK,mb,0,kvb,0,bat);
        tma4d(&v_tma,base+4*TMA_BLK,mb,0,kvb,1,bat);
    }};
    if(nt>0) tld_b(0,kv_buf0,mbs0);
    if(nt>1) tld_b(BN,kv_buf1,mbs1);
    #pragma unroll 1
    for(int t=0;t<nt;t++){
        int kvb=t*BN;
        int cur=(t&1);
        int cb=cur?kv_buf1:kv_buf0;
        int cm=cur?mbs1:mbs0;
        int *cph=cur?&mph1:&mph0;
        mbar_wait(cm,*cph);(*cph)^=1;
        if(t+2<nt) tld_b((t+2)*BN,cb,cm);
        int kn0c=cb,kn1c=cb+TMA_BLK,kpc=cb+2*TMA_BLK,v0c=cb+3*TMA_BLK,v1c=cb+4*TMA_BLK;
        #pragma unroll
        for(int half=0;half<2;half++){
            int noff=half*HALF_N;
            do_qk_half(sf, noff, qpl, qnl, kpc, kn0c, kn1c, kr, kc, lid);
            do_mask_softmax(sf, noff, kvb, q_start, qs, kv_len, wid, lid, sml2, ms, ds, of);
            do_pv_half(sf, noff, v0c, v1c, lid, ds, of);
        }
    }
    finalize_o(of, ms, ds);
    write_o(of, O, bo, qs, q_len, H, head, wid, lid);
}

// Split-K kernel with K/V split pipeline - 4D TMA
__global__ __launch_bounds__(NT,2)
void mla_splitk_kernel(
    const __grid_constant__ CUtensorMap k_tma,
    const __grid_constant__ CUtensorMap v_tma,
    const bf16*__restrict__ Qn,const bf16*__restrict__ Qp,
    float*__restrict__ partial,
    const int q_len,const int kv_len,
    const int q_start,const int H,const float sml2,
    const int num_splits,const int nqb)
{
    const int head=blockIdx.x;
    const int yidx=blockIdx.y;
    const int split_id=yidx%num_splits;
    const int qb_rev=yidx/num_splits;
    const int qb=nqb-1-qb_rev;
    const int qs=qb*BM;
    const int bat=blockIdx.z,tid=threadIdx.x,wid=tid/32,lid=tid%32;
    const long long bqn=(long long)bat*q_len*H*D_QK_NOPE,bqp=(long long)bat*q_len*H*D_QK_ROPE;
    extern __shared__ __align__(128) uint8_t sm[];
    int sb=__cvta_generic_to_shared(sm);
    int qn_s=sb,qp_s=sb+Q_NOPE_SZ;
    int kn0=sb+KN0_OFF,kn1=sb+KN1_OFF,kps=sb+KP_OFF,v0s=sb+V0_OFF,v1s=sb+V1_OFF;
    int mbk=sb+MBK_OFF, mbv=sb+MBV_OFF;
    if(tid==0){mbar_init(mbk,1);mbar_init(mbv,1);asm volatile("fence.mbarrier_init.release.cluster;");}
    __syncthreads();
    {
        constexpr int SN=D_QK_NOPE*2,SP=D_QK_ROPE*2;
        for(int i=tid;i<BM*(D_QK_NOPE/8);i+=NT){int r=i/(D_QK_NOPE/8),c=i%(D_QK_NOPE/8),qi=qs+r;
            int a=qn_s+swz<SN>(r,c);
            if(qi<q_len)cpa(a,Qn+bqn+(long long)qi*H*D_QK_NOPE+(long long)head*D_QK_NOPE+c*8);
            else asm volatile("st.shared.v4.u32 [%0],{0,0,0,0};\n"::"r"(a));}
        for(int i=tid;i<BM*(D_QK_ROPE/8);i+=NT){int r=i/(D_QK_ROPE/8),c=i%(D_QK_ROPE/8),qi=qs+r;
            int a=qp_s+swz<SP>(r,c);
            if(qi<q_len)cpa(a,Qp+bqp+(long long)qi*H*D_QK_ROPE+(long long)head*D_QK_ROPE+c*8);
            else asm volatile("st.shared.v4.u32 [%0],{0,0,0,0};\n"::"r"(a));}
        cpa_commit();cpa_wait<0>();__syncthreads();
    }
    int qnl=qn_s+swz<(D_QK_NOPE*2)>(wid*16+(lid%16),lid/16);
    int qpl=qp_s+swz<(D_QK_ROPE*2)>(wid*16+(lid%16),lid/16);
    const int kr=(lid%8)+(lid/16)*8,kc=(lid%16)/8;

    float of[NMDV][8];
    #pragma unroll
    for(int i=0;i<NMDV;i++)for(int j=0;j<8;j++)of[i][j]=0.f;
    float ms[2]={-INFINITY,-INFINITY},ds[2]={1.f,1.f};
    float sf0[HALF_N][8], sf1[HALF_N][8];
    
    int kvend=min(kv_len,q_start+qs+BM);
    int total_tiles=cdiv(kvend,BN);
    int tiles_per_split=cdiv(total_tiles,num_splits);
    int t_start=split_id*tiles_per_split;
    int t_end=min(t_start+tiles_per_split,total_tiles);
    int nt=t_end-t_start;
    int mphk=0, mphv=0;

    auto tld_k=[&](int kvb){if(tid==0){
        mbar_tx(mbk,3*TMA_BLK);
        tma4d(&k_tma,kn0,mbk,0,kvb,0,bat); tma4d(&k_tma,kn1,mbk,0,kvb,1,bat);
        tma4d(&k_tma,kps,mbk,0,kvb,2,bat);
    }};
    auto tld_v=[&](int kvb){if(tid==0){
        mbar_tx(mbv,2*TMA_BLK);
        tma4d(&v_tma,v0s,mbv,0,kvb,0,bat); tma4d(&v_tma,v1s,mbv,0,kvb,1,bat);
    }};
    if(nt>0){tld_k(t_start*BN);tld_v(t_start*BN);}

    #pragma unroll 1
    for(int t=t_start;t<t_end;t++){
        int kvb=t*BN;
        mbar_wait(mbk,mphk);mphk^=1;
        do_qk_half(sf0, 0, qpl, qnl, kps, kn0, kn1, kr, kc, lid);
        do_qk_half(sf1, HALF_N, qpl, qnl, kps, kn0, kn1, kr, kc, lid);
        __syncthreads();
        if(t+1<t_end)tld_k((t+1)*BN);
        do_mask_softmax(sf0, 0, kvb, q_start, qs, kv_len, wid, lid, sml2, ms, ds, of);
        mbar_wait(mbv,mphv);mphv^=1;
        do_pv_half(sf0, 0, v0s, v1s, lid, ds, of);
        do_mask_softmax(sf1, HALF_N, kvb, q_start, qs, kv_len, wid, lid, sml2, ms, ds, of);
        do_pv_half(sf1, HALF_N, v0s, v1s, lid, ds, of);
        __syncthreads();
        if(t+1<t_end)tld_v((t+1)*BN);
    }
    
    // Write partial results
    const long long stride_row = D_V + 4;
    const long long stride_head = BM * stride_row;
    const long long stride_qb = H * stride_head;
    const long long stride_split = (long long)nqb * stride_qb;
    const long long stride_bat = (long long)num_splits * stride_split;
    long long pbase = (long long)bat * stride_bat + (long long)split_id * stride_split + 
                      (long long)qb * stride_qb + (long long)head * stride_head;
    
    int g=lid/4,t2=lid%4;
    {
        int row = wid*16 + g;
        if(qs+row < q_len){
            long long roff = pbase + (long long)row * stride_row;
            #pragma unroll
            for(int md=0;md<NMDV;md++){
                int db=md*16;
                *(float2*)&partial[roff + db + 2*t2] = make_float2(of[md][0], of[md][1]);
                *(float2*)&partial[roff + db + 2*t2+8] = make_float2(of[md][4], of[md][5]);
            }
            if(t2==0){ partial[roff + D_V] = ms[0]; partial[roff + D_V+1] = ds[0]; }
        }
    }
    {
        int row = wid*16 + g + 8;
        if(qs+row < q_len){
            long long roff = pbase + (long long)row * stride_row;
            #pragma unroll
            for(int md=0;md<NMDV;md++){
                int db=md*16;
                *(float2*)&partial[roff + db + 2*t2] = make_float2(of[md][2], of[md][3]);
                *(float2*)&partial[roff + db + 2*t2+8] = make_float2(of[md][6], of[md][7]);
            }
            if(t2==0){ partial[roff + D_V] = ms[1]; partial[roff + D_V+1] = ds[1]; }
        }
    }
}

// Reduction kernel
__global__ __launch_bounds__(256)
void mla_reduce_kernel(
    const float*__restrict__ partial, bf16*__restrict__ O,
    const int q_len,const int H,const int num_splits,const int nqb,const float sm_scale)
{
    const int head=blockIdx.x, qb=blockIdx.y, bat=blockIdx.z;
    const int qs=qb*BM;
    const int row = threadIdx.x / 4;
    const int col_group = threadIdx.x % 4;
    
    const long long stride_row = D_V + 4;
    const long long stride_head = BM * stride_row;
    const long long stride_qb = H * stride_head;
    const long long stride_split = (long long)nqb * stride_qb;
    const long long stride_bat = (long long)num_splits * stride_split;
    
    if(qs+row >= q_len) return;
    
    float m_global = -INFINITY, d_global = 0.f;
    float o_local[32];
    #pragma unroll
    for(int d=0;d<32;d++) o_local[d] = 0.f;
    int d_start = col_group * 32;
    
    for(int s=0;s<num_splits;s++){
        long long roff = (long long)bat * stride_bat + (long long)s * stride_split + 
                        (long long)qb * stride_qb + (long long)head * stride_head +
                        (long long)row * stride_row;
        float m_s, d_s;
        if(col_group == 0){
            m_s = partial[roff + D_V];
            d_s = partial[roff + D_V + 1];
        }
        m_s = __shfl_sync(0xffffffff, m_s, (threadIdx.x & ~3));
        d_s = __shfl_sync(0xffffffff, d_s, (threadIdx.x & ~3));
        if(m_s == -INFINITY) continue;
        float vals[32];
        #pragma unroll
        for(int d=0;d<32;d+=4){
            float4 v = *(const float4*)&partial[roff + d_start + d];
            vals[d]=v.x; vals[d+1]=v.y; vals[d+2]=v.z; vals[d+3]=v.w;
        }
        if(m_s > m_global){
            float scale = expf((m_global - m_s) * sm_scale);
            d_global = d_global * scale + d_s;
            #pragma unroll
            for(int d=0;d<32;d++) o_local[d] = o_local[d] * scale + vals[d];
            m_global = m_s;
        } else {
            float scale = expf((m_s - m_global) * sm_scale);
            d_global += d_s * scale;
            #pragma unroll
            for(int d=0;d<32;d++) o_local[d] += vals[d] * scale;
        }
    }
    float dr = (m_global != -INFINITY) ? (1.0f / d_global) : 0.f;
    long long ooff = (long long)bat*q_len*H*D_V + (long long)(qs+row)*H*D_V + (long long)head*D_V;
    #pragma unroll
    for(int d=0;d<32;d+=4){
        bf16_2 lo = __float22bfloat162_rn(make_float2(o_local[d]*dr, o_local[d+1]*dr));
        bf16_2 hi = __float22bfloat162_rn(make_float2(o_local[d+2]*dr, o_local[d+3]*dr));
        int2 packed;
        packed.x = *(int*)&lo;
        packed.y = *(int*)&hi;
        *(int2*)&O[ooff+d_start+d] = packed;
    }
}

void make_tma_descs(CUtensorMap& k_tma, CUtensorMap& v_tma, const bf16*K, const bf16*V, int B, int kv_len){
    // 4D TMA for batched K: [64, kv_len, D_QK/64, B]
    {uint64_t gD[4]={64,(uint64_t)kv_len,(uint64_t)(D_QK/64),(uint64_t)B};
     uint64_t gS[3]={(uint64_t)(D_QK*sizeof(bf16)),64*sizeof(bf16),(uint64_t)kv_len*(uint64_t)D_QK*sizeof(bf16)};
     uint32_t bD[4]={64,(uint32_t)BN,1,1}; uint32_t eS[4]={1,1,1,1};
     cuTensorMapEncodeTiled(&k_tma,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,4,(void*)K,gD,gS,bD,eS,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);}
    // 4D TMA for batched V: [64, kv_len, D_V/64, B]
    {uint64_t gD[4]={64,(uint64_t)kv_len,(uint64_t)(D_V/64),(uint64_t)B};
     uint64_t gS[3]={(uint64_t)(D_V*sizeof(bf16)),64*sizeof(bf16),(uint64_t)kv_len*(uint64_t)D_V*sizeof(bf16)};
     uint32_t bD[4]={64,(uint32_t)BN,1,1}; uint32_t eS[4]={1,1,1,1};
     cuTensorMapEncodeTiled(&v_tma,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,4,(void*)V,gD,gS,bD,eS,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);}
}

void launch_mla_chunked_tma(const CUtensorMap& k_tma, const CUtensorMap& v_tma,
    const bf16*Qn,const bf16*Qp,
    bf16*O,float*partial_buf,int B,int q_len,int kv_len,int q_start,int H,float sm,int num_splits,cudaStream_t st=0){
    float sml2=sm*1.44269504089f;
    
    int nqb64 = cdiv(q_len, 64);
    int total64 = H * nqb64 * B;
    bool use_bm32 = (total64 < 148);
    
    CUtensorMap k_tma_copy = k_tma, v_tma_copy = v_tma;
    if(use_bm32 && num_splits <= 1){
        int nqb32 = cdiv(q_len, BM32);
        int total32 = H * nqb32 * B;
        dim3 g(H, nqb32, B);
        if(total32 <= 148){
            auto k = mla_chunked_kernel_bm32_db;
            cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, DB_SMEM_SZ32);
            k<<<g, NT32, DB_SMEM_SZ32, st>>>(k_tma_copy, v_tma_copy, Qn, Qp, O, q_len, kv_len, q_start, H, sml2);
        } else {
            auto k = mla_chunked_kernel_bm32;
            cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SZ32);
            k<<<g, NT32, SMEM_SZ32, st>>>(k_tma_copy, v_tma_copy, Qn, Qp, O, q_len, kv_len, q_start, H, sml2);
        }
        return;
    }

    int nqb=cdiv(q_len,BM);
    if(num_splits <= 1){
        int total_work = H * nqb * B;
        // Use persistent kernel for BS4-range (297-600 blocks) where wave tail is worst
        if(total_work > 296 && total_work <= 600) {
            int nblocks = min(total_work, 296);
            auto k=mla_persistent_kernel;
            cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_SZ);
            k<<<nblocks,NT,SMEM_SZ,st>>>(k_tma_copy,v_tma_copy,Qn,Qp,O,q_len,kv_len,q_start,H,sml2,nqb,total_work);
        } else if(total_work > 1500) {
            // Interleaved kernel for very high parallelism (BS16+)
            dim3 g(H,nqb,B);
            auto k=mla_chunked_kernel_interleaved;
            cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_SZ);
            k<<<g,NT,SMEM_SZ,st>>>(k_tma_copy,v_tma_copy,Qn,Qp,O,q_len,kv_len,q_start,H,sml2);
        } else {
            dim3 g(H,nqb,B);
            auto k=mla_chunked_kernel;
            cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_SZ);
            k<<<g,NT,SMEM_SZ,st>>>(k_tma_copy,v_tma_copy,Qn,Qp,O,q_len,kv_len,q_start,H,sml2);
        }
    } else {
        dim3 g(H,nqb*num_splits,B);
        auto k=mla_splitk_kernel;
        cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_SZ);
        k<<<g,NT,SMEM_SZ,st>>>(k_tma_copy,v_tma_copy,Qn,Qp,partial_buf,q_len,kv_len,q_start,H,sml2,num_splits,nqb);
        dim3 rg(H,nqb,B);
        mla_reduce_kernel<<<rg,256,0,st>>>(partial_buf,O,q_len,H,num_splits,nqb,sm);
    }
}

// Legacy wrapper that creates TMA descriptors per call (for validation)
void launch_mla_chunked(const bf16*Qn,const bf16*Qp,const bf16*K,const bf16*V,
    bf16*O,float*partial_buf,int B,int q_len,int kv_len,int q_start,int H,float sm,int num_splits,cudaStream_t st=0){
    CUtensorMap k_tma,v_tma;
    make_tma_descs(k_tma,v_tma,K,V,B,kv_len);
    launch_mla_chunked_tma(k_tma,v_tma,Qn,Qp,O,partial_buf,B,q_len,kv_len,q_start,H,sm,num_splits,st);
}

bool load_bin(const char*p,void*d,size_t b){FILE*f=fopen(p,"rb");if(!f)return false;size_t r=fread(d,1,b,f);fclose(f);return r==b;}

int main(int argc,char**argv){
    cuInit(0);
    int H=16;
    float sm=1.0f/sqrtf(192.0f);
    
    struct Config { const char* name; int batch; int chunk; int kv_len; };
    Config configs[] = {
        {"BS1_C512_KV4096",  1,  512, 4096},
        {"BS4_C512_KV4096",  4,  512, 4096},
        {"BS8_C512_KV4096",  8,  512, 4096},
        {"BS16_C512_KV4096", 16, 512, 4096},
    };
    int nc = 4;
    
    bool val = false;
    int vBatch = 1, vChunk = 64, vKV = 128;
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"--validate")==0) val=true;
        else if(strcmp(argv[i],"--batch")==0&&i+1<argc) vBatch=atoi(argv[++i]);
        else if(strcmp(argv[i],"--chunk")==0&&i+1<argc) vChunk=atoi(argv[++i]);
        else if(strcmp(argv[i],"--kv")==0&&i+1<argc) vKV=atoi(argv[++i]);
    }
    
    auto compute_splits = [&](int B, int chunk, int kv_len) -> int {
        int nqb64 = cdiv(chunk, 64);
        int total64 = H * nqb64 * B;
        if(total64 >= 148) return 1;
        int nqb32 = cdiv(chunk, BM32);
        int total32 = H * nqb32 * B;
        if(total32 >= 296) return 1;
        int avg_tiles = cdiv(kv_len, BN);
        if(avg_tiles < 12) return 1;
        if(total32 >= 148 && avg_tiles <= 48 && total64 >= 148) return 1;
        int nqb_sk = cdiv(chunk, 64);
        int total_sk = H * nqb_sk * B;
        int splits = cdiv(256, total_sk);
        if(splits > 4) splits = 4;
        if(splits < 1) splits = 1;
        return splits;
    };
    
    if(val){
        int B=vBatch, q_len=vChunk, kv_len=vKV, q_start=kv_len-q_len;
        int nsplits = compute_splits(B, q_len, kv_len);
        printf("=== Validation: B=%d q_len=%d kv_len=%d q_start=%d H=%d splits=%d ===\n",B,q_len,kv_len,q_start,H,nsplits);
        bf16*dQn,*dQp,*dK,*dV,*dO;
        float*dPartial;
        size_t qns=(size_t)B*q_len*H*D_QK_NOPE*2,qps=(size_t)B*q_len*H*D_QK_ROPE*2;
        size_t ks=(size_t)B*kv_len*D_QK*2,vs=(size_t)B*kv_len*D_V*2,os=(size_t)B*q_len*H*D_V*2;
        int nqb=cdiv(q_len,BM);
        size_t ps=(size_t)nsplits*B*nqb*H*BM*(D_V+4)*sizeof(float);
        cudaMalloc(&dQn,qns);cudaMalloc(&dQp,qps);cudaMalloc(&dK,ks);cudaMalloc(&dV,vs);cudaMalloc(&dO,os);
        cudaMalloc(&dPartial,ps);
        
        // Generate random data
        bf16*hQn=(bf16*)malloc(qns),*hQp=(bf16*)malloc(qps),*hK=(bf16*)malloc(ks);
        bf16*hV=(bf16*)malloc(vs),*hO=(bf16*)malloc(os);
        srand(42);
        auto rbf=[](){ return __float2bfloat16((float)(rand()%2001-1000)/1000.0f); };
        for(size_t i=0;i<(size_t)B*q_len*H*D_QK_NOPE;i++) hQn[i]=rbf();
        for(size_t i=0;i<(size_t)B*q_len*H*D_QK_ROPE;i++) hQp[i]=rbf();
        for(size_t i=0;i<(size_t)B*kv_len*D_QK;i++) hK[i]=rbf();
        for(size_t i=0;i<(size_t)B*kv_len*D_V;i++) hV[i]=rbf();
        
        cudaMemcpy(dQn,hQn,qns,cudaMemcpyHostToDevice);cudaMemcpy(dQp,hQp,qps,cudaMemcpyHostToDevice);
        cudaMemcpy(dK,hK,ks,cudaMemcpyHostToDevice);cudaMemcpy(dV,hV,vs,cudaMemcpyHostToDevice);
        cudaMemset(dO,0,os);
        launch_mla_chunked(dQn,dQp,dK,dV,dO,dPartial,B,q_len,kv_len,q_start,H,sm,nsplits);
        cudaDeviceSynchronize();
        cudaError_t err=cudaGetLastError();
        if(err!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(err));return 1;}
        cudaMemcpy(hO,dO,os,cudaMemcpyDeviceToHost);
        
        // CPU reference
        float* ref = (float*)malloc((size_t)B*q_len*H*D_V*sizeof(float));
        for(int b=0;b<B;b++){
            for(int qi=0;qi<q_len;qi++){
                for(int h=0;h<H;h++){
                    float maxs=-INFINITY;
                    int qpos = q_start + qi;
                    // compute QK
                    std::vector<float> scores(kv_len);
                    for(int ki=0;ki<kv_len;ki++){
                        if(ki>qpos){scores[ki]=-INFINITY;continue;}
                        float s=0;
                        // NOPE part
                        for(int d=0;d<D_QK_NOPE;d++){
                            float qv=__bfloat162float(hQn[(size_t)b*q_len*H*D_QK_NOPE+(size_t)qi*H*D_QK_NOPE+(size_t)h*D_QK_NOPE+d]);
                            float kv=__bfloat162float(hK[(size_t)b*kv_len*D_QK+(size_t)ki*D_QK+d]);
                            s+=qv*kv;
                        }
                        // ROPE part
                        for(int d=0;d<D_QK_ROPE;d++){
                            float qv=__bfloat162float(hQp[(size_t)b*q_len*H*D_QK_ROPE+(size_t)qi*H*D_QK_ROPE+(size_t)h*D_QK_ROPE+d]);
                            float kv=__bfloat162float(hK[(size_t)b*kv_len*D_QK+(size_t)ki*D_QK+D_QK_NOPE+d]);
                            s+=qv*kv;
                        }
                        scores[ki]=s*sm;
                        maxs=fmaxf(maxs,scores[ki]);
                    }
                    float sumexp=0;
                    for(int ki=0;ki<=qpos&&ki<kv_len;ki++){
                        scores[ki]=expf(scores[ki]-maxs);
                        sumexp+=scores[ki];
                    }
                    for(int d=0;d<D_V;d++){
                        float o=0;
                        for(int ki=0;ki<=qpos&&ki<kv_len;ki++){
                            float vv=__bfloat162float(hV[(size_t)b*kv_len*D_V+(size_t)ki*D_V+d]);
                            o+=scores[ki]*vv;
                        }
                        ref[(size_t)b*q_len*H*D_V+(size_t)qi*H*D_V+(size_t)h*D_V+d]=o/sumexp;
                    }
                }
            }
        }
        
        float me=0,mae=0;int worst_i=0,worst_ai=0;
        for(size_t i=0;i<(size_t)B*q_len*H*D_V;i++){
            float a=__bfloat162float(hO[i]),b=ref[i];
            float denom=fmaxf(fmaxf(fabsf(b),fabsf(a)),1e-3f);
            float e=fabsf(a-b)/denom;
            float ae=fabsf(a-b);
            if(e>me){me=e;worst_i=(int)i;}
            if(ae>mae){mae=ae;worst_ai=(int)i;}
        }
        printf("max_rel_err=%.6f at index %d\n",me,worst_i);
        printf("max_abs_err=%.6f at index %d\n",mae,worst_ai);
        if(worst_i>=0){
            printf("kernel=%.6f ref=%.6f\n",__bfloat162float(hO[worst_i]),ref[worst_i]);
        }
        printf("K O[:8]: ");for(int i=0;i<8;i++)printf("%.4f ",__bfloat162float(hO[i]));
        printf("\nR O[:8]: ");for(int i=0;i<8;i++)printf("%.4f ",ref[i]);printf("\n");
        free(hQn);free(hQp);free(hK);free(hV);free(hO);free(ref);
        cudaFree(dQn);cudaFree(dQp);cudaFree(dK);cudaFree(dV);cudaFree(dO);cudaFree(dPartial);
        return(me<5e-3)?0:1;
    }
    
    printf("=== MLA Chunked Prefill Batched ===\n");
    int maxKV=4096,maxChunk=512,maxB=16;
    bf16*dQn,*dQp,*dK,*dV,*dO;
    float*dPartial;
    size_t qns=(size_t)maxB*maxChunk*H*D_QK_NOPE*2,qps=(size_t)maxB*maxChunk*H*D_QK_ROPE*2;
    size_t ks=(size_t)maxB*maxKV*D_QK*2,vs=(size_t)maxB*maxKV*D_V*2,os=(size_t)maxB*maxChunk*H*D_V*2;
    int max_nqb=cdiv(maxChunk,BM);
    size_t ps=(size_t)16*maxB*max_nqb*H*BM*(D_V+4)*sizeof(float);
    cudaMalloc(&dQn,qns);cudaMalloc(&dQp,qps);cudaMalloc(&dK,ks);cudaMalloc(&dV,vs);cudaMalloc(&dO,os);
    cudaMalloc(&dPartial,ps);
    cudaMemset(dQn,1,qns);cudaMemset(dQp,1,qps);cudaMemset(dK,1,ks);cudaMemset(dV,1,vs);

    size_t flushSz = 192*1024*1024;
    char* dFlush; cudaMalloc(&dFlush, flushSz);
    char* hFlush = (char*)malloc(flushSz);
    
    cudaEvent_t est,een;cudaEventCreate(&est);cudaEventCreate(&een);
    
    printf("KERNEL_RESULT {");
    for(int ci=0;ci<nc;ci++){
        int B=configs[ci].batch, chunk=configs[ci].chunk, kv_len=configs[ci].kv_len;
        int q_start = kv_len - chunk;
        int nsplits = compute_splits(B, chunk, kv_len);
        fprintf(stderr,"%s: B=%d splits=%d\n",configs[ci].name,B,nsplits);
        
        CUtensorMap k_tma,v_tma;
        make_tma_descs(k_tma,v_tma,dK,dV,B,kv_len);
        for(int i=0;i<30;i++) launch_mla_chunked_tma(k_tma,v_tma,dQn,dQp,dO,dPartial,B,chunk,kv_len,q_start,H,sm,nsplits);
        cudaDeviceSynchronize();
        cudaError_t err=cudaGetLastError();
        if(err!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(err));return 1;}
        
        int NI=200;
        std::vector<float> times;
        for(int bi=0;bi<7;bi++){
            cudaMemcpy(hFlush, dFlush, flushSz, cudaMemcpyDeviceToHost);
            cudaEventRecord(est);
            for(int i=0;i<NI;i++) launch_mla_chunked_tma(k_tma,v_tma,dQn,dQp,dO,dPartial,B,chunk,kv_len,q_start,H,sm,nsplits);
            cudaEventRecord(een);cudaEventSynchronize(een);
            float ms;cudaEventElapsedTime(&ms,est,een);times.push_back(ms/NI);
        }
        std::sort(times.begin(),times.end());float ms=times[times.size()/2];
        double fl=2.0*(double)B*H*chunk*kv_len*(576+512);
        double tf=fl/(ms/1000.0)/1e12;
        printf("\"%s\": %.2f",configs[ci].name,tf);
        if(ci<nc-1) printf(", ");
        fprintf(stderr,"%s: %.2f TFLOPS, %.1f us\n",configs[ci].name,tf,ms*1000.0);
    }
    printf("}\n");
    
    cudaEventDestroy(est);cudaEventDestroy(een);
    cudaFree(dQn);cudaFree(dQp);cudaFree(dK);cudaFree(dV);cudaFree(dO);
    cudaFree(dPartial);cudaFree(dFlush);free(hFlush);
    return 0;
}
