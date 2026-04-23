v028

// MLA Prefill Attention Kernel — combined mma.sync + tcgen05 with TMA
// mma.sync for S<=1024, tcgen05 with TMA-based CKV loading for S>1024
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <vector>

using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;

static constexpr int D_CKV = 512;
static constexpr int D_KPE = 64;
static constexpr int D_V   = 512;
static constexpr int BN    = 64;
static constexpr int WARP_SIZE = 32;
static constexpr int MMA_K = 16;
static constexpr int NUM_MMA_KV = BN / MMA_K;
static constexpr int NUM_MMA_N16 = BN / 16;
static constexpr int NUM_MMA_D_CKV_K = D_CKV / MMA_K;
static constexpr int NUM_MMA_D_KPE_K = D_KPE / MMA_K;

template <int STRIDE>
__device__ __forceinline__ int swizzle(int row, int col) {
    if constexpr (STRIDE >= 128)
        col ^= (row % 8) / (128 / STRIDE > 1 ? 128 / STRIDE : 1);
    return row * STRIDE + col * 16;
}

// PTX Intrinsics
__device__ __forceinline__ void ldmatrix_x4(uint32_t reg[4], int addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(reg[0]),"=r"(reg[1]),"=r"(reg[2]),"=r"(reg[3]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t reg[4], int addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(reg[0]),"=r"(reg[1]),"=r"(reg[2]),"=r"(reg[3]) : "r"(addr));
}
__device__ __forceinline__ void mma_m16n8k16_bf16(const uint32_t A[4], const uint32_t B[2], float C[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(C[0]),"+f"(C[1]),"+f"(C[2]),"+f"(C[3])
        : "r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),"r"(B[0]),"r"(B[1]));
}
__device__ __forceinline__ void mma_m16n8k16_bf16_init(const uint32_t A[4], const uint32_t B[2], float C[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(C[0]),"=f"(C[1]),"=f"(C[2]),"=f"(C[3])
        : "r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),"r"(B[0]),"r"(B[1]),
          "f"(0.f),"f"(0.f),"f"(0.f),"f"(0.f));
}
__device__ __forceinline__ void mma_m16n16k16_bf16(const uint32_t A[4], const uint32_t B[4], float C[8]) {
    mma_m16n8k16_bf16(A, &B[0], &C[0]); mma_m16n8k16_bf16(A, &B[2], &C[4]);
}
__device__ __forceinline__ void mma_m16n16k16_bf16_init(const uint32_t A[4], const uint32_t B[4], float C[8]) {
    mma_m16n8k16_bf16_init(A, &B[0], &C[0]); mma_m16n8k16_bf16_init(A, &B[2], &C[4]);
}
__device__ __forceinline__ void mma_rowsum_bf16(float* d, uint32_t* s_u32) {
    asm volatile("{mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,_,%1,_},{%2,%3,%4,%5},{%6,%7},{%0,0.,%1,0.};}\n"
        : "+f"(d[0]),"+f"(d[1])
        : "r"(s_u32[0]),"r"(s_u32[1]),"r"(s_u32[2]),"r"(s_u32[3]),
          "r"(1065369472u),"r"(1065369472u));
}
__device__ __forceinline__ void cp_async_128b(int dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_128b_pred(int dst, const void* src, bool pred) {
    asm volatile("{.reg .pred p; setp.ne.b32 p, %2, 0; @!p st.shared.v4.u32 [%0], {0,0,0,0}; @p cp.async.cg.shared.global [%0], [%1], 16;}\n"
        :: "r"(dst), "l"(src), "r"((int)pred));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ float shfl_xor(float val, int mask) { return __shfl_xor_sync(0xffffffff, val, mask); }
__device__ __forceinline__ float fast_exp2f(float x) { float r; asm volatile("ex2.approx.ftz.f32 %0,%1;\n":"=f"(r):"f"(x)); return r; }
__device__ __forceinline__ uint32_t float2_to_bf16x2(float a, float b) {
    bf16_2 v = __float22bfloat162_rn(make_float2(a, b)); return *reinterpret_cast<uint32_t*>(&v);
}
__host__ __device__ __forceinline__ int cdiv(int a, int b) { return (a + b - 1) / b; }

// TMA load helper
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* desc, int smem_addr, int mbar_addr, int coord0, int coord1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_addr), "l"((uint64_t)desc), "r"(coord0), "r"(coord1), "r"(mbar_addr) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(int mbar_addr, int tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mbar_addr), "r"(tx_bytes));
}

// ============================================================
// mma.sync kernel (for S<=1024) — unchanged from v016
// ============================================================
template <int BM_T, int WARP_D_T, bool REVERSE_BLOCKS>
__global__ __launch_bounds__(256, (BM_T <= 32 ? 2 : 1))
void mla_prefill_mmasync_kernel(
    const bf16* __restrict__ Q_nope, const bf16* __restrict__ Q_pe,
    const bf16* __restrict__ CKV, const bf16* __restrict__ KPE,
    bf16* __restrict__ O,
    const int S, const int H, const float sm_scale_log2)
{
    [[maybe_unused]] static constexpr int WARP_M_T = 8 / WARP_D_T;
    static constexpr int D_V_PER_WARP = D_V / WARP_D_T;
    static constexpr int NUM_MMA_D_V_WARP = D_V_PER_WARP / 16;
    static constexpr int NUM_N_SHARD = NUM_MMA_N16 / WARP_D_T;
    static constexpr int NUM_STAGES = 2;
    static constexpr int Q_NOPE_SIZE = BM_T * D_CKV * sizeof(bf16);
    static constexpr int Q_PE_SIZE   = BM_T * D_KPE * sizeof(bf16);
    static constexpr int CKV_STAGE_SIZE = BN * D_CKV * sizeof(bf16);
    static constexpr int KPE_STAGE_SIZE = BN * D_KPE * sizeof(bf16);
    const int head = blockIdx.x;
    const int num_q_blocks_total = cdiv(S, BM_T);
    const int q_block = REVERSE_BLOCKS ? (num_q_blocks_total - 1 - blockIdx.y) : blockIdx.y;
    const int q_start = q_block * BM_T;
    const int batch = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_m = warp_id / WARP_D_T;
    const int warp_d = warp_id % WARP_D_T;
    const long long bqno = (long long)batch*S*H*D_CKV;
    const long long bqpo = (long long)batch*S*H*D_KPE;
    const long long bcko = (long long)batch*S*D_CKV;
    const long long bkpo = (long long)batch*S*D_KPE;
    const long long boo  = (long long)batch*S*H*D_V;
    extern __shared__ __align__(128) uint8_t smem_raw[];
    int smem_base = static_cast<int>(__cvta_generic_to_shared(smem_raw));
    int q_nope_smem = smem_base;
    int q_pe_smem = q_nope_smem + Q_NOPE_SIZE;
    int ckv_smem_base = q_pe_smem + Q_PE_SIZE;
    int kpe_smem_base = ckv_smem_base + NUM_STAGES * CKV_STAGE_SIZE;
    auto ckv_smem = [&](int s) { return ckv_smem_base + s * CKV_STAGE_SIZE; };
    auto kpe_smem = [&](int s) { return kpe_smem_base + s * KPE_STAGE_SIZE; };
    constexpr int M_WG_OFF = Q_NOPE_SIZE+Q_PE_SIZE+NUM_STAGES*(CKV_STAGE_SIZE+KPE_STAGE_SIZE);
    float* m_wg = reinterpret_cast<float*>(smem_raw + M_WG_OFF);
    {
        constexpr int QNL = (BM_T*D_CKV)/8, QPL = (BM_T*D_KPE)/8;
        constexpr int SNO = D_CKV*sizeof(bf16), SPO = D_KPE*sizeof(bf16);
        for (int i = tid; i < QNL; i += 256) {
            int r=i/(D_CKV/8), c=i%(D_CKV/8), qi=q_start+r;
            int sa = q_nope_smem + swizzle<SNO>(r,c);
            const bf16* gp = Q_nope + bqno + (long long)qi*H*D_CKV + (long long)head*D_CKV + c*8;
            if(qi<S) cp_async_128b(sa,gp); else asm volatile("st.shared.v4.u32 [%0],{0,0,0,0};\n"::"r"(sa));
        }
        for (int i = tid; i < QPL; i += 256) {
            int r=i/(D_KPE/8), c=i%(D_KPE/8), qi=q_start+r;
            int sa = q_pe_smem + swizzle<SPO>(r,c);
            const bf16* gp = Q_pe + bqpo + (long long)qi*H*D_KPE + (long long)head*D_KPE + c*8;
            if(qi<S) cp_async_128b(sa,gp); else asm volatile("st.shared.v4.u32 [%0],{0,0,0,0};\n"::"r"(sa));
        }
        cp_async_commit(); cp_async_wait<0>(); __syncthreads();
    }
    constexpr int STRIDE_NOPE_B = D_CKV*sizeof(bf16), STRIDE_PE_B = D_KPE*sizeof(bf16);
    int q_nope_ldm_base = q_nope_smem + swizzle<STRIDE_NOPE_B>(warp_m*16+(lane_id%16), lane_id/16);
    int q_pe_ldm_base = q_pe_smem + swizzle<STRIDE_PE_B>(warp_m*16+(lane_id%16), lane_id/16);
    float o_frag[NUM_MMA_D_V_WARP][8];
    #pragma unroll
    for(int i=0;i<NUM_MMA_D_V_WARP;i++) for(int j=0;j<8;j++) o_frag[i][j]=0.f;
    float m_state[2]={-INFINITY,-INFINITY}, d_state[2]={1.f,1.f};
    float s_frag[NUM_N_SHARD][8];
    int kv_end = min(S, q_start+BM_T);
    int num_kv_tiles = cdiv(kv_end, BN);
    auto load_kv_tile = [&](int kv_tile, int stage) {
        int kv_base = kv_tile*BN;
        constexpr int CL=(BN*D_CKV)/8, KL2=(BN*D_KPE)/8;
        constexpr int SC=D_CKV*sizeof(bf16), SK=D_KPE*sizeof(bf16);
        if(kv_base+BN<=S) {
            for(int i=tid;i<CL;i+=256){int r=i/(D_CKV/8),c=i%(D_CKV/8);
                cp_async_128b(ckv_smem(stage)+swizzle<SC>(r,c), CKV+bcko+(long long)(kv_base+r)*D_CKV+c*8);}
            for(int i=tid;i<KL2;i+=256){int r=i/(D_KPE/8),c=i%(D_KPE/8);
                cp_async_128b(kpe_smem(stage)+swizzle<SK>(r,c), KPE+bkpo+(long long)(kv_base+r)*D_KPE+c*8);}
        } else {
            for(int i=tid;i<CL;i+=256){int r=i/(D_CKV/8),c=i%(D_CKV/8);int ki=kv_base+r;
                cp_async_128b_pred(ckv_smem(stage)+swizzle<SC>(r,c), CKV+bcko+(long long)ki*D_CKV+c*8, ki<S);}
            for(int i=tid;i<KL2;i+=256){int r=i/(D_KPE/8),c=i%(D_KPE/8);int ki=kv_base+r;
                cp_async_128b_pred(kpe_smem(stage)+swizzle<SK>(r,c), KPE+bkpo+(long long)ki*D_KPE+c*8, ki<S);}
        }
        cp_async_commit();
    };
    if(num_kv_tiles>0) load_kv_tile(0,0);
    constexpr int STRIDE_CKV_B = D_CKV*sizeof(bf16), STRIDE_KPE_B = D_KPE*sizeof(bf16);
    const int k_row = (lane_id%8)+(lane_id/16)*8;
    const int k_col_base = (lane_id%16)/8;
    const int n_offset = warp_d*NUM_N_SHARD;
    constexpr int BARRIER_THREADS = WARP_D_T*WARP_SIZE;
    #pragma unroll 1
    for(int kv_tile=0; kv_tile<num_kv_tiles; kv_tile++) {
        int stage = kv_tile%NUM_STAGES, kv_base = kv_tile*BN;
        cp_async_wait<0>(); __syncthreads();
        if(kv_tile+1<num_kv_tiles) load_kv_tile(kv_tile+1,(kv_tile+1)%NUM_STAGES);
        {
            int kpe_ldm = kpe_smem(stage)+swizzle<STRIDE_KPE_B>(k_row,k_col_base);
            int ckv_ldm = ckv_smem(stage)+swizzle<STRIDE_CKV_B>(k_row,k_col_base);
            #pragma unroll
            for(int mk=0;mk<NUM_MMA_D_KPE_K;mk++){
                {uint32_t qr[4]; ldmatrix_x4(qr, q_pe_ldm_base^(mk*32));
                 #pragma unroll
                 for(int nl=0;nl<NUM_N_SHARD;nl++){uint32_t kr[4];
                    ldmatrix_x4(kr,(kpe_ldm+(n_offset+nl)*16*STRIDE_KPE_B)^(mk*32));
                    if(mk==0)mma_m16n16k16_bf16_init(qr,kr,s_frag[nl]); else mma_m16n16k16_bf16(qr,kr,s_frag[nl]);}}
                {uint32_t qr[4]; ldmatrix_x4(qr, q_nope_ldm_base^(mk*32));
                 #pragma unroll
                 for(int nl=0;nl<NUM_N_SHARD;nl++){uint32_t kr[4];
                    ldmatrix_x4(kr,(ckv_ldm+(n_offset+nl)*16*STRIDE_CKV_B)^(mk*32));
                    mma_m16n16k16_bf16(qr,kr,s_frag[nl]);}}
            }
            #pragma unroll
            for(int mk=NUM_MMA_D_KPE_K;mk<NUM_MMA_D_CKV_K;mk++){
                uint32_t qr[4]; ldmatrix_x4(qr, q_nope_ldm_base^(mk*32));
                #pragma unroll
                for(int nl=0;nl<NUM_N_SHARD;nl++){uint32_t kr[4];
                    ldmatrix_x4(kr,(ckv_ldm+(n_offset+nl)*16*STRIDE_CKV_B)^(mk*32));
                    mma_m16n16k16_bf16(qr,kr,s_frag[nl]);}
            }
        }
        if(kv_base+BN>q_start){
            int qrb=q_start+warp_m*16;
            #pragma unroll
            for(int nl=0;nl<NUM_N_SHARD;nl++){int mng=n_offset+nl;
                #pragma unroll
                for(int ri=0;ri<8;ri++){
                    int rit=((ri&2)==0)?(lane_id/4):(lane_id/4+8);
                    int kvc=2*(lane_id%4)+((ri&4)?8:0)+(ri&1);
                    int qp=qrb+rit, kvp=kv_base+mng*16+kvc;
                    if(!((kvp<=qp)&&(kvp<S))) s_frag[nl][ri]=-INFINITY;}}
        }
        {
            float mp[2]={m_state[0],m_state[1]};
            #pragma unroll
            for(int j=0;j<2;j++){
                #pragma unroll
                for(int nl=0;nl<NUM_N_SHARD;nl++){
                    float lm=fmaxf(fmaxf(s_frag[nl][j*2],s_frag[nl][j*2+1]),fmaxf(s_frag[nl][j*2+4],s_frag[nl][j*2+5]));
                    m_state[j]=fmaxf(m_state[j],lm);}
                m_state[j]=fmaxf(m_state[j],shfl_xor(m_state[j],0x2));
                m_state[j]=fmaxf(m_state[j],shfl_xor(m_state[j],0x1));
                if(lane_id%4==0) m_wg[warp_d*BM_T+warp_m*16+j*8+lane_id/4]=m_state[j];
            }
            asm volatile("bar.sync %0,%1;"::"r"(1+warp_m),"r"(BARRIER_THREADS));
            #pragma unroll
            for(int j=0;j<2;j++){
                float gm=-INFINITY;
                #pragma unroll
                for(int wd=0;wd<WARP_D_T;wd++) gm=fmaxf(gm,m_wg[wd*BM_T+warp_m*16+j*8+lane_id/4]);
                m_state[j]=gm;
                float nms=-(m_state[j]*sm_scale_log2);
                float sc=fast_exp2f(__fmaf_rn(mp[j],sm_scale_log2,nms));
                d_state[j]*=sc;
                #pragma unroll
                for(int md=0;md<NUM_MMA_D_V_WARP;md++){
                    o_frag[md][j*2+0]*=sc; o_frag[md][j*2+1]*=sc;
                    o_frag[md][j*2+4]*=sc; o_frag[md][j*2+5]*=sc;}
                #pragma unroll
                for(int nl=0;nl<NUM_N_SHARD;nl++){
                    s_frag[nl][j*2+0]=fast_exp2f(__fmaf_rn(s_frag[nl][j*2+0],sm_scale_log2,nms));
                    s_frag[nl][j*2+1]=fast_exp2f(__fmaf_rn(s_frag[nl][j*2+1],sm_scale_log2,nms));
                    s_frag[nl][j*2+4]=fast_exp2f(__fmaf_rn(s_frag[nl][j*2+4],sm_scale_log2,nms));
                    s_frag[nl][j*2+5]=fast_exp2f(__fmaf_rn(s_frag[nl][j*2+5],sm_scale_log2,nms));}
            }
        }
        uint32_t pf[NUM_N_SHARD][4];
        #pragma unroll
        for(int nl=0;nl<NUM_N_SHARD;nl++){
            #pragma unroll
            for(int i=0;i<4;i++) pf[nl][i]=float2_to_bf16x2(s_frag[nl][i*2],s_frag[nl][i*2+1]);
            mma_rowsum_bf16(d_state,pf[nl]);
        }
        constexpr int P_STRIDE = BN*sizeof(bf16);
        int p_smem = kpe_smem(stage);
        #pragma unroll
        for(int nl=0;nl<NUM_N_SHARD;nl++){
            int mng=n_offset+nl, pr=warp_m*16+lane_id/4;
            int pc=mng*16+2*(lane_id%4), pc2=mng*16+2*(lane_id%4)+8;
            int cu=pc/8,cu2=pc2/8,co=(pc%8)*2,co2=(pc2%8)*2;
            asm volatile("st.shared.u32 [%0],%1;"::"r"(p_smem+swizzle<P_STRIDE>(pr,cu)+co),"r"(pf[nl][0]));
            asm volatile("st.shared.u32 [%0],%1;"::"r"(p_smem+swizzle<P_STRIDE>(pr+8,cu)+co),"r"(pf[nl][1]));
            asm volatile("st.shared.u32 [%0],%1;"::"r"(p_smem+swizzle<P_STRIDE>(pr,cu2)+co2),"r"(pf[nl][2]));
            asm volatile("st.shared.u32 [%0],%1;"::"r"(p_smem+swizzle<P_STRIDE>(pr+8,cu2)+co2),"r"(pf[nl][3]));
        }
        asm volatile("bar.sync %0,%1;"::"r"(1+warp_m),"r"(BARRIER_THREADS));
        {
            int vcb=warp_d*(D_CKV/2/8)+lane_id/16;
            int plr=warp_m*16+(lane_id%16), plc=lane_id/16;
            #pragma unroll
            for(int mk=0;mk<NUM_MMA_KV;mk++){
                uint32_t pfr[4];
                ldmatrix_x4(pfr,p_smem+swizzle<P_STRIDE>(plr,mk*2+plc));
                #pragma unroll
                for(int md=0;md<NUM_MMA_D_V_WARP;md++){
                    uint32_t vf[4]; int vr=(lane_id%16)+mk*16, vc=vcb+md*2;
                    ldmatrix_x4_trans(vf,ckv_smem(stage)+swizzle<STRIDE_CKV_B>(vr,vc));
                    mma_m16n16k16_bf16(pfr,vf,o_frag[md]);}
            }
        }
    }
    {
        float* ds = m_wg;
        #pragma unroll
        for(int j=0;j<2;j++){if(lane_id%4==0)ds[warp_d*BM_T+warp_m*16+j*8+lane_id/4]=d_state[j];}
        asm volatile("bar.sync %0,%1;"::"r"(1+warp_m),"r"(BARRIER_THREADS));
        #pragma unroll
        for(int j=0;j<2;j++){d_state[j]=0.f;
            #pragma unroll
            for(int wd=0;wd<WARP_D_T;wd++)d_state[j]+=ds[wd*BM_T+warp_m*16+j*8+lane_id/4];}
        float dr[2];
        #pragma unroll
        for(int j=0;j<2;j++){if(m_state[j]!=-INFINITY)asm volatile("rcp.approx.ftz.f32 %0,%1;":"=f"(dr[j]):"f"(d_state[j]));else dr[j]=0.f;}
        #pragma unroll
        for(int md=0;md<NUM_MMA_D_V_WARP;md++)
            #pragma unroll
            for(int ri=0;ri<8;ri++) o_frag[md][ri]*=dr[(ri%4)/2];
    }
    {
        int g=lane_id/4, t=lane_id%4;
        #pragma unroll
        for(int md=0;md<NUM_MMA_D_V_WARP;md++){
            int db=warp_d*D_V/WARP_D_T+md*16;
            int qp=q_start+warp_m*16+g;
            if(qp<S){long long off=boo+(long long)qp*H*D_V+(long long)head*D_V+db;
                *reinterpret_cast<bf16_2*>(&O[off+2*t])=__float22bfloat162_rn(make_float2(o_frag[md][0],o_frag[md][1]));
                *reinterpret_cast<bf16_2*>(&O[off+2*t+8])=__float22bfloat162_rn(make_float2(o_frag[md][4],o_frag[md][5]));}
            qp=q_start+warp_m*16+g+8;
            if(qp<S){long long off=boo+(long long)qp*H*D_V+(long long)head*D_V+db;
                *reinterpret_cast<bf16_2*>(&O[off+2*t])=__float22bfloat162_rn(make_float2(o_frag[md][2],o_frag[md][3]));
                *reinterpret_cast<bf16_2*>(&O[off+2*t+8])=__float22bfloat162_rn(make_float2(o_frag[md][6],o_frag[md][7]));}
        }
    }
}

// ============================================================
// tcgen05 helpers
// ============================================================
__device__ __forceinline__ uint32_t elect_sync_tc() {
    uint32_t p = 0;
    asm volatile("{.reg .pred %%px;\n\t elect.sync _|%%px, 0xFFFFFFFF;\n\t @%%px mov.s32 %0, 1;}" : "+r"(p));
    return p;
}
__device__ __forceinline__ void mbar_init(int addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count));
}
__device__ __forceinline__ void mbar_wait(int addr, int phase) {
    asm volatile("{.reg .pred P;\n\t"
        "WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1, 0x989680;\n\t"
        "@P bra DONE;\n\t bra WAIT;\n\t DONE:}" :: "r"(addr), "r"(phase));
}
__device__ __forceinline__ constexpr uint64_t desc_enc(uint64_t x) { return (x & 0x3FFFFULL) >> 4; }
__device__ __forceinline__ uint64_t make_smem_desc(int smem_addr) {
    constexpr uint64_t SBO = 8ULL * 128;
    return desc_enc(smem_addr) | (desc_enc(SBO) << 32) | (1ULL << 46) | (2ULL << 61);
}
__device__ __forceinline__ void tcgen05_mma_f(int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, int acc) {
    asm volatile("{.reg .pred p;\n\t setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;}"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void tcgen05_commit_f(int mbar_addr) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(mbar_addr) : "memory");
}
__device__ __forceinline__ int swz128(int row, int col16) {
    return row * 128 + (col16 ^ (row & 7)) * 16;
}

// ============================================================
// tcgen05 MLA Kernel with TMA-based CKV loading
// ============================================================
static constexpr int TC_BM = 128;
static constexpr int TC_BK = 64;
static constexpr int TC_TB = 256;
static constexpr int TC_K_ITERS = 9;
static constexpr int TC_V_CHUNKS = 8;
static constexpr int TC_Q_BLOCK = TC_BM * TC_BK * 2;
static constexpr int TC_K_BLOCK = BN * TC_BK * 2;
static constexpr int TC_NQ = 9;
static constexpr int TC_CKV_SET_SIZE = 8 * TC_K_BLOCK + TC_Q_BLOCK;
static constexpr int TC_SMEM = TC_NQ * TC_Q_BLOCK + TC_CKV_SET_SIZE;

// Streaming Q version: Q not pre-loaded to SMEM, loaded per-tile from L2
// SMEM layout: 2 Q buffers (double-buffer) + CKV set
static constexpr int TC2_Q_BUF = 2 * TC_Q_BLOCK;  // 2 Q chunks for double-buffering
static constexpr int TC2_SMEM = TC2_Q_BUF + TC_CKV_SET_SIZE; // ~114KB

__global__ __launch_bounds__(TC_TB)
void mla_prefill_tcgen05_kernel(
    const __grid_constant__ CUtensorMap ckv_tma_desc,
    const __grid_constant__ CUtensorMap kpe_tma_desc,
    const __grid_constant__ CUtensorMap qnope_tma_desc,
    const __grid_constant__ CUtensorMap qpe_tma_desc,
    const bf16* __restrict__ Q_nope, const bf16* __restrict__ Q_pe,
    const bf16* __restrict__ CKV, const bf16* __restrict__ KPE,
    bf16* __restrict__ O, int S, int H, float sm_scale_log2)
{
    const int head = blockIdx.x;
    const int q_block = cdiv(S, TC_BM) - 1 - blockIdx.y;
    const int batch = blockIdx.z;
    const int q_start = q_block * TC_BM;
    const int tid = threadIdx.x;
    const int wid = tid / 32;

    const long long bqno = (long long)batch*S*H*D_CKV;
    const long long bqpo = (long long)batch*S*H*D_KPE;
    const long long boo  = (long long)batch*S*H*D_V;

    extern __shared__ __align__(1024) char smem_buf[];
    const int sb = __cvta_generic_to_shared(smem_buf);
    auto qslot = [&](int s) { return sb + s * TC_Q_BLOCK; };
    auto ckb = [&](int vc) { return sb + TC_NQ*TC_Q_BLOCK + vc*TC_K_BLOCK; };
    auto wrk = [&]() { return sb + TC_NQ*TC_Q_BLOCK + 8*TC_K_BLOCK; };

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ uint64_t mbars[4];
    __shared__ int tmem_buf[1];
    const int mainloop_bar = __cvta_generic_to_shared(&mbars[0]);
    const int tma_bar = __cvta_generic_to_shared(&mbars[1]);

    if (wid == 0 && elect_sync_tc()) {
        mbar_init(mainloop_bar, 1);
        mbar_init(tma_bar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (wid == 1) {
        int a = __cvta_generic_to_shared(tmem_buf);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(D_V));
    }
    __syncthreads();
    const int taddr = tmem_buf[0];

    constexpr uint32_t idesc_qk = (1U<<4)|(1U<<7)|(1U<<10)|((BN>>3)<<17)|((TC_BM>>4)<<24);
    constexpr uint32_t idesc_pv = (1U<<4)|(1U<<7)|(1U<<10)|(1U<<16)|((TC_BK>>3)<<17)|((TC_BM>>4)<<24);

    int kv_end = min(S, q_start + TC_BM);
    int num_kv_tiles = cdiv(kv_end, BN);
    float row_max = -1e30f, row_sum = 0.0f;
    float o_save[BN];
    int mma_phase = 0;
    int tma_phase = 0;

    // Load Q via TMA (3D descriptors with head indexing)
    {
        if (tid == 0) {
            constexpr int q_tx = 8 * TC_BM * TC_BK * (int)sizeof(bf16) + TC_BM * TC_BK * (int)sizeof(bf16);
            mbar_expect_tx(tma_bar, q_tx);
            for (int ki = 0; ki < 8; ki++) {
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%2, %3, %4}], [%5];"
                    :: "r"(qslot(ki)), "l"((uint64_t)&qnope_tma_desc),
                       "r"(ki * TC_BK), "r"(head), "r"(q_start),
                       "r"(tma_bar) : "memory");
            }
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                "[%0], [%1, {%2, %3, %4}], [%5];"
                :: "r"(qslot(8)), "l"((uint64_t)&qpe_tma_desc),
                   "r"(0), "r"(head), "r"(q_start),
                   "r"(tma_bar) : "memory");
        }
        mbar_wait(tma_bar, tma_phase); tma_phase ^= 1;
        __syncthreads();
    }


    // TMA-based CKV loading helper
    auto tma_load_ckv = [&](int kv_base_l) {
        if (tid == 0) {
            constexpr int tx_bytes = (8 * BN * TC_BK + BN * TC_BK) * sizeof(bf16); // 9 tiles
            mbar_expect_tx(tma_bar, tx_bytes);
            for (int vc = 0; vc < 8; vc++) {
                tma_load_2d(&ckv_tma_desc, ckb(vc), tma_bar, vc * TC_BK, kv_base_l);
            }
            tma_load_2d(&kpe_tma_desc, wrk(), tma_bar, 0, kv_base_l);
        }
    };

    // Pre-load CKV tile 0 via TMA
    if (num_kv_tiles > 0) {
        tma_load_ckv(0);
        mbar_wait(tma_bar, tma_phase); tma_phase ^= 1;
        __syncthreads();
    }

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_base = kv_tile * BN;

        // QK
        if (wid == 1 && elect_sync_tc()) {
            for (int ki = 0; ki < TC_K_ITERS; ki++) {
                int qb = qslot(ki);
                int kb = (ki<8) ? ckb(ki) : wrk();
                for (int k2 = 0; k2 < TC_BK/MMA_K; k2++)
                    tcgen05_mma_f(taddr, make_smem_desc(qb+k2*32), make_smem_desc(kb+k2*32), idesc_qk, (ki==0&&k2==0)?0:1);
            }
            tcgen05_commit_f(mainloop_bar);
        }
        mbar_wait(mainloop_bar, mma_phase); mma_phase ^= 1;

        // Softmax
        float sv[BN];
        float corr = 1.0f;
        if (tid < 128) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            for (int c = 0; c < BN; c += 16) {
                int a = taddr + (tid<<16) + c;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                    : "=f"(sv[c]),"=f"(sv[c+1]),"=f"(sv[c+2]),"=f"(sv[c+3]),
                      "=f"(sv[c+4]),"=f"(sv[c+5]),"=f"(sv[c+6]),"=f"(sv[c+7]),
                      "=f"(sv[c+8]),"=f"(sv[c+9]),"=f"(sv[c+10]),"=f"(sv[c+11]),
                      "=f"(sv[c+12]),"=f"(sv[c+13]),"=f"(sv[c+14]),"=f"(sv[c+15])
                    : "r"(a));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
            }
            int qp = q_start + tid;
            #pragma unroll
            for (int c = 0; c < BN; c++) {
                int kvp = kv_base + c;
                if (kvp > qp || kvp >= S) sv[c] = -1e30f;
                else sv[c] *= sm_scale_log2;
            }
            float tm = -1e30f;
            #pragma unroll
            for (int c = 0; c < BN; c++) tm = fmaxf(tm, sv[c]);
            float nm = fmaxf(row_max, tm);
            corr = fast_exp2f(row_max - nm);
            float ts = 0.0f;
            #pragma unroll
            for (int c = 0; c < BN; c++) { sv[c] = fast_exp2f(sv[c]-nm); ts += sv[c]; }
            row_sum = corr * row_sum + ts;
            row_max = nm;

            int ps = wrk();
            #pragma unroll
            for (int c = 0; c < BN; c += 8) {
                uint32_t w[4];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    bf16 b0 = __float2bfloat16(sv[c+j*2]), b1 = __float2bfloat16(sv[c+j*2+1]);
                    w[j] = (uint32_t)(*(uint16_t*)&b0) | ((uint32_t)(*(uint16_t*)&b1) << 16);
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ps+swz128(tid,c/8)), "r"(w[0]),"r"(w[1]),"r"(w[2]),"r"(w[3]));
            }
        }

        // O rescale — use both warp groups (256 threads) for 2x parallelism
        // Threads 0-127 handle cols BN..BN+224-1, threads 128-255 handle cols BN+224..511
        if (kv_tile > 0 && corr != 1.0f) {
            int tmem_row = tid % 128;
            int col_half = tid / 128;  // 0 or 1
            int c_start = BN + col_half * ((D_V - BN) / 2);
            int c_end = c_start + (D_V - BN) / 2;
            // Broadcast corr from warp group 0 to warp group 1 via shared mem
            float my_corr;
            {
                float* corr_buf = reinterpret_cast<float*>(smem_buf);
                if (tid < 128) corr_buf[tid] = corr;
                __syncthreads();
                my_corr = corr_buf[tmem_row];
                __syncthreads();
            }
            if (my_corr != 1.0f) {
                for (int c = c_start; c < c_end; c += 16) {
                    float t[16]; int a = taddr+(tmem_row<<16)+c;
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                        :"=f"(t[0]),"=f"(t[1]),"=f"(t[2]),"=f"(t[3]),"=f"(t[4]),"=f"(t[5]),"=f"(t[6]),"=f"(t[7]),
                         "=f"(t[8]),"=f"(t[9]),"=f"(t[10]),"=f"(t[11]),"=f"(t[12]),"=f"(t[13]),"=f"(t[14]),"=f"(t[15])
                        :"r"(a));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int i=0;i<16;i++) t[i]*=my_corr;
                    uint32_t* u=(uint32_t*)t;
                    asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
                        "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
                        ::"r"(a),"r"(u[0]),"r"(u[1]),"r"(u[2]),"r"(u[3]),"r"(u[4]),"r"(u[5]),"r"(u[6]),"r"(u[7]),
                          "r"(u[8]),"r"(u[9]),"r"(u[10]),"r"(u[11]),"r"(u[12]),"r"(u[13]),"r"(u[14]),"r"(u[15]));
                }
            }
        }
        __syncthreads();

        // PV
        if (wid == 1 && elect_sync_tc()) {
            int pab = (kv_tile > 0) ? 1 : 0;
            for (int vc = 0; vc < TC_V_CHUNKS; vc++) {
                int ot = taddr + vc * TC_BK;
                int vab = (vc < 1) ? 0 : pab;
                int first = 1;
                for (int k2 = 0; k2 < BN/MMA_K; k2++) {
                    tcgen05_mma_f(ot, make_smem_desc(wrk()+k2*32), make_smem_desc(ckb(vc)+k2*16*128), idesc_pv, (first&&!vab)?0:1);
                    first = 0;
                }
            }
            tcgen05_commit_f(mainloop_bar);
        }
        mbar_wait(mainloop_bar, mma_phase); mma_phase ^= 1;

        // After PV: CKV is free. Start TMA load for next tile (non-blocking!)
        if (kv_tile + 1 < num_kv_tiles) {
            tma_load_ckv((kv_tile + 1) * BN);
        }

        // O merge
        if (kv_tile > 0 && tid < 128) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            for (int c = 0; c < BN; c += 16) {
                float t[16]; int a = taddr+(tid<<16)+c;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                    :"=f"(t[0]),"=f"(t[1]),"=f"(t[2]),"=f"(t[3]),"=f"(t[4]),"=f"(t[5]),"=f"(t[6]),"=f"(t[7]),
                     "=f"(t[8]),"=f"(t[9]),"=f"(t[10]),"=f"(t[11]),"=f"(t[12]),"=f"(t[13]),"=f"(t[14]),"=f"(t[15])
                    :"r"(a));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int i=0;i<16;i++) t[i]+=corr*o_save[c+i];
                uint32_t* u=(uint32_t*)t;
                asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
                    "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
                    ::"r"(a),"r"(u[0]),"r"(u[1]),"r"(u[2]),"r"(u[3]),"r"(u[4]),"r"(u[5]),"r"(u[6]),"r"(u[7]),
                      "r"(u[8]),"r"(u[9]),"r"(u[10]),"r"(u[11]),"r"(u[12]),"r"(u[13]),"r"(u[14]),"r"(u[15]));
            }
        }

        // O save
        if (kv_tile + 1 < num_kv_tiles && tid < 128) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            for (int c = 0; c < BN; c += 16) {
                int a = taddr + (tid<<16) + c;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                    : "=f"(o_save[c]),"=f"(o_save[c+1]),"=f"(o_save[c+2]),"=f"(o_save[c+3]),
                      "=f"(o_save[c+4]),"=f"(o_save[c+5]),"=f"(o_save[c+6]),"=f"(o_save[c+7]),
                      "=f"(o_save[c+8]),"=f"(o_save[c+9]),"=f"(o_save[c+10]),"=f"(o_save[c+11]),
                      "=f"(o_save[c+12]),"=f"(o_save[c+13]),"=f"(o_save[c+14]),"=f"(o_save[c+15])
                    : "r"(a));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
            }
        }

        // Wait for TMA CKV load
        if (kv_tile + 1 < num_kv_tiles) {
            mbar_wait(tma_bar, tma_phase); tma_phase ^= 1;
            __syncthreads();
        }
    }

    // Epilogue: SMEM-staged coalesced output
    {
    constexpr int STAGE_ROWS = 64;
    constexpr int STAGE_STRIDE = D_V * 2;
    float inv = 0.0f;
    if (tid < 128) {
        asm volatile("tcgen05.fence::after_thread_sync;");
        inv = (row_sum > 0) ? 1.0f/row_sum : 0.0f;
    }
    for (int half = 0; half < 2; half++) {
        int row_off = half * STAGE_ROWS;
        if (tid < 128) {
            int my_row = tid;
            if (my_row >= row_off && my_row < row_off + STAGE_ROWS && q_start + my_row < S) {
                int smem_row = my_row - row_off;
                for (int vc = 0; vc < TC_V_CHUNKS; vc++) {
                    for (int c = 0; c < TC_BK; c += 16) {
                        float t[16]; int a = taddr+vc*TC_BK+(tid<<16)+c;
                        asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                            :"=f"(t[0]),"=f"(t[1]),"=f"(t[2]),"=f"(t[3]),"=f"(t[4]),"=f"(t[5]),"=f"(t[6]),"=f"(t[7]),
                             "=f"(t[8]),"=f"(t[9]),"=f"(t[10]),"=f"(t[11]),"=f"(t[12]),"=f"(t[13]),"=f"(t[14]),"=f"(t[15])
                            :"r"(a));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        int smem_off = sb + smem_row * STAGE_STRIDE + (vc*TC_BK + c) * 2;
                        #pragma unroll
                        for (int i = 0; i < 16; i += 2) {
                            bf16_2 v = __float22bfloat162_rn(make_float2(t[i]*inv, t[i+1]*inv));
                            *reinterpret_cast<bf16_2*>((char*)0 + smem_off + i*2) = v;
                        }
                    }
                }
            }
        }
        __syncthreads();
        constexpr int ELEMS_PER_THREAD = D_V / TC_TB;
        for (int r = 0; r < STAGE_ROWS; r++) {
            int qp = q_start + row_off + r;
            if (qp < S) {
                int col = tid * 2;
                bf16_2 val = *reinterpret_cast<bf16_2*>((char*)0 + sb + r * STAGE_STRIDE + col * 2);
                long long go = boo + (long long)qp*H*D_V + (long long)head*D_V + col;
                *reinterpret_cast<bf16_2*>(&O[go]) = val;
            }
        }
        __syncthreads();
    }
    }
    __syncthreads();
    if (wid == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(D_V));
}

// ============================================================
// Host: create TMA descriptors + launch
// ============================================================
void launch_mla_prefill(
    const bf16* Q_nope, const bf16* Q_pe, const bf16* CKV, const bf16* KPE,
    bf16* O, int B, int S, int H, float sm_scale, cudaStream_t stream = 0)
{
    float sm_scale_log2 = sm_scale * 1.44269504089f;
    if (S <= 512) {
        constexpr int BM_L=32, WD=4;
        constexpr int ss = BM_L*D_CKV*2+BM_L*D_KPE*2+2*(BN*D_CKV*2+BN*D_KPE*2)+WD*BM_L*4;
        dim3 grid(H, cdiv(S,BM_L), B);
        auto k = mla_prefill_mmasync_kernel<BM_L, WD, true>;
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, ss);
        k<<<grid, 256, ss, stream>>>(Q_nope, Q_pe, CKV, KPE, O, S, H, sm_scale_log2);
    } else if (S <= 1024) {
        constexpr int BM_L=64, WD=2;
        constexpr int ss = BM_L*D_CKV*2+BM_L*D_KPE*2+2*(BN*D_CKV*2+BN*D_KPE*2)+WD*BM_L*4;
        dim3 grid(H, cdiv(S,BM_L), B);
        auto k = mla_prefill_mmasync_kernel<BM_L, WD, true>;
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, ss);
        k<<<grid, 256, ss, stream>>>(Q_nope, Q_pe, CKV, KPE, O, S, H, sm_scale_log2);
    } else {
        // Create TMA descriptors for CKV and KPE
        CUtensorMap ckv_tma, kpe_tma;
        // CKV: [D_CKV, S] bf16, stride D_CKV*2 between rows (S dim)
        {
            uint64_t gDim[2] = {(uint64_t)D_CKV, (uint64_t)S};
            uint64_t gStr[1] = {(uint64_t)(D_CKV * sizeof(bf16))};
            uint32_t bDim[2] = {(uint32_t)TC_BK, (uint32_t)BN};
            uint32_t eStr[2] = {1, 1};
            cuTensorMapEncodeTiled(&ckv_tma, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
                (void*)CKV, gDim, gStr, bDim, eStr,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
        }
        // KPE: [D_KPE, S] bf16
        {
            uint64_t gDim[2] = {(uint64_t)D_KPE, (uint64_t)S};
            uint64_t gStr[1] = {(uint64_t)(D_KPE * sizeof(bf16))};
            uint32_t bDim[2] = {(uint32_t)TC_BK, (uint32_t)BN};
            uint32_t eStr[2] = {1, 1};
            cuTensorMapEncodeTiled(&kpe_tma, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
                (void*)KPE, gDim, gStr, bDim, eStr,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
        }
        // Q_nope: [D_CKV, H, S] bf16 — 3D for per-head strided access
        CUtensorMap qnope_tma, qpe_tma;
        {
            uint64_t gDim[3] = {(uint64_t)D_CKV, (uint64_t)H, (uint64_t)S};
            uint64_t gStr[2] = {(uint64_t)(D_CKV * sizeof(bf16)), (uint64_t)((uint64_t)H * D_CKV * sizeof(bf16))};
            uint32_t bDim[3] = {(uint32_t)TC_BK, 1, (uint32_t)TC_BM};
            uint32_t eStr[3] = {1, 1, 1};
            cuTensorMapEncodeTiled(&qnope_tma, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
                (void*)Q_nope, gDim, gStr, bDim, eStr,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
        }
        // Q_pe: [D_KPE, H, S] bf16
        {
            uint64_t gDim[3] = {(uint64_t)D_KPE, (uint64_t)H, (uint64_t)S};
            uint64_t gStr[2] = {(uint64_t)(D_KPE * sizeof(bf16)), (uint64_t)((uint64_t)H * D_KPE * sizeof(bf16))};
            uint32_t bDim[3] = {(uint32_t)TC_BK, 1, (uint32_t)TC_BM};
            uint32_t eStr[3] = {1, 1, 1};
            cuTensorMapEncodeTiled(&qpe_tma, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
                (void*)Q_pe, gDim, gStr, bDim, eStr,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
        }
        int nqb = cdiv(S, TC_BM);
        dim3 grid(H, nqb, B);
        auto k = mla_prefill_tcgen05_kernel;
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, TC_SMEM);
        k<<<grid, TC_TB, TC_SMEM, stream>>>(ckv_tma, kpe_tma, qnope_tma, qpe_tma, Q_nope, Q_pe, CKV, KPE, O, S, H, sm_scale_log2);
    }
}

// ============================================================
// Benchmark
// ============================================================
int main(int argc, char** argv) {
    cuInit(0);
    int B=1, H=16;
    int seq_lens[]={512,1024,2048}; int nc=3;
    float sm_scale = 1.0f/sqrtf(576.0f);
    printf("=== MLA Prefill Combined Kernel (B200, H=%d) ===\n", H);
    int S_max=2048;
    bf16 *dQn,*dQp,*dC,*dK,*dO;
    size_t qns=(size_t)B*S_max*H*D_CKV*sizeof(bf16);
    size_t qps=(size_t)B*S_max*H*D_KPE*sizeof(bf16);
    size_t cs=(size_t)B*S_max*D_CKV*sizeof(bf16);
    size_t ks=(size_t)B*S_max*D_KPE*sizeof(bf16);
    size_t os=(size_t)B*S_max*H*D_V*sizeof(bf16);
    cudaMalloc(&dQn,qns); cudaMalloc(&dQp,qps);
    cudaMalloc(&dC,cs); cudaMalloc(&dK,ks); cudaMalloc(&dO,os);
    void* df; cudaMalloc(&df,128*1024*1024);
    cudaMemset(dQn,1,qns); cudaMemset(dQp,1,qps);
    cudaMemset(dC,1,cs); cudaMemset(dK,1,ks);
    cudaEvent_t st,en; cudaEventCreate(&st); cudaEventCreate(&en);
    printf("KERNEL_RESULT {");
    for(int ci=0;ci<nc;ci++){
        int S=seq_lens[ci];
        for(int i=0;i<20;i++) launch_mla_prefill(dQn,dQp,dC,dK,dO,B,S,H,sm_scale);
        cudaDeviceSynchronize();
        cudaError_t err=cudaGetLastError();
        if(err!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(err));return 1;}
        int NI=100; std::vector<float> bt;
        for(int bi=0;bi<5;bi++){
            cudaEventRecord(st);
            for(int i=0;i<NI;i++) launch_mla_prefill(dQn,dQp,dC,dK,dO,B,S,H,sm_scale);
            cudaEventRecord(en); cudaEventSynchronize(en);
            float ms; cudaEventElapsedTime(&ms,st,en); bt.push_back(ms/NI);
        }
        std::sort(bt.begin(),bt.end()); float ms=bt[bt.size()/2];
        double fl=2.0*(double)B*H*S*S*(D_CKV+D_KPE+D_V);
        double tf=fl/(ms/1000.0)/1e12;
        printf("\"S%d\": %.2f",S,tf);
        if(ci<nc-1) printf(", ");
        fprintf(stderr,"S%d: %.2f TFLOPS, %.1f us\n",S,tf,ms*1000.0);
    }
    printf("}\n");
    printf("KERNEL_RESULT_REFERENCE {\"S512\": 276.44, \"S1024\": 502.40, \"S2048\": 1450.73}\n");
    cudaEventDestroy(st); cudaEventDestroy(en);
    cudaFree(dQn); cudaFree(dQp); cudaFree(dC); cudaFree(dK); cudaFree(dO); cudaFree(df);
    return 0;
}
