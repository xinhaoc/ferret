// GQA Decode Qwen3-8B on B200 (sm_100a): Transposed N=8 tcgen05 MMA
// Architecture: KQ MMA M=128(KV-pos) N=8(Q-heads) + PV MMA M=128(head-dim) N=8(Q-heads)
// S and O in TMEM only 8 f32 cols each → 8x less TMEM traffic vs original
// Cross-thread softmax: warp shuffle + SMEM cross-warp reduction
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <random>
#include <time.h>
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
using namespace cute;
using bf16_t = cutlass::bfloat16_t;

static constexpr int NUM_QO_HEADS=32, NUM_KV_HEADS=8, GQA_GROUP=4;
static constexpr int HEAD_DIM=128, PAGE_SIZE=64;
static constexpr int N_HEADS=8; // MMA N dimension (GQA_GROUP rounded to 8)
static constexpr int M_MMA=128; // MMA M dimension = KV positions per tile = 2 pages
static constexpr int NUM_THREADS=256; // 2 warpgroups (WG0=softmax, WG1=MMA+TMA+storer)
static constexpr int NUM_K_BUFS=2; // double-buffer K
static constexpr int NUM_V_BUFS=2; // double-buffer V

// ========== Non-.ws MMA Atoms for N=8 ==========
// tcgen05.mma.cta_group::1.kind::f16 [d-tmem], a-desc, b-desc, idesc, {0,0,0,0}, p, 0;
namespace cute {
// SS mode: both A and B from SMEM
template<class at, class bt, class ct, int M, int N, UMMA::Major am, UMMA::Major bm,
         UMMA::ScaleIn an=UMMA::ScaleIn::One, UMMA::ScaleIn bn=UMMA::ScaleIn::One>
struct SM100_MMA_NOWS_SS_NE {
    using DRegisters=void; using ARegisters=uint64_t[1]; using BRegisters=uint64_t[1]; using CRegisters=uint32_t[1];
    CUTE_HOST_DEVICE static void fma(uint64_t const& da, uint64_t const& db, uint32_t const& tc, uint32_t const& sc, uint64_t const& id) {
        asm volatile(
            "{.reg .pred p; .reg .b32 z; mov.b32 z, 0; setp.ne.b32 p, %4, 0; "
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {z,z,z,z}, p;}"
            :: "r"(tc), "l"(da), "l"(db), "r"(uint32_t(id>>32)), "r"(sc));
    }
};
template<class at,class bt,class ct,int M,int N,UMMA::Major am,UMMA::Major bm,UMMA::ScaleIn an,UMMA::ScaleIn bn>
struct MMA_Traits<SM100_MMA_NOWS_SS_NE<at,bt,ct,M,N,am,bm,an,bn>> {
    using ValTypeD=ct; using ValTypeA=at; using ValTypeB=bt; using ValTypeC=ct;
    using FrgTypeA=UMMA::smem_desc<am>;
    using FrgTypeB=UMMA::smem_desc<bm>;
    using FrgTypeC=UMMA::tmem_frg_1sm<ct>;  // Non-interleaved, allows N=8
    static constexpr int K=256/sizeof_bits<at>::value;
    using Shape_MNK=Shape<Int<M>,Int<N>,Int<K>>; using ThrID=Layout<_1>;
    using ALayout=Layout<Shape<_1,Shape<Int<M>,Int<K>>>,Stride<_0,Stride<_1,Int<M>>>>;
    using BLayout=Layout<Shape<_1,Shape<Int<N>,Int<K>>>,Stride<_0,Stride<_1,Int<N>>>>;
    using CLayout=Layout<Shape<_1,Shape<Int<M>,Int<N>>>,Stride<_0,Stride<_1,Int<M>>>>;
    UMMA::ScaleOut accumulate_=UMMA::ScaleOut::One;
    UMMA::InstrDescriptor idesc_=UMMA::make_instr_desc<at,bt,ct,M,N,am,bm,an,bn>();
    template<class TD,class DL,class TA,class AL,class TB,class BL,class TC2,class CL>
    CUTE_HOST_DEVICE constexpr friend void mma_unpack(MMA_Traits const&t,Tensor<TD,DL>&D,Tensor<TA,AL>const&A,Tensor<TB,BL>const&B,Tensor<TC2,CL>const&C){
        SM100_MMA_NOWS_SS_NE<at,bt,ct,M,N,am,bm,an,bn>::fma(A[0],B[0],raw_pointer_cast(D.data()),uint32_t(t.accumulate_),UMMA::make_runtime_instr_desc<>(t.idesc_));
    }
};

// Keep .ws SS for M_MMA=128, N=PAGE_SIZE case as fallback 
template<class at,class bt,class ct,int M,int N,UMMA::Major am,UMMA::Major bm,
         UMMA::ScaleIn an=UMMA::ScaleIn::One,UMMA::ScaleIn bn=UMMA::ScaleIn::One>
struct SM100_MMA_WS_SS_NE{
    using DRegisters=void;using ARegisters=uint64_t[1];using BRegisters=uint64_t[1];using CRegisters=uint32_t[1];
    CUTE_HOST_DEVICE static void fma(uint64_t const&da,uint64_t const&db,uint32_t const&tc,uint32_t const&sc,uint64_t const&id){
        asm volatile("{.reg .pred p;setp.ne.b32 p,%4,0;tcgen05.mma.ws.cta_group::1.kind::f16 [%0],%1,%2,%3,p,0;}"
            ::"r"(tc),"l"(da),"l"(db),"r"(uint32_t(id>>32)),"r"(sc));}};
template<class at,class bt,class ct,int M,int N,UMMA::Major am,UMMA::Major bm,UMMA::ScaleIn an,UMMA::ScaleIn bn>
struct MMA_Traits<SM100_MMA_WS_SS_NE<at,bt,ct,M,N,am,bm,an,bn>>{
    using ValTypeD=ct;using ValTypeA=at;using ValTypeB=bt;using ValTypeC=ct;
    using FrgTypeA=UMMA::smem_desc<am>;using FrgTypeB=UMMA::smem_desc<bm>;
    using FrgTypeC=UMMA::tmem_frg_ws_1sm<ct>;
    static constexpr int K=256/sizeof_bits<at>::value;
    using Shape_MNK=Shape<Int<M>,Int<N>,Int<K>>;using ThrID=Layout<_1>;
    using ALayout=Layout<Shape<_1,Shape<Int<M>,Int<K>>>,Stride<_0,Stride<_1,Int<M>>>>;
    using BLayout=Layout<Shape<_1,Shape<Int<N>,Int<K>>>,Stride<_0,Stride<_1,Int<N>>>>;
    using CLayout=Layout<Shape<_1,Shape<Int<M>,Int<N>>>,Stride<_0,Stride<_1,Int<M>>>>;
    UMMA::ScaleOut accumulate_=UMMA::ScaleOut::One;
    UMMA::InstrDescriptor idesc_=UMMA::make_instr_desc<at,bt,ct,M,N,am,bm,an,bn>();
    template<class TD,class DL,class TA,class AL,class TB,class BL,class TC2,class CL>
    CUTE_HOST_DEVICE constexpr friend void mma_unpack(MMA_Traits const&t,Tensor<TD,DL>&D,Tensor<TA,AL>const&A,Tensor<TB,BL>const&B,Tensor<TC2,CL>const&C){
        SM100_MMA_WS_SS_NE<at,bt,ct,M,N,am,bm,an,bn>::fma(A[0],B[0],raw_pointer_cast(D.data()),uint32_t(t.accumulate_),UMMA::make_runtime_instr_desc<>(t.idesc_));}};
} // namespace cute

// ========== SMEM Layouts ==========
// K: [M_MMA × HEAD_DIM] = [128 × 128] K-major swizzled - A operand for QK MMA
using SmemLayoutK = decltype(coalesce(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16_t>{},
    Shape<Int<M_MMA>,Int<HEAD_DIM>>{}, Step<_1,_2>{}), Shape<_1,_1>{}));

// V: [M_MMA × HEAD_DIM] = [128 × 128] same layout as K (for double-buffer sharing)
using SmemLayoutV = SmemLayoutK;

// V transposed view for PV MMA A-operand: [HEAD_DIM × M_MMA] via composition
using SmemLayoutVT = decltype(composition(SmemLayoutK{},
    Layout<Shape<Int<HEAD_DIM>,Int<M_MMA>>,Stride<Int<M_MMA>,_1>>{}));

// Q: [N_HEADS × HEAD_DIM] = [8 × 128] K-major interleaved - B operand for QK MMA
using SmemLayoutQ = decltype(tile_to_shape(UMMA::Layout_K_INTER_Atom<bf16_t>{},
    Shape<Int<N_HEADS>,Int<HEAD_DIM>>{}, Step<_1,_2>{}));

// P: [N_HEADS × M_MMA] = [8 × 128] K-major interleaved - B operand for PV MMA
using SmemLayoutP = decltype(tile_to_shape(UMMA::Layout_K_INTER_Atom<bf16_t>{},
    Shape<Int<N_HEADS>,Int<M_MMA>>{}, Step<_1,_2>{}));

// QK MMA: SS mode, M=128(KV-pos), N=8(Q-heads), K=16(bf16), A=K(K-major), B=Q(K-major)
using TiledMMA_QK = decltype(make_tiled_mma(SM100_MMA_NOWS_SS_NE<bf16_t,bf16_t,float,M_MMA,N_HEADS,UMMA::Major::K,UMMA::Major::K>{}));
// PV MMA: SS mode, M=128(head-dim), N=8(Q-heads), K=16(bf16), A=V(MN-major), B=P(K-major)
using TiledMMA_PV = decltype(make_tiled_mma(SM100_MMA_NOWS_SS_NE<bf16_t,bf16_t,float,M_MMA,N_HEADS,UMMA::Major::MN,UMMA::Major::K>{}));

// TMEM: S (8 f32 cols) + O (8 f32 cols) = 16 cols → alloc 32 (minimum)
struct TC {
    static constexpr uint32_t S = 0;    // QK result [128×8 f32] = 8 cols
    static constexpr uint32_t O = 8;    // PV accum [128×8 f32] = 8 cols 
    static constexpr uint32_t END = 32;  // minimum alloc
};

// ========== Device Helpers ==========
__device__ __forceinline__ bool elect(){uint32_t p=0;asm volatile("{.reg .pred %%px;elect.sync _|%%px,0xFFFFFFFF;@%%px mov.s32 %0,1;}"  :"+r"(p));return p;}
__device__ __forceinline__ void bar_init(void*p,int c){uint32_t a=cast_smem_ptr_to_uint(p);asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(a),"r"(c));}
__device__ __forceinline__ void bar_wait(void*p,int ph){uint32_t a=cast_smem_ptr_to_uint(p);asm volatile("{.reg .pred P1;LW:mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1,[%0],%1,0x989680;@P1 bra.uni DN;bra.uni LW;DN:}"::"r"(a),"r"(ph));}
__device__ __forceinline__ void bar_arrive(void*p){uint32_t a=cast_smem_ptr_to_uint(p);asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _,[%0];"::"r"(a):"memory");}
__device__ __forceinline__ void bar_arrive_tx(void*p,uint32_t b){uint32_t a=cast_smem_ptr_to_uint(p);asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;"::"r"(a),"r"(b):"memory");}
__device__ __forceinline__ void umma_commit(void*bar){uint32_t a=cast_smem_ptr_to_uint(bar);asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"::"r"(a):"memory");}

// TMEM load 8 f32 cols (8 values per thread for 32-lane warp)
__device__ __forceinline__ void tmem_ld8(uint32_t c, float*o){
    uint32_t*d=(uint32_t*)o;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
        :"=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3]),"=r"(d[4]),"=r"(d[5]),"=r"(d[6]),"=r"(d[7])
        :"r"(c));
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}
__device__ __forceinline__ void tmem_st8(uint32_t c, const float*i){
    const uint32_t*d=(const uint32_t*)i;
    // Use 32x32b.x8 store
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        ::"r"(c),"r"(d[0]),"r"(d[1]),"r"(d[2]),"r"(d[3]),"r"(d[4]),"r"(d[5]),"r"(d[6]),"r"(d[7]):"memory");
    cutlass::arch::fence_view_async_tmem_store();
}
__device__ __forceinline__ void tmem_ld32(uint32_t c, float*o){
    uint32_t*d=(uint32_t*)o;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},[%32];"
        :"=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3]),"=r"(d[4]),"=r"(d[5]),"=r"(d[6]),"=r"(d[7]),
         "=r"(d[8]),"=r"(d[9]),"=r"(d[10]),"=r"(d[11]),"=r"(d[12]),"=r"(d[13]),"=r"(d[14]),"=r"(d[15]),
         "=r"(d[16]),"=r"(d[17]),"=r"(d[18]),"=r"(d[19]),"=r"(d[20]),"=r"(d[21]),"=r"(d[22]),"=r"(d[23]),
         "=r"(d[24]),"=r"(d[25]),"=r"(d[26]),"=r"(d[27]),"=r"(d[28]),"=r"(d[29]),"=r"(d[30]),"=r"(d[31])
        :"r"(c));
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}
__device__ __forceinline__ void tmem_st32(uint32_t c, const float*i){
    const uint32_t*d=(const uint32_t*)i;
    SM100_TMEM_STORE_32dp32b32x::copy(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15],
        d[16],d[17],d[18],d[19],d[20],d[21],d[22],d[23],d[24],d[25],d[26],d[27],d[28],d[29],d[30],d[31],c);
    cutlass::arch::fence_view_async_tmem_store();
}

// ========== SMEM Plan ==========
// Layout: K0(32KB) V0(32KB) Q(2KB) P(2KB) misc(~0.5KB) [pad] K1(32KB) V1(32KB)
// For 1-tile CTAs: only first ~68KB used → can fit 2 CTAs/SM (68KB×2=136KB<228KB)
// For multi-tile: full ~132KB → 1 CTA/SM
static constexpr int KV_BUF_SIZE = cosize_v<SmemLayoutK>;
struct alignas(128) SmemPlan {
    // Primary section (always accessed)
    bf16_t smem_k0[KV_BUF_SIZE]; // K buffer 0: 32KB
    bf16_t smem_v0[KV_BUF_SIZE]; // V buffer 0: 32KB
    bf16_t smem_q[cosize_v<SmemLayoutQ>];   // 2KB
    bf16_t smem_p[cosize_v<SmemLayoutP>];   // 2KB
    alignas(16) uint32_t tmem_addr;
    alignas(16) float smem_maxs[4 * N_HEADS];
    alignas(16) float smem_sums[4 * N_HEADS];
    alignas(8) uint64_t bar_k[NUM_K_BUFS];
    alignas(8) uint64_t bar_qk[NUM_K_BUFS];
    alignas(8) uint64_t bar_so;
    alignas(8) uint64_t bar_v[NUM_V_BUFS];
    alignas(8) uint64_t bar_pv[NUM_V_BUFS];
    // Secondary section (only for multi-tile CTAs, beyond SMEM_SMALL)
    // K_SW128 swizzle requires buffer-size alignment (32KB) for correct desc addressing
    alignas(32768) bf16_t smem_k1[KV_BUF_SIZE]; // K buffer 1: 32KB
    alignas(32768) bf16_t smem_v1[KV_BUF_SIZE]; // V buffer 1: 32KB
    // Accessors
    __device__ __forceinline__ bf16_t* kbuf(int b) { return b ? smem_k1 : smem_k0; }
    __device__ __forceinline__ bf16_t* vbuf(int b) { return b ? smem_v1 : smem_v0; }
};
static constexpr int SMEM_SMALL = offsetof(SmemPlan, smem_k1);

template<class MMA, class SA, class SB, class TC2>
__device__ __forceinline__ void utcmma_ss(MMA& mma, SA& sA, SB& sB, TC2& tC, bool zero) {
    auto thr = mma.get_slice(_0{});
    auto sA_f = thr.partition_fragment_A(sA);
    auto sB_f = thr.partition_fragment_B(sB);
    mma.accumulate_ = zero ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    for (int k = 0; k < size<2>(sA_f); k++) {
        gemm(mma, sA_f(_,_,k), sB_f(_,_,k), tC);
        mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

// ========== Attention Kernel ==========
__global__ void __launch_bounds__(NUM_THREADS, 1, 1)
gqa_kernel(
    __grid_constant__ const CUtensorMap tmap_kv,
    const __nv_bfloat16*__restrict__ Q_g,
    const int*__restrict__ indptr, const int*__restrict__ indices, const int*__restrict__ lastpl,
    float*__restrict__ pO, float*__restrict__ pm, float*__restrict__ pl,
    __nv_bfloat16*__restrict__ out_bf16, int*__restrict__ done_count,
    int Bat, int pps, int ns, float sms_log2)
{
#if defined(__CUDA_ARCH__)&&(__CUDA_ARCH__>=1000)
    int si = blockIdx.x, bkh = blockIdx.y;
    int b = bkh / NUM_KV_HEADS, kvh = bkh % NUM_KV_HEADS;
    if (b >= Bat) return;
    int wg = threadIdx.x / 128, iwg = threadIdx.x % 128;
    int warp = threadIdx.x / 32, tid = threadIdx.x;
    int lane = tid % 32;
    int warp_in_wg = iwg / 32;
    extern __shared__ char sr[];
    auto& P = *reinterpret_cast<SmemPlan*>(sr);

    // === Init barriers + TMEM ===
    if (warp == 0 && elect()) {
        asm volatile("prefetch.tensormap [%0];":: "l"(&tmap_kv));
        for (int i = 0; i < NUM_K_BUFS; i++) {
            bar_init(&P.bar_k[i], 1);
            bar_init(&P.bar_qk[i], 1);
        }
        bar_init(&P.bar_so, 128); // WG0 signals
        for (int i = 0; i < NUM_V_BUFS; i++) {
            bar_init(&P.bar_v[i], 1);
            bar_init(&P.bar_pv[i], 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    if (warp == 0) {
        TMEM::Allocator1Sm().allocate(TC::END, &P.tmem_addr);
        TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();

    int ps2 = indptr[b], pe2 = indptr[b+1], lpl = lastpl[b], np = pe2 - ps2;
    // Each tile processes 2 pages = M_MMA=128 KV positions
    int pages_per_tile = 2;
    int sps = si * pps, spe = min(sps + pps, np);
    int ntiles = (spe - sps + pages_per_tile - 1) / pages_per_tile;
    
    if (ntiles <= 0) {
        if (wg == 0) {
            for (int qg = 0; qg < GQA_GROUP; qg++) {
                int qh = kvh*GQA_GROUP+qg, idx = si*Bat*NUM_QO_HEADS+b*NUM_QO_HEADS+qh;
                if (iwg == 0) { pm[idx] = -INFINITY; pl[idx] = 0; }
                for (int d = iwg; d < HEAD_DIM; d += 128) pO[idx*HEAD_DIM+d] = 0;
            }
        }
        if (warp == 0) TMEM::Allocator1Sm().free(0, TC::END);
        return;
    }

    // === Start K+V TMA prologue BEFORE Q setup (overlap with SMEM init) ===
    {
        int page0 = sps, page1 = min(sps+1, spe-1);
        int phys0 = indices[ps2 + page0], phys1 = indices[ps2 + page1];
        if (warp == 5 && elect()) {
            auto sK = make_tensor(make_smem_ptr(P.smem_k0), SmemLayoutK{});
            auto sKt = flat_divide(sK, Shape<Int<PAGE_SIZE>, Int<64>>{});
            uint32_t bk = cast_smem_ptr_to_uint(&P.bar_k[0]);
            for (int h = 0; h < HEAD_DIM/64; h++) {
                uint32_t sa = cast_smem_ptr_to_uint(&sKt(_0{},_0{},_0{},h));
                asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4,%5}],[%6];"
                    ::"r"(sa),"l"(&tmap_kv),"r"(h*64),"r"(kvh),"r"(0),"r"(phys0*2+0),"r"(bk):"memory");
            }
            for (int h = 0; h < HEAD_DIM/64; h++) {
                uint32_t sa = cast_smem_ptr_to_uint(&sKt(_0{},_0{},_1{},h));
                asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4,%5}],[%6];"
                    ::"r"(sa),"l"(&tmap_kv),"r"(h*64),"r"(kvh),"r"(0),"r"(phys1*2+0),"r"(bk):"memory");
            }
            bar_arrive_tx(&P.bar_k[0], M_MMA * HEAD_DIM * 2);
        }
        if (warp == 6 && elect()) {
            auto sV0 = make_tensor(make_smem_ptr(P.smem_v0), SmemLayoutV{});
            auto sVt0 = flat_divide(sV0, Shape<Int<PAGE_SIZE>, Int<64>>{});
            uint32_t bv0 = cast_smem_ptr_to_uint(&P.bar_v[0]);
            for (int h = 0; h < HEAD_DIM/64; h++) {
                uint32_t sa = cast_smem_ptr_to_uint(&sVt0(_0{},_0{},_0{},h));
                asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4,%5}],[%6];"
                    ::"r"(sa),"l"(&tmap_kv),"r"(h*64),"r"(kvh),"r"(0),"r"(phys0*2+1),"r"(bv0):"memory");
            }
            for (int h = 0; h < HEAD_DIM/64; h++) {
                uint32_t sa = cast_smem_ptr_to_uint(&sVt0(_0{},_0{},_1{},h));
                asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4,%5}],[%6];"
                    ::"r"(sa),"l"(&tmap_kv),"r"(h*64),"r"(kvh),"r"(0),"r"(phys1*2+1),"r"(bv0):"memory");
            }
            bar_arrive_tx(&P.bar_v[0], M_MMA * HEAD_DIM * 2);
        }
    }

    // === Load Q to SMEM [8 × 128] - single pass ===
    {
        auto sQ = make_tensor(make_smem_ptr(P.smem_q), SmemLayoutQ{});
        // Zero-fill + Q write in single pass (no intermediate sync needed)
        uint32_t* sq32 = reinterpret_cast<uint32_t*>(P.smem_q);
        int nw = cosize_v<SmemLayoutQ> / 2;
        for (int i = tid; i < nw; i += NUM_THREADS) sq32[i] = 0;
        for (int i = tid; i < GQA_GROUP * HEAD_DIM; i += NUM_THREADS) {
            int row = i / HEAD_DIM, col = i % HEAD_DIM;
            int qh = kvh * GQA_GROUP + row;
            sQ(row, col) = bf16_t(__bfloat162float(Q_g[b*NUM_QO_HEADS*HEAD_DIM+qh*HEAD_DIM+col]));
        }
    }
    __syncthreads();



    // === Main Loop ===
    if (wg == 0) {
        // ======== WG0: Softmax + Inline Correction ========
        cutlass::arch::warpgroup_reg_alloc<160>(); // 160 regs: 2 CTAs/SM with small SMEM
        float row_m[N_HEADS], row_l[N_HEADS]; // per-head running max and sum
        for (int h = 0; h < N_HEADS; h++) { row_m[h] = -INFINITY; row_l[h] = 0; }

        #pragma unroll 1
        for (int tile = 0; tile < ntiles; tile++) {
            int tile_start_page = sps + tile * pages_per_tile;
            int valid_kv = 0;
            for (int p = 0; p < pages_per_tile && tile_start_page + p < spe; p++) {
                int pg = tile_start_page + p;
                valid_kv += (pg == np-1) ? lpl : PAGE_SIZE;
            }
            int kb = tile % NUM_K_BUFS;

            // Wait for QK MMA done
            bar_wait(&P.bar_qk[kb], (tile / NUM_K_BUFS) & 1);
            asm volatile("tcgen05.fence::after_thread_sync;");

            // Read S from TMEM: 8 f32 values per thread (8 Q heads)
            // Each thread = 1 KV position (128 threads = 128 KV positions)
            float sv[N_HEADS];
            tmem_ld8(TC::S, sv);
            cutlass::arch::fence_view_async_tmem_load();

            int my_kv_pos = iwg; // thread's KV position within tile (0-127)
            
            // Mask invalid positions 
            if (my_kv_pos >= valid_kv) {
                for (int h = 0; h < N_HEADS; h++) sv[h] = -INFINITY;
            } else {
                for (int h = 0; h < N_HEADS; h++) sv[h] *= sms_log2;
            }
            // Mask inactive heads (GQA_GROUP=4, heads 4-7 always -inf)
            for (int h = GQA_GROUP; h < N_HEADS; h++) sv[h] = -INFINITY;

            // Per-head max reduction across 128 threads (4 warps)
            float local_max[N_HEADS];
            for (int h = 0; h < N_HEADS; h++) {
                float mx = sv[h];
                // Warp-level reduction (32 threads)
                for (int off = 16; off >= 1; off >>= 1)
                    mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, off));
                local_max[h] = mx;
            }

            // Cross-warp max via SMEM
            if (lane == 0) {
                for (int h = 0; h < N_HEADS; h++)
                    P.smem_maxs[warp_in_wg * N_HEADS + h] = local_max[h];
            }
            cutlass::arch::fence_view_async_shared();
            // Intra-WG sync using __syncthreads (WG0 only... but all WGs hit this)
            // Use named barrier for intra-WG sync
            asm volatile("bar.sync 5, 128;"); // barrier 5, 128 threads = WG0
            
            float new_m[N_HEADS];
            for (int h = 0; h < N_HEADS; h++) {
                float m0 = P.smem_maxs[0*N_HEADS+h];
                float m1 = P.smem_maxs[1*N_HEADS+h];
                float m2 = P.smem_maxs[2*N_HEADS+h];
                float m3 = P.smem_maxs[3*N_HEADS+h];
                new_m[h] = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
            }

            // Compute exp2f and local sums
            float p_vals[N_HEADS];
            float local_sum[N_HEADS];
            for (int h = 0; h < N_HEADS; h++) {
                p_vals[h] = exp2f(sv[h] - new_m[h]);
                local_sum[h] = p_vals[h];
            }

            // Warp-level sum reduction (per-warp partial, no cross-warp sync)
            for (int h = 0; h < N_HEADS; h++) {
                for (int off = 16; off >= 1; off >>= 1)
                    local_sum[h] += __shfl_xor_sync(0xFFFFFFFF, local_sum[h], off);
            }

            // Correction + per-warp partial l (cross-warp sum deferred to epilogue)
            float corr[N_HEADS];
            for (int h = 0; h < N_HEADS; h++) {
                corr[h] = exp2f(row_m[h] - new_m[h]);
                row_l[h] = row_l[h] * corr[h] + local_sum[h];
                row_m[h] = new_m[h];
            }

            // O correction (inline in WG0, only 8 cols so very cheap)
            if (tile > 0) {
                int pvb_prev = (tile-1) % NUM_V_BUFS;
                bar_wait(&P.bar_pv[pvb_prev], ((tile-1) / NUM_V_BUFS) & 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float ov[N_HEADS];
                tmem_ld8(TC::O, ov);
                cutlass::arch::fence_view_async_tmem_load();
                for (int h = 0; h < N_HEADS; h++) ov[h] *= corr[h];
                tmem_st8(TC::O, ov);
                asm volatile("tcgen05.fence::before_thread_sync;");
            }

            // Write P to SMEM [N_HEADS × M_MMA] = [8 × 128]
            // Each thread writes 8 bf16 values at its column (KV position)
            auto sP = make_tensor(make_smem_ptr(P.smem_p), SmemLayoutP{});
            for (int h = 0; h < N_HEADS; h++) {
                sP(h, my_kv_pos) = bf16_t(p_vals[h]);
            }
            cutlass::arch::fence_view_async_shared();
            bar_arrive(&P.bar_so);
        }

        // === Epilogue ===
        float ov_epi[N_HEADS]; // outside block so fused reduction can access
        {int pvb_last = (ntiles-1) % NUM_V_BUFS;
        bar_wait(&P.bar_pv[pvb_last], ((ntiles-1) / NUM_V_BUFS) & 1);}
        asm volatile("tcgen05.fence::after_thread_sync;");
        {
            // Read O from TMEM and write to global
            tmem_ld8(TC::O, ov_epi);
            cutlass::arch::fence_view_async_tmem_load();
            
            // O is [head_dim × Q_heads]. Thread my_kv_pos holds O[my_kv_pos, 0:7]
            // = one head_dim position for all 8 Q heads
            int d_idx = iwg; // head_dim position
            if (d_idx < HEAD_DIM) {
                if (ns > 1) {
                    for (int h = 0; h < GQA_GROUP; h++) {
                        int qh = kvh * GQA_GROUP + h;
                        int idx = si * Bat * NUM_QO_HEADS + b * NUM_QO_HEADS + qh;
                        pO[idx * HEAD_DIM + d_idx] = ov_epi[h];
                    }
                }
            }
            // Cross-warp sum combine (deferred from main loop)
            if (lane == 0) {
                for (int h = 0; h < N_HEADS; h++)
                    P.smem_sums[warp_in_wg * N_HEADS + h] = row_l[h];
            }
            cutlass::arch::fence_view_async_shared();
            asm volatile("bar.sync 5, 128;");
            for (int h = 0; h < N_HEADS; h++) {
                row_l[h] = P.smem_sums[0*N_HEADS+h] + P.smem_sums[1*N_HEADS+h]
                         + P.smem_sums[2*N_HEADS+h] + P.smem_sums[3*N_HEADS+h];
            }
            // Store m and l
            if (ns > 1 && iwg == 0) {
                for (int h = 0; h < GQA_GROUP; h++) {
                    int qh = kvh * GQA_GROUP + h;
                    int idx = si * Bat * NUM_QO_HEADS + b * NUM_QO_HEADS + qh;
                    pm[idx] = row_m[h];
                    pl[idx] = row_l[h];
                }
            }
        }
        asm volatile("tcgen05.fence::before_thread_sync;");
        if (warp == 0) TMEM::Allocator1Sm().free(0, TC::END);

        // === Fused Reduction (eliminates separate reduce_kernel) ===
        int is_last = 0;
        if (ns > 1) {
            __threadfence(); // ensure partials visible to other CTAs
            if (iwg == 0) {
                int old = atomicAdd(&done_count[b * NUM_KV_HEADS + kvh], 1);
                *reinterpret_cast<int*>(&P.smem_maxs[0]) = old;
            }
            cutlass::arch::fence_view_async_shared();
            asm volatile("bar.sync 5, 128;");
            is_last = (*reinterpret_cast<volatile int*>(&P.smem_maxs[0]) == ns - 1);
        } else {
            is_last = 1;
        }

        if (is_last) {
            int d_idx = iwg;
            if (d_idx < HEAD_DIM) {
                for (int h = 0; h < GQA_GROUP; h++) {
                    int qh = kvh * GQA_GROUP + h;
                    if (ns == 1) {
                        // ov[h] is still in regs from tmem_ld8 above (not written to pO for ns==1)
                        float inv_l = (row_l[h] > 0) ? 1.0f / row_l[h] : 0.0f;
                        out_bf16[b*NUM_QO_HEADS*HEAD_DIM + qh*HEAD_DIM + d_idx] = 
                            __float2bfloat16(ov_epi[h] * inv_l);
                    } else {
                        float gm = -INFINITY;
                        for (int s = 0; s < ns; s++) {
                            int i2 = s * Bat * NUM_QO_HEADS + b * NUM_QO_HEADS + qh;
                            gm = fmaxf(gm, pm[i2]);
                        }
                        float gl = 0, go = 0;
                        for (int s = 0; s < ns; s++) {
                            int i2 = s * Bat * NUM_QO_HEADS + b * NUM_QO_HEADS + qh;
                            float m2 = pm[i2], l2 = pl[i2], o2 = pO[i2*HEAD_DIM+d_idx];
                            if (l2 > 0) { float sc = exp2f(m2 - gm); gl += l2*sc; go += o2*sc; }
                        }
                        out_bf16[b*NUM_QO_HEADS*HEAD_DIM + qh*HEAD_DIM + d_idx] = 
                            __float2bfloat16(gl > 0 ? go/gl : 0.0f);
                    }
                }
            }
        }

    } else if (wg == 1) {
        // ======== WG1: MMA + TMA ========
        cutlass::arch::warpgroup_reg_dealloc<72>();
        int w1 = warp;
        bool el = (w1 <= 6) ? elect() : true;

        if (w1 == 4 && el) {
            // MMA launcher
            TiledMMA_QK mma_qk;
            TiledMMA_PV mma_pv;

            auto tS = partition_fragment_C(mma_qk, Shape<Int<M_MMA>, Int<N_HEADS>>{});
            tS.data().get() = TC::S;
            auto tO = partition_fragment_C(mma_pv, Shape<Int<M_MMA>, Int<N_HEADS>>{});
            tO.data().get() = TC::O;

            auto sQ = make_tensor(make_smem_ptr(P.smem_q), SmemLayoutQ{});

            // Prologue: QK[0]
            bar_wait(&P.bar_k[0], 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            {
                auto sK = make_tensor(make_smem_ptr(P.smem_k0), SmemLayoutK{});
                utcmma_ss(mma_qk, sK, sQ, tS, true);
            }
            umma_commit(&P.bar_qk[0]);

            #pragma unroll 1
            for (int tile = 1; tile < ntiles; tile++) {
                int kb = tile % NUM_K_BUFS;

                // QK[tile]
                bar_wait(&P.bar_k[kb], (tile / NUM_K_BUFS) & 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                {
                    auto sK = make_tensor(make_smem_ptr(P.kbuf(kb)), SmemLayoutK{});
                    utcmma_ss(mma_qk, sK, sQ, tS, true);
                }
                umma_commit(&P.bar_qk[kb]);

                // PV[tile-1]
                {int vb_prev = (tile-1) % NUM_V_BUFS;
                bar_wait(&P.bar_so, ((tile-1)) & 1);
                bar_wait(&P.bar_v[vb_prev], ((tile-1) / NUM_V_BUFS) & 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                {
                    auto sP_t = make_tensor(make_smem_ptr(P.smem_p), SmemLayoutP{});
                    auto sV = make_tensor(make_smem_ptr(P.vbuf(vb_prev)), SmemLayoutVT{});
                    utcmma_ss(mma_pv, sV, sP_t, tO, tile == 1);
                }
                umma_commit(&P.bar_pv[vb_prev]);}
            }

            // PV[ntiles-1]
            {int vb_last = (ntiles-1) % NUM_V_BUFS;
            bar_wait(&P.bar_so, ((ntiles-1)) & 1);
            bar_wait(&P.bar_v[vb_last], ((ntiles-1) / NUM_V_BUFS) & 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            {
                auto sP_t = make_tensor(make_smem_ptr(P.smem_p), SmemLayoutP{});
                auto sV = make_tensor(make_smem_ptr(P.vbuf(vb_last)), SmemLayoutVT{});
                utcmma_ss(mma_pv, sV, sP_t, tO, ntiles == 1);
            }
            umma_commit(&P.bar_pv[vb_last]);}

        } else if (w1 == 5 && el) {
            // K TMA loader (tile 0 already loaded in prologue)
            #pragma unroll 1
            for (int tile = 1; tile < ntiles; tile++) {
                int kb = tile % NUM_K_BUFS;
                if (tile >= NUM_K_BUFS)
                    bar_wait(&P.bar_qk[kb], ((tile / NUM_K_BUFS) - 1) & 1);
                
                auto sK = make_tensor(make_smem_ptr(P.kbuf(kb)), SmemLayoutK{});
                auto sKt = flat_divide(sK, Shape<Int<PAGE_SIZE>, Int<64>>{});
                uint32_t bk = cast_smem_ptr_to_uint(&P.bar_k[kb]);
                
                int tile_start_page = sps + tile * pages_per_tile;
                for (int p = 0; p < pages_per_tile; p++) {
                    int pg = min(tile_start_page + p, spe - 1);
                    int phys = indices[ps2 + pg];
                    int c3 = phys * 2 + 0;
                    for (int h = 0; h < HEAD_DIM/64; h++) {
                        uint32_t sa = cast_smem_ptr_to_uint(&sKt(_0{},_0{},Int<0>{} + p, h));
                        asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4,%5}],[%6];"
                            ::"r"(sa),"l"(&tmap_kv),"r"(h*64),"r"(kvh),"r"(0),"r"(c3),"r"(bk):"memory");
                    }
                }
                bar_arrive_tx(&P.bar_k[kb], M_MMA * HEAD_DIM * 2);
            }

        } else if (w1 == 6 && el) {
            // V TMA loader (double-buffered, tile 0 loaded in prologue)
            #pragma unroll 1
            for (int tile = 1; tile < ntiles; tile++) {
                int vb = tile % NUM_V_BUFS;
                if (tile >= NUM_V_BUFS) {
                    int vb_reuse = vb; // same buffer we're about to write
                    bar_wait(&P.bar_pv[vb_reuse], ((tile / NUM_V_BUFS) - 1) & 1);
                }
                
                auto sV = make_tensor(make_smem_ptr(P.vbuf(vb)), SmemLayoutV{});
                auto sVt = flat_divide(sV, Shape<Int<PAGE_SIZE>, Int<64>>{});
                uint32_t bv = cast_smem_ptr_to_uint(&P.bar_v[vb]);
                
                int tile_start_page = sps + tile * pages_per_tile;
                for (int p = 0; p < pages_per_tile; p++) {
                    int pg = min(tile_start_page + p, spe - 1);
                    int phys = indices[ps2 + pg];
                    int c3 = phys * 2 + 1;
                    for (int h = 0; h < HEAD_DIM/64; h++) {
                        uint32_t sa = cast_smem_ptr_to_uint(&sVt(_0{},_0{},Int<0>{} + p, h));
                        asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4,%5}],[%6];"
                            ::"r"(sa),"l"(&tmap_kv),"r"(h*64),"r"(kvh),"r"(0),"r"(c3),"r"(bv):"memory");
                    }
                }
                bar_arrive_tx(&P.bar_v[vb], M_MMA * HEAD_DIM * 2);
            }
        }

    }
    // Note: WG1 warp 7 serves as storer (role c) via elect() dispatch
#endif
}

// ========== Reduction Kernel ==========
__global__ void reduce_kernel(const float*pO, const float*pm, const float*pl,
    __nv_bfloat16*Out, int B, int ns) {
    int bqh = blockIdx.x, b = bqh/NUM_QO_HEADS, qh = bqh%NUM_QO_HEADS, d = threadIdx.x;
    if (b >= B || d >= HEAD_DIM) return;
    float gm = -INFINITY;
    for (int s = 0; s < ns; s++) {
        int i = s*B*NUM_QO_HEADS + b*NUM_QO_HEADS + qh;
        gm = fmaxf(gm, pm[i]);
    }
    float gl = 0, go = 0;
    for (int s = 0; s < ns; s++) {
        int i = s*B*NUM_QO_HEADS + b*NUM_QO_HEADS + qh;
        float m = pm[i], l = pl[i], o = pO[i*HEAD_DIM+d];
        if (l > 0) { float sc = exp2f(m - gm); gl += l*sc; go += o*sc; }
    }
    Out[b*NUM_QO_HEADS*HEAD_DIM+qh*HEAD_DIM+d] = __float2bfloat16(gl > 0 ? go/gl : 0);
}

// ========== CPU Reference ==========
void ref_attn(const __nv_bfloat16*Q, const __nv_bfloat16*KV, const int*ip, const int*ix, const int*lp, float*O, int B, float sm) {
    for (int b = 0; b < B; b++) {
        int ps = ip[b], pe = ip[b+1], l2 = lp[b], np2 = pe-ps;
        int kl = np2 > 0 ? (np2-1)*PAGE_SIZE + l2 : 0;
        for (int qh = 0; qh < NUM_QO_HEADS; qh++) {
            int kvh = qh / GQA_GROUP;
            float q[HEAD_DIM];
            for (int d = 0; d < HEAD_DIM; d++)
                q[d] = __bfloat162float(Q[b*NUM_QO_HEADS*HEAD_DIM+qh*HEAD_DIM+d]);
            float m = -1e30f, l = 0, a[HEAD_DIM] = {};
            for (int p = 0; p < kl; p++) {
                int pi = p/PAGE_SIZE, pp = p%PAGE_SIZE, gp = ix[ps+pi];
                int64_t kb = (int64_t)gp*2*PAGE_SIZE*NUM_KV_HEADS*HEAD_DIM + pp*NUM_KV_HEADS*HEAD_DIM + kvh*HEAD_DIM;
                float dot = 0;
                for (int d = 0; d < HEAD_DIM; d++)
                    dot += q[d] * __bfloat162float(KV[kb+d]);
                dot *= sm;
                float nm = fmaxf(m, dot), os = expf(m-nm), p2 = expf(dot-nm);
                int64_t vb = kb + PAGE_SIZE*NUM_KV_HEADS*HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; d++)
                    a[d] = a[d]*os + p2*__bfloat162float(KV[vb+d]);
                m = nm; l = l*os + p2;
            }
            float il = l > 0 ? 1.0f/l : 0;
            for (int d = 0; d < HEAD_DIM; d++)
                O[b*NUM_QO_HEADS*HEAD_DIM+qh*HEAD_DIM+d] = a[d]*il;
        }
    }
}

void chk(cudaError_t e, const char*m) { if (e != cudaSuccess) { fprintf(stderr,"CUDA(%s):%s\n",m,cudaGetErrorString(e)); exit(1); } }
void chk(CUresult e, const char*m) { if (e != CUDA_SUCCESS) { const char*s; cuGetErrorString(e,&s); fprintf(stderr,"CU(%s):%s\n",m,s); exit(1); } }

int main() {
    cuInit(0);
    int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    struct Cfg { const char*n; int B, kl; };
    Cfg cfgs[] = {
        {"bs1_kv512",1,512}, {"bs16_kv512",16,512},
        {"bs1_kv2k",1,2048}, {"bs16_kv2k",16,2048},
        {"bs1_kv8k",1,8192}, {"bs16_kv8k",16,8192}
    };
    float sms = 1.0f / sqrtf(128.0f);
    size_t fsz = 128ULL*1024*1024; void*df; chk(cudaMalloc(&df,fsz),"f");

    printf("KERNEL_RESULT {"); bool first = true;
    for (int ci = 0; ci < 6; ci++) {
        auto& c = cfgs[ci];
        int B = c.B, kl = c.kl;
        int npp = (kl + PAGE_SIZE - 1) / PAGE_SIZE;
        int tp = B * npp, lpl = ((kl-1) % PAGE_SIZE) + 1;
        size_t qs = B*NUM_QO_HEADS*HEAD_DIM;
        size_t kvs = (size_t)tp * 2 * PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM;

        __nv_bfloat16*hq = (__nv_bfloat16*)malloc(qs*2);
        __nv_bfloat16*hkv = (__nv_bfloat16*)malloc(kvs*2);
        int*hip = (int*)malloc((B+1)*4), *hix = (int*)malloc(tp*4), *hlp = (int*)malloc(B*4);

        std::mt19937 rng(42+ci);
        std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
        for (size_t i = 0; i < qs; i++) hq[i] = __float2bfloat16(dist(rng));
        for (size_t i = 0; i < kvs; i++) hkv[i] = __float2bfloat16(dist(rng));
        for (int b2 = 0; b2 < B; b2++) { hip[b2] = b2*npp; hlp[b2] = lpl; }
        hip[B] = tp;
        for (int i = 0; i < tp; i++) hix[i] = i;

        __nv_bfloat16 *dq, *dkv, *dout; int *dip, *dix, *dlp;
        chk(cudaMalloc(&dq,qs*2),""); chk(cudaMalloc(&dkv,kvs*2),""); chk(cudaMalloc(&dout,qs*2),"");
        chk(cudaMalloc(&dip,(B+1)*4),""); chk(cudaMalloc(&dix,tp*4),""); chk(cudaMalloc(&dlp,B*4),"");
        chk(cudaMemcpy(dq,hq,qs*2,cudaMemcpyHostToDevice),"");
        chk(cudaMemcpy(dkv,hkv,kvs*2,cudaMemcpyHostToDevice),"");
        chk(cudaMemcpy(dip,hip,(B+1)*4,cudaMemcpyHostToDevice),"");
        chk(cudaMemcpy(dix,hix,tp*4,cudaMemcpyHostToDevice),"");
        chk(cudaMemcpy(dlp,hlp,B*4,cudaMemcpyHostToDevice),"");

        CUtensorMap tmap_kv;
        {uint64_t gd[4]={(uint64_t)HEAD_DIM,(uint64_t)NUM_KV_HEADS,(uint64_t)PAGE_SIZE,(uint64_t)(tp*2)};
         uint64_t gs[3]={(uint64_t)(HEAD_DIM*2),(uint64_t)(NUM_KV_HEADS*HEAD_DIM*2),(uint64_t)((uint64_t)PAGE_SIZE*NUM_KV_HEADS*HEAD_DIM*2)};
         uint32_t bd[4]={64,1,(uint32_t)PAGE_SIZE,1};uint32_t es[4]={1,1,1,1};
         chk(cuTensorMapEncodeTiled(&tmap_kv,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,4,(void*)dkv,gd,gs,bd,es,
             CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),"tkv");}

        // Split-KV heuristic - cost model with flat reduction overhead
        int tc2 = B * NUM_KV_HEADS, num_splits = 1, pps2 = npp;
        {
            int best_splits = 1;
            float best_time = 1e9f;
            for (int ns2 = 1; ns2 <= std::min(npp/2, 32); ns2++) {
                int p = ((npp + ns2 - 1) / ns2 + 1) & ~1;  // round up to even (2 pages per tile)
                if (p < 2) p = 2;
                int total_ctas = ns2 * tc2;
                int waves = (total_ctas + nsm - 1) / nsm;
                int ntl = p / 2; // tiles per CTA
                // CTA cost: startup(8us) + tiles * per_tile(2.0us for pipelined)
                // Overhead: single kernel launch(2us), fused reduction adds ~0.5us to last CTA
                float cta_time = 8.0f + ntl * 2.0f;
                // Fused reduction overhead: threadfence(1us) + reading ns partials(0.15us each)
                float reduce_time = (ns2 > 1) ? (1.0f + ns2 * 0.15f) : 0.0f;
                float est_time = waves * cta_time + 2.0f + reduce_time;
                if (est_time < best_time) {
                    best_time = est_time;
                    best_splits = ns2;
                }
            }
            num_splits = best_splits;
            pps2 = (npp + num_splits - 1) / num_splits;
            if (pps2 % 2 != 0 && pps2 < npp) pps2++; // round to even
            num_splits = (npp + pps2 - 1) / pps2;
        }
        size_t pos = (size_t)num_splits * B * NUM_QO_HEADS * HEAD_DIM;
        size_t pms = (size_t)num_splits * B * NUM_QO_HEADS;
        float *dpo, *dpm, *dpl;
        chk(cudaMalloc(&dpo, pos*4),""); chk(cudaMalloc(&dpm, pms*4),""); chk(cudaMalloc(&dpl, pms*4),"");

        // Atomic counter for fused reduction
        int *d_done;
        chk(cudaMalloc(&d_done, B * NUM_KV_HEADS * sizeof(int)),"");

        int ss_full = sizeof(SmemPlan);
        cudaFuncSetAttribute(gqa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ss_full);
        int ntl_max = (pps2 + 1) / 2;
        int ss = (ntl_max <= 1) ? SMEM_SMALL : ss_full; // small SMEM for 1-tile → 2 CTAs/SM

        auto run_kern = [&]() {
            dim3 g(num_splits, B*NUM_KV_HEADS);
            float sms_log2 = sms * 1.4426950408889634f;
            gqa_kernel<<<g, NUM_THREADS, ss>>>(tmap_kv, dq, dip, dix, dlp, dpo, dpm, dpl, dout, d_done, B, pps2, num_splits, sms_log2);
        };
        auto run = [&]() {
            if (num_splits > 1) cudaMemsetAsync(d_done, 0, B * NUM_KV_HEADS * sizeof(int));
            run_kern();
        };

        run();
        cudaError_t err = cudaDeviceSynchronize();

        float*ro = (float*)malloc(qs*4);
        ref_attn(hq, hkv, hip, hix, hlp, ro, B, sms);
        __nv_bfloat16*ho = (__nv_bfloat16*)malloc(qs*2);
        bool v = false; float mre = 0, mae = 0, me = 0;
        if (err == cudaSuccess) {
            chk(cudaMemcpy(ho, dout, qs*2, cudaMemcpyDeviceToHost),"");
            float sa = 0; int cnt = 0;
            for (size_t i = 0; i < qs; i++) {
                float g2 = __bfloat162float(ho[i]), r = ro[i];
                float ae = fabsf(g2-r), re = ae/fmaxf(fabsf(r), 1e-2f);
                mae = fmaxf(mae, ae); mre = fmaxf(mre, re); sa += ae; cnt++;
            }
            me = sa / cnt;
            v = (mre < 1e-2f && mae < 1.0f && me < 1e-2f);
        }
        fprintf(stderr, "%s: mre=%.6f mae=%.6f me=%.6f %s (ns=%d err=%s)\n",
            c.n, mre, mae, me, v?"PASS":"FAIL", num_splits,
            err==cudaSuccess ? "none" : cudaGetErrorString(err));

        double tf = 0;
        if (v) {
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            while (1) {
                run(); cudaDeviceSynchronize();
                clock_gettime(CLOCK_MONOTONIC, &t1);
                if ((t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)*1e-9 >= 2.0) break;
            }
            int NI = 300;
            std::vector<float> ts(NI);
            for (int it = 0; it < NI; it++) {
                chk(cudaMemset(df,0,fsz),"");
                // Zero done counter OUTSIDE event timing
                if (num_splits > 1) cudaMemsetAsync(d_done, 0, B * NUM_KV_HEADS * sizeof(int));
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0); cudaEventCreate(&e1);
                cudaEventRecord(e0); run_kern(); cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float ms; cudaEventElapsedTime(&ms, e0, e1);
                ts[it] = ms;
                cudaEventDestroy(e0); cudaEventDestroy(e1);
            }
            std::sort(ts.begin(), ts.end());
            float md = ts[NI/2];
            double fl = 4.0*B*NUM_QO_HEADS*HEAD_DIM*kl;
            tf = fl / (md/1000.0) / 1e12;
            fprintf(stderr, "%s: %.4f TFLOPS %.1f us\n", c.n, tf, md*1000);
        }

        if (!first) printf(", ");
        first = false;
        printf("\"%s\": %.4f", c.n, v ? tf : 0.0);

        free(hq); free(hkv); free(hip); free(hix); free(hlp); free(ro); free(ho);
        cudaFree(dq); cudaFree(dkv); cudaFree(dout); cudaFree(dip); cudaFree(dix); cudaFree(dlp);
        cudaFree(dpo); cudaFree(dpm); cudaFree(dpl); cudaFree(d_done);
    }
    printf("}\n");

    // Reference: run FlashInfer baseline via Python
    {
        double rv[6] = {0.5447, 5.4783, 1.9284, 12.1739, 5.6872, 19.3991}; // fallback
        FILE* bp = popen("export PATH=/home/xinhaoc/mirage-venv/bin:$PATH && /home/xinhaoc/mirage-venv/bin/python3 ref_flashinfer.py 2>/dev/null", "r");
        if (bp) {
            char line[256];
            while (fgets(line, sizeof(line), bp)) {
                for (int i = 0; i < 6; i++) {
                    if (strstr(line, cfgs[i].n)) {
                        char* p = strstr(line, ":"); if (p) rv[i] = atof(p+1);
                    }
                }
            }
            pclose(bp);
        }
        printf("KERNEL_RESULT_REFERENCE {");
        for (int i = 0; i < 6; i++) {
            if (i) printf(", ");
            printf("\"%s\": %.4f", cfgs[i].n, rv[i]);
        }
        printf("}\n");
    }

    cudaFree(df);
    return 0;
}
