// =============================================================================
// Unified Paged GQA Attention for Qwen3-30B-A3B (TP=4 per-rank shape)
// Handles ALL 16 configs: Q_LEN in {1,2,3,4} x SEQ_LEN in {128,512,4096,32768}
//
// Architecture: 3 kernel paths + host dispatcher:
//   Path A (Q=1): v012-style decode + combine (CUDA-core, split-KV)
//   Path B (Q>1, seq>=4096): v007-style multitoken + separate combine
//   Path C (Q>1, seq<=512): v009-style fused atomic combine (single launch)
// =============================================================================

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

using bf16_t = __nv_bfloat16;

static constexpr int HD = 128;
static constexpr int NQH = 8;
static constexpr int NKH = 1;
static constexpr int PS = 4096;
static constexpr int64_t KVS_TOK = (int64_t)NKH * HD;       // 128
static constexpr int64_t KVS_KV  = (int64_t)PS * KVS_TOK;   // 524288
static constexpr int64_t KVS_PAGE = 2 * KVS_KV;             // 1048576

static constexpr int BLK = 256; // 8 warps x 32 threads

// =============================================================================
// PATH A: Q=1 decode kernel (v012 design)
// Split-KV + SMEM-staged KV + warp-per-head, CUDA-core
// =============================================================================
static constexpr int CHUNK_Q1 = 64;

struct SmemQ1 {
    bf16_t kbuf[CHUNK_Q1][HD];
    bf16_t vbuf[CHUNK_Q1][HD];
};

__global__ void __launch_bounds__(BLK)
paged_gqa_decode(
    const bf16_t* __restrict__ Q,
    const bf16_t* __restrict__ paged_kv,
    const int* __restrict__ kv_indices,
    float* __restrict__ pO,       // [nsplits, NQH, HD]
    float* __restrict__ pLSE,     // [nsplits, NQH]
    int seq_len, int chunk_size, float sm_scale_log2
) {
    __shared__ SmemQ1 sm;
    
    const int sid = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lid = tid % 32;
    
    const int ts = sid * chunk_size;
    const int te = min(ts + chunk_size, seq_len);
    
    if (ts >= seq_len) {
        if (lid == 0) pLSE[sid * NQH + wid] = -INFINITY;
        for (int d = lid; d < HD; d += 32)
            pO[(sid * NQH + wid) * HD + d] = 0.0f;
        return;
    }
    
    float qr[4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        qr[i] = __bfloat162float(Q[wid * HD + lid * 4 + i]);
    
    float m_val = -INFINITY, l_val = 0.0f;
    float oa[4] = {0, 0, 0, 0};
    
    for (int cs = ts; cs < te; cs += CHUNK_Q1) {
        int ce = min(cs + CHUNK_Q1, te);
        int cl = ce - cs;
        
        // Load K and V to SMEM
        {
            const int pl0 = cs / PS;
            const int tip0 = cs % PS;
            const int pp0 = __ldg(kv_indices + pl0);
            const bf16_t* kbase0 = paged_kv + (int64_t)pp0 * KVS_PAGE + (int64_t)tip0 * KVS_TOK;
            const bf16_t* vbase0 = kbase0 + KVS_KV;
            
            const int page_remain = PS - tip0;
            const bool crosses = (cl > page_remain);
            
            const int total_u4 = cl * (HD / 8);
            for (int idx = tid; idx < total_u4; idx += BLK) {
                int tok_local = idx / (HD / 8);
                int u4_idx = idx % (HD / 8);
                
                const bf16_t* kp;
                const bf16_t* vp;
                if (!crosses || tok_local < page_remain) {
                    kp = kbase0 + (int64_t)tok_local * KVS_TOK;
                    vp = vbase0 + (int64_t)tok_local * KVS_TOK;
                } else {
                    int pp1 = __ldg(kv_indices + pl0 + 1);
                    int tip1 = tok_local - page_remain;
                    kp = paged_kv + (int64_t)pp1 * KVS_PAGE + (int64_t)tip1 * KVS_TOK;
                    vp = kp + KVS_KV;
                }
                ((uint4*)sm.kbuf[tok_local])[u4_idx] = __ldg(((const uint4*)kp) + u4_idx);
                ((uint4*)sm.vbuf[tok_local])[u4_idx] = __ldg(((const uint4*)vp) + u4_idx);
            }
        }
        __syncthreads();
        
        // QK^T + online softmax + PV (per-token fused, lower register pressure)
        {
            #pragma unroll 4
            for (int t = 0; t < cl; t++) {
                // Load K and V in one pass
                uint32_t kw0 = ((const uint32_t*)sm.kbuf[t])[lid * 2];
                uint32_t kw1 = ((const uint32_t*)sm.kbuf[t])[lid * 2 + 1];
                float k0 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&kw0));
                float k1 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&kw0) + 1));
                float k2 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&kw1));
                float k3 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&kw1) + 1));
                
                float dot = qr[0]*k0 + qr[1]*k1 + qr[2]*k2 + qr[3]*k3;
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    dot += __shfl_xor_sync(0xFFFFFFFF, dot, o);
                float score = dot * sm_scale_log2;
                
                float nm = fmaxf(m_val, score);
                float corr = exp2f(m_val - nm);
                float p = exp2f(score - nm);
                
                uint32_t vw0 = ((const uint32_t*)sm.vbuf[t])[lid * 2];
                uint32_t vw1 = ((const uint32_t*)sm.vbuf[t])[lid * 2 + 1];
                float v0 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&vw0));
                float v1 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&vw0) + 1));
                float v2 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&vw1));
                float v3 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&vw1) + 1));
                
                oa[0] = oa[0] * corr + p * v0;
                oa[1] = oa[1] * corr + p * v1;
                oa[2] = oa[2] * corr + p * v2;
                oa[3] = oa[3] * corr + p * v3;
                l_val = l_val * corr + p;
                m_val = nm;
            }
        }
        __syncthreads();
    }
    
    float inv_l = (l_val > 0.0f) ? __fdividef(1.0f, l_val) : 0.0f;
    float lse = (l_val > 0.0f) ? (m_val + __log2f(l_val)) : -INFINITY;
    
    if (lid == 0) pLSE[sid * NQH + wid] = lse;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        pO[(sid * NQH + wid) * HD + lid * 4 + i] = oa[i] * inv_l;
}

// Q=1 decode with cp.async double-buffered pipeline (for long seq)
struct SmemQ1_DB {
    bf16_t kbuf[2][CHUNK_Q1][HD];
    bf16_t vbuf[2][CHUNK_Q1][HD];
};

__device__ __forceinline__ void load_q1_async(
    SmemQ1_DB& sm, int buf, int cl,
    const bf16_t* __restrict__ paged_kv,
    const int* __restrict__ kv_indices,
    int cs, int tid
) {
    const int pl0 = cs / PS;
    const int tip0 = cs % PS;
    const int pp0 = __ldg(kv_indices + pl0);
    const bf16_t* kbase0 = paged_kv + (int64_t)pp0 * KVS_PAGE + (int64_t)tip0 * KVS_TOK;
    const bf16_t* vbase0 = kbase0 + KVS_KV;
    const int total_u4 = cl * (HD / 8);
    for (int idx = tid; idx < total_u4; idx += BLK) {
        int tok_local = idx / (HD / 8);
        int u4_idx = idx % (HD / 8);
        const bf16_t* kp = kbase0 + (int64_t)tok_local * KVS_TOK;
        const bf16_t* vp = vbase0 + (int64_t)tok_local * KVS_TOK;
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
            :: "l"((uint4*)&sm.kbuf[buf][tok_local][u4_idx*8]), "l"((const uint4*)kp + u4_idx) : "memory");
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
            :: "l"((uint4*)&sm.vbuf[buf][tok_local][u4_idx*8]), "l"((const uint4*)vp + u4_idx) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__global__ void __launch_bounds__(BLK)
paged_gqa_decode_async(
    const bf16_t* __restrict__ Q,
    const bf16_t* __restrict__ paged_kv,
    const int* __restrict__ kv_indices,
    float* __restrict__ pO, float* __restrict__ pLSE,
    int seq_len, int chunk_size, float sm_scale_log2
) {
    __shared__ SmemQ1_DB sm;
    const int sid = blockIdx.x, tid = threadIdx.x, wid = tid/32, lid = tid%32;
    const int ts = sid * chunk_size, te = min(ts + chunk_size, seq_len);
    if (ts >= seq_len) {
        if (lid == 0) pLSE[sid * NQH + wid] = -INFINITY;
        for (int d = lid; d < HD; d += 32) pO[(sid * NQH + wid) * HD + d] = 0.0f;
        return;
    }
    float qr[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) qr[i] = __bfloat162float(Q[wid * HD + lid * 4 + i]);
    float m_val = -INFINITY, l_val = 0.0f, oa[4] = {0,0,0,0};

    int num_chunks = (te - ts + CHUNK_Q1 - 1) / CHUNK_Q1;
    // Prefetch first chunk
    if (num_chunks > 0) { int cl0 = min(ts + CHUNK_Q1, te) - ts; load_q1_async(sm, 0, cl0, paged_kv, kv_indices, ts, tid); }

    for (int ci = 0; ci < num_chunks; ci++) {
        int cs = ts + ci * CHUNK_Q1;
        int cl = min(cs + CHUNK_Q1, te) - cs;
        int cur_buf = ci & 1, nxt_buf = 1 - cur_buf;
        if (ci + 1 < num_chunks) {
            int ncs = cs + CHUNK_Q1, ncl = min(ncs + CHUNK_Q1, te) - ncs;
            load_q1_async(sm, nxt_buf, ncl, paged_kv, kv_indices, ncs, tid);
        }
        if (ci + 1 < num_chunks) { asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }
        else { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
        __syncthreads();
        #pragma unroll 4
        for (int t = 0; t < cl; t++) {
            uint32_t kw0 = ((const uint32_t*)sm.kbuf[cur_buf][t])[lid * 2];
            uint32_t kw1 = ((const uint32_t*)sm.kbuf[cur_buf][t])[lid * 2 + 1];
            float k0 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&kw0));
            float k1 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&kw0) + 1));
            float k2 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&kw1));
            float k3 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&kw1) + 1));
            float dot = qr[0]*k0 + qr[1]*k1 + qr[2]*k2 + qr[3]*k3;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xFFFFFFFF, dot, o);
            float score = dot * sm_scale_log2;
            float nm = fmaxf(m_val, score), corr = exp2f(m_val - nm), p = exp2f(score - nm);
            uint32_t vw0 = ((const uint32_t*)sm.vbuf[cur_buf][t])[lid * 2];
            uint32_t vw1 = ((const uint32_t*)sm.vbuf[cur_buf][t])[lid * 2 + 1];
            float v0 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&vw0));
            float v1 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&vw0) + 1));
            float v2 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&vw1));
            float v3 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&vw1) + 1));
            oa[0] = oa[0]*corr + p*v0; oa[1] = oa[1]*corr + p*v1;
            oa[2] = oa[2]*corr + p*v2; oa[3] = oa[3]*corr + p*v3;
            l_val = l_val*corr + p; m_val = nm;
        }
        __syncthreads();
    }
    float inv_l = (l_val > 0.0f) ? __fdividef(1.0f, l_val) : 0.0f;
    float lse = (l_val > 0.0f) ? (m_val + __log2f(l_val)) : -INFINITY;
    if (lid == 0) pLSE[sid * NQH + wid] = lse;
    #pragma unroll
    for (int i = 0; i < 4; i++) pO[(sid * NQH + wid) * HD + lid * 4 + i] = oa[i] * inv_l;
}

// Q=1 combine kernel
__global__ void combine_kernel_q1(
    const float* __restrict__ pO, const float* __restrict__ pLSE,
    bf16_t* __restrict__ Out, int nsplits
) {
    int h = blockIdx.x, d = threadIdx.x;
    if (h >= NQH || d >= HD) return;
    
    float mx = -INFINITY;
    for (int s = 0; s < nsplits; s++)
        mx = fmaxf(mx, pLSE[s * NQH + h]);
    
    float so = 0, sw = 0;
    for (int s = 0; s < nsplits; s++) {
        float w = exp2f(pLSE[s * NQH + h] - mx);
        sw += w;
        so += w * pO[(s * NQH + h) * HD + d];
    }
    Out[h * HD + d] = __float2bfloat16(sw > 0 ? __fdividef(so, sw) : 0.0f);
}

// =============================================================================
// PATH A2: Q=1 fused decode+combine (single launch for short seq)
// =============================================================================
__global__ void __launch_bounds__(BLK)
paged_gqa_decode_fused(
    const bf16_t* __restrict__ Q,
    const bf16_t* __restrict__ paged_kv,
    const int* __restrict__ kv_indices,
    float* __restrict__ pO,       // [nsplits, NQH, HD]
    float* __restrict__ pLSE,     // [nsplits, NQH]
    int* __restrict__ counter,    // single counter
    bf16_t* __restrict__ Out,     // [NQH, HD]
    int seq_len, int chunk_size, int nsplits, float sm_scale_log2
) {
    __shared__ SmemQ1 sm;
    
    const int sid = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lid = tid % 32;
    
    const int ts = sid * chunk_size;
    const int te = min(ts + chunk_size, seq_len);
    
    if (ts >= seq_len) {
        if (lid == 0) pLSE[sid * NQH + wid] = -INFINITY;
        for (int d = lid; d < HD; d += 32)
            pO[(sid * NQH + wid) * HD + d] = 0.0f;
    } else {
        float qr[4];
        #pragma unroll
        for (int i = 0; i < 4; i++)
            qr[i] = __bfloat162float(Q[wid * HD + lid * 4 + i]);
        
        float m_val = -INFINITY, l_val = 0.0f;
        float oa[4] = {0, 0, 0, 0};
        
        for (int cs = ts; cs < te; cs += CHUNK_Q1) {
            int ce = min(cs + CHUNK_Q1, te);
            int cl = ce - cs;
            
            {
                const int pl0 = cs / PS;
                const int tip0 = cs % PS;
                const int pp0 = __ldg(kv_indices + pl0);
                const bf16_t* kbase0 = paged_kv + (int64_t)pp0 * KVS_PAGE + (int64_t)tip0 * KVS_TOK;
                const bf16_t* vbase0 = kbase0 + KVS_KV;
                
                const int page_remain = PS - tip0;
                const bool crosses = (cl > page_remain);
                
                const int total_u4 = cl * (HD / 8);
                for (int idx = tid; idx < total_u4; idx += BLK) {
                    int tok_local = idx / (HD / 8);
                    int u4_idx = idx % (HD / 8);
                    
                    const bf16_t* kp;
                    const bf16_t* vp;
                    if (!crosses || tok_local < page_remain) {
                        kp = kbase0 + (int64_t)tok_local * KVS_TOK;
                        vp = vbase0 + (int64_t)tok_local * KVS_TOK;
                    } else {
                        int pp1 = __ldg(kv_indices + pl0 + 1);
                        int tip1 = tok_local - page_remain;
                        kp = paged_kv + (int64_t)pp1 * KVS_PAGE + (int64_t)tip1 * KVS_TOK;
                        vp = kp + KVS_KV;
                    }
                    ((uint4*)sm.kbuf[tok_local])[u4_idx] = __ldg(((const uint4*)kp) + u4_idx);
                    ((uint4*)sm.vbuf[tok_local])[u4_idx] = __ldg(((const uint4*)vp) + u4_idx);
                }
            }
            __syncthreads();
            
            {
                #pragma unroll 4
                for (int t = 0; t < cl; t++) {
                    uint32_t kw0 = ((const uint32_t*)sm.kbuf[t])[lid * 2];
                    uint32_t kw1 = ((const uint32_t*)sm.kbuf[t])[lid * 2 + 1];
                    float k0 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&kw0));
                    float k1 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&kw0) + 1));
                    float k2 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&kw1));
                    float k3 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&kw1) + 1));
                    
                    float dot = qr[0]*k0 + qr[1]*k1 + qr[2]*k2 + qr[3]*k3;
                    #pragma unroll
                    for (int o = 16; o > 0; o >>= 1)
                        dot += __shfl_xor_sync(0xFFFFFFFF, dot, o);
                    float score = dot * sm_scale_log2;
                    
                    float nm = fmaxf(m_val, score);
                    float corr = exp2f(m_val - nm);
                    float p = exp2f(score - nm);
                    
                    uint32_t vw0 = ((const uint32_t*)sm.vbuf[t])[lid * 2];
                    uint32_t vw1 = ((const uint32_t*)sm.vbuf[t])[lid * 2 + 1];
                    float v0 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&vw0));
                    float v1 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&vw0) + 1));
                    float v2 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&vw1));
                    float v3 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&vw1) + 1));
                    
                    oa[0] = oa[0] * corr + p * v0;
                    oa[1] = oa[1] * corr + p * v1;
                    oa[2] = oa[2] * corr + p * v2;
                    oa[3] = oa[3] * corr + p * v3;
                    l_val = l_val * corr + p;
                    m_val = nm;
                }
            }
            __syncthreads();
        }
        
        float inv_l = (l_val > 0.0f) ? __fdividef(1.0f, l_val) : 0.0f;
        float lse = (l_val > 0.0f) ? (m_val + __log2f(l_val)) : -INFINITY;
        
        if (lid == 0) pLSE[sid * NQH + wid] = lse;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            pO[(sid * NQH + wid) * HD + lid * 4 + i] = oa[i] * inv_l;
    }
    
    // Fused combine: last CTA writes final bf16 output
    __threadfence();
    
    __shared__ int is_last;
    if (tid == 0) {
        int old = atomicAdd(counter, 1);
        is_last = (old == nsplits - 1);
    }
    __syncthreads();
    
    if (is_last) {
        int h = wid;
        float mx = -INFINITY;
        for (int s = 0; s < nsplits; s++)
            mx = fmaxf(mx, pLSE[s * NQH + h]);
        
        float out_val[4] = {0, 0, 0, 0};
        float total_w = 0;
        for (int s = 0; s < nsplits; s++) {
            float w = exp2f(pLSE[s * NQH + h] - mx);
            total_w += w;
            #pragma unroll
            for (int i = 0; i < 4; i++)
                out_val[i] += w * pO[(s * NQH + h) * HD + lid * 4 + i];
        }
        float inv_w = (total_w > 0) ? __fdividef(1.0f, total_w) : 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            Out[h * HD + lid * 4 + i] = __float2bfloat16(out_val[i] * inv_w);
        
        if (tid == 0) *counter = 0;
    }
}

// =============================================================================
// PATH B: Q>1 multitoken split kernel + separate combine (v007 design)
// =============================================================================
static constexpr int CHUNK_MT = 64;

struct SmemMT {
    bf16_t kbuf[CHUNK_MT][HD];
    bf16_t vbuf[CHUNK_MT][HD];
};

// Double-buffered SMEM for software pipelining
struct SmemMT_DB {
    bf16_t kbuf[2][CHUNK_MT][HD];  // 2 buffers for K
    bf16_t vbuf[2][CHUNK_MT][HD];  // 2 buffers for V
};

// Helper: issue cp.async loads for a chunk into SMEM buffer slot
__device__ __forceinline__ void load_kv_async(
    SmemMT_DB& sm, int buf, int cl,
    const bf16_t* __restrict__ paged_kv,
    const int* __restrict__ kv_indices,
    int cs, int tid
) {
    const int pl0 = cs / PS;
    const int tip0 = cs % PS;
    const int pp0 = __ldg(kv_indices + pl0);
    const bf16_t* kbase0 = paged_kv + (int64_t)pp0 * KVS_PAGE + (int64_t)tip0 * KVS_TOK;
    const bf16_t* vbase0 = kbase0 + KVS_KV;
    
    const int total_u4 = cl * (HD / 8);
    for (int idx = tid; idx < total_u4; idx += BLK) {
        int tok_local = idx / (HD / 8);
        int u4_idx = idx % (HD / 8);
        const bf16_t* kp = kbase0 + (int64_t)tok_local * KVS_TOK;
        const bf16_t* vp = vbase0 + (int64_t)tok_local * KVS_TOK;
        // Use cp.async for 16-byte copies from global to shared
        asm volatile(
            "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
            :: "l"((uint4*)&sm.kbuf[buf][tok_local][u4_idx*8]),
               "l"((const uint4*)kp + u4_idx)
            : "memory"
        );
        asm volatile(
            "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
            :: "l"((uint4*)&sm.vbuf[buf][tok_local][u4_idx*8]),
               "l"((const uint4*)vp + u4_idx)
            : "memory"
        );
    }
    // Commit the async group
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// Helper: wait for async group to complete
__device__ __forceinline__ void wait_async_group(int n) {
    if (n == 0) {
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    } else {
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    }
}

// Helper: compute attention for one chunk from SMEM buffer
template<int QL>
__device__ __forceinline__ void compute_chunk(
    SmemMT_DB& sm, int buf, int cs, int cl,
    float qr[QL][4], float m_val[QL], float l_val[QL], float oa[QL][4],
    const int causal_end[QL], float sm_scale_log2, int lid
) {
    int min_ecl = max(min(cs + cl, causal_end[0]) - cs, 0);
    int max_ecl = max(min(cs + cl, causal_end[QL-1]) - cs, 0);
    
    #pragma unroll 2
    for (int t = 0; t < min_ecl; t++) {
        uint32_t kw0 = ((const uint32_t*)sm.kbuf[buf][t])[lid * 2];
        uint32_t kw1 = ((const uint32_t*)sm.kbuf[buf][t])[lid * 2 + 1];
        float k0 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&kw0));
        float k1 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&kw0) + 1));
        float k2 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&kw1));
        float k3 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&kw1) + 1));
        
        uint32_t vw0 = ((const uint32_t*)sm.vbuf[buf][t])[lid * 2];
        uint32_t vw1 = ((const uint32_t*)sm.vbuf[buf][t])[lid * 2 + 1];
        float v0 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&vw0));
        float v1 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&vw0) + 1));
        float v2 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&vw1));
        float v3 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&vw1) + 1));
        
        #pragma unroll
        for (int q = 0; q < QL; q++) {
            float dot = qr[q][0]*k0 + qr[q][1]*k1 + qr[q][2]*k2 + qr[q][3]*k3;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, o);
            float score = dot * sm_scale_log2;
            float nm = fmaxf(m_val[q], score);
            float corr = exp2f(m_val[q] - nm);
            float p = exp2f(score - nm);
            oa[q][0] = oa[q][0] * corr + p * v0;
            oa[q][1] = oa[q][1] * corr + p * v1;
            oa[q][2] = oa[q][2] * corr + p * v2;
            oa[q][3] = oa[q][3] * corr + p * v3;
            l_val[q] = l_val[q] * corr + p;
            m_val[q] = nm;
        }
    }
    
    for (int t = min_ecl; t < max_ecl; t++) {
        uint32_t kw0 = ((const uint32_t*)sm.kbuf[buf][t])[lid * 2];
        uint32_t kw1 = ((const uint32_t*)sm.kbuf[buf][t])[lid * 2 + 1];
        float k0 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&kw0));
        float k1 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&kw0) + 1));
        float k2 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&kw1));
        float k3 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&kw1) + 1));
        
        uint32_t vw0 = ((const uint32_t*)sm.vbuf[buf][t])[lid * 2];
        uint32_t vw1 = ((const uint32_t*)sm.vbuf[buf][t])[lid * 2 + 1];
        float v0 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&vw0));
        float v1 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&vw0) + 1));
        float v2 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&vw1));
        float v3 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&vw1) + 1));
        
        #pragma unroll
        for (int q = 0; q < QL; q++) {
            if (cs + t >= causal_end[q]) continue;
            float dot = qr[q][0]*k0 + qr[q][1]*k1 + qr[q][2]*k2 + qr[q][3]*k3;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, o);
            float score = dot * sm_scale_log2;
            float nm = fmaxf(m_val[q], score);
            float corr = exp2f(m_val[q] - nm);
            float p = exp2f(score - nm);
            oa[q][0] = oa[q][0] * corr + p * v0;
            oa[q][1] = oa[q][1] * corr + p * v1;
            oa[q][2] = oa[q][2] * corr + p * v2;
            oa[q][3] = oa[q][3] * corr + p * v3;
            l_val[q] = l_val[q] * corr + p;
            m_val[q] = nm;
        }
    }
}

template<int QL>
__global__ void __launch_bounds__(BLK)
paged_gqa_multitoken(
    const bf16_t* __restrict__ Q,       // [QL, NQH, HD]
    const bf16_t* __restrict__ paged_kv,
    const int* __restrict__ kv_indices,
    float* __restrict__ pO,             // [nsplits, QL, NQH, HD]
    float* __restrict__ pLSE,           // [nsplits, QL, NQH]
    int seq_len, int chunk_size, float sm_scale_log2,
    int q_len
) {
    __shared__ SmemMT_DB sm;
    
    const int sid = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lid = tid % 32;
    
    const int ts = sid * chunk_size;
    const int te = min(ts + chunk_size, seq_len);
    
    int causal_end[QL];
    #pragma unroll
    for (int q = 0; q < QL; q++)
        causal_end[q] = seq_len - q_len + 1 + q;
    
    if (ts >= seq_len) {
        #pragma unroll
        for (int q = 0; q < QL; q++) {
            if (lid == 0) pLSE[(sid * QL + q) * NQH + wid] = -INFINITY;
            for (int d = lid; d < HD; d += 32)
                pO[((sid * QL + q) * NQH + wid) * HD + d] = 0.0f;
        }
        return;
    }
    
    float qr[QL][4];
    #pragma unroll
    for (int q = 0; q < QL; q++) {
        #pragma unroll
        for (int i = 0; i < 4; i++)
            qr[q][i] = __bfloat162float(Q[(q * NQH + wid) * HD + lid * 4 + i]);
    }
    
    float m_val[QL], l_val[QL];
    float oa[QL][4];
    #pragma unroll
    for (int q = 0; q < QL; q++) {
        m_val[q] = -INFINITY;
        l_val[q] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            oa[q][i] = 0.0f;
    }
    
    // Software pipelined loop with cp.async double buffering
    int num_chunks = (te - ts + CHUNK_MT - 1) / CHUNK_MT;
    
    // Prefetch first chunk into buffer 0
    if (num_chunks > 0) {
        int cs0 = ts;
        int cl0 = min(cs0 + CHUNK_MT, te) - cs0;
        load_kv_async(sm, 0, cl0, paged_kv, kv_indices, cs0, tid);
    }
    
    for (int ci = 0; ci < num_chunks; ci++) {
        int cs = ts + ci * CHUNK_MT;
        int cl = min(cs + CHUNK_MT, te) - cs;
        int cur_buf = ci & 1;
        int nxt_buf = 1 - cur_buf;
        
        // Prefetch next chunk (if any) into the other buffer
        if (ci + 1 < num_chunks) {
            int ncs = cs + CHUNK_MT;
            int ncl = min(ncs + CHUNK_MT, te) - ncs;
            load_kv_async(sm, nxt_buf, ncl, paged_kv, kv_indices, ncs, tid);
        }
        
        // Wait for current chunk's loads to complete
        wait_async_group(ci + 1 < num_chunks ? 1 : 0);
        __syncthreads();
        
        // Compute on current buffer
        compute_chunk<QL>(sm, cur_buf, cs, cl, qr, m_val, l_val, oa, causal_end, sm_scale_log2, lid);
        
        __syncthreads();
    }
    
    #pragma unroll
    for (int q = 0; q < QL; q++) {
        float inv_l = (l_val[q] > 0.0f) ? __fdividef(1.0f, l_val[q]) : 0.0f;
        float lse = (l_val[q] > 0.0f) ? (m_val[q] + __log2f(l_val[q])) : -INFINITY;
        if (lid == 0) pLSE[(sid * QL + q) * NQH + wid] = lse;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            pO[((sid * QL + q) * NQH + wid) * HD + lid * 4 + i] = oa[q][i] * inv_l;
    }
}

// Q>1 combine kernel (separate launch)
template<int QL>
__global__ void combine_kernel_mt(
    const float* __restrict__ pO, const float* __restrict__ pLSE,
    bf16_t* __restrict__ Out, int nsplits
) {
    int qh = blockIdx.x;
    int q = qh / NQH;
    int h = qh % NQH;
    int d = threadIdx.x;
    if (q >= QL || h >= NQH || d >= HD) return;
    
    float mx = -INFINITY;
    for (int s = 0; s < nsplits; s++)
        mx = fmaxf(mx, pLSE[(s * QL + q) * NQH + h]);
    
    float so = 0, sw = 0;
    for (int s = 0; s < nsplits; s++) {
        float w = exp2f(pLSE[(s * QL + q) * NQH + h] - mx);
        sw += w;
        so += w * pO[((s * QL + q) * NQH + h) * HD + d];
    }
    Out[(q * NQH + h) * HD + d] = __float2bfloat16(sw > 0 ? __fdividef(so, sw) : 0.0f);
}

// =============================================================================
// PATH B2: Q>1 per-row decode for long sequences (high SM utilization)
// Grid: (nsplits, q_len) — each CTA handles one q_row, one KV split
// =============================================================================
__global__ void __launch_bounds__(BLK)
paged_gqa_mt_perrow(
    const bf16_t* __restrict__ Q,       // [q_len, NQH, HD]
    const bf16_t* __restrict__ paged_kv,
    const int* __restrict__ kv_indices,
    float* __restrict__ pO,             // [nsplits, q_len, NQH, HD]
    float* __restrict__ pLSE,           // [nsplits, q_len, NQH]
    int seq_len, int chunk_size, float sm_scale_log2,
    int q_len
) {
    __shared__ SmemQ1 sm;
    
    const int sid = blockIdx.x;
    const int q_row = blockIdx.y;
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lid = tid % 32;
    
    const int causal_end = seq_len - q_len + 1 + q_row;
    const int ts = sid * chunk_size;
    const int te = min(ts + chunk_size, causal_end);
    
    const int partial_idx = sid * q_len + q_row;
    
    if (ts >= te) {
        if (lid == 0) pLSE[partial_idx * NQH + wid] = -INFINITY;
        for (int d = lid; d < HD; d += 32)
            pO[(partial_idx * NQH + wid) * HD + d] = 0.0f;
        return;
    }
    
    // Load Q for this q_row
    float qr[4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        qr[i] = __bfloat162float(Q[(q_row * NQH + wid) * HD + lid * 4 + i]);
    
    float m_val = -INFINITY, l_val = 0.0f;
    float oa[4] = {0, 0, 0, 0};
    
    for (int cs = ts; cs < te; cs += CHUNK_Q1) {
        int ce = min(cs + CHUNK_Q1, te);
        int cl = ce - cs;
        
        // Load K,V to SMEM — CHUNK_Q1=32 divides PS=4096, no page crossing
        {
            const int pl0 = cs / PS;
            const int tip0 = cs % PS;
            const int pp0 = __ldg(kv_indices + pl0);
            const bf16_t* kbase0 = paged_kv + (int64_t)pp0 * KVS_PAGE + (int64_t)tip0 * KVS_TOK;
            const bf16_t* vbase0 = kbase0 + KVS_KV;
            
            const int total_u4 = cl * (HD / 8);
            for (int idx = tid; idx < total_u4; idx += BLK) {
                int tok_local = idx / (HD / 8);
                int u4_idx = idx % (HD / 8);
                const bf16_t* kp = kbase0 + (int64_t)tok_local * KVS_TOK;
                const bf16_t* vp = vbase0 + (int64_t)tok_local * KVS_TOK;
                ((uint4*)sm.kbuf[tok_local])[u4_idx] = __ldg(((const uint4*)kp) + u4_idx);
                ((uint4*)sm.vbuf[tok_local])[u4_idx] = __ldg(((const uint4*)vp) + u4_idx);
            }
        }
        __syncthreads();
        
        // Per-token online softmax + PV
        {
            #pragma unroll 4
            for (int t = 0; t < cl; t++) {
                uint32_t kw0 = ((const uint32_t*)sm.kbuf[t])[lid * 2];
                uint32_t kw1 = ((const uint32_t*)sm.kbuf[t])[lid * 2 + 1];
                float k0 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&kw0));
                float k1 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&kw0) + 1));
                float k2 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&kw1));
                float k3 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&kw1) + 1));
                
                float dot = qr[0]*k0 + qr[1]*k1 + qr[2]*k2 + qr[3]*k3;
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    dot += __shfl_xor_sync(0xFFFFFFFF, dot, o);
                float score = dot * sm_scale_log2;
                
                float nm = fmaxf(m_val, score);
                float corr = exp2f(m_val - nm);
                float p = exp2f(score - nm);
                
                uint32_t vw0 = ((const uint32_t*)sm.vbuf[t])[lid * 2];
                uint32_t vw1 = ((const uint32_t*)sm.vbuf[t])[lid * 2 + 1];
                float v0 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&vw0));
                float v1 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&vw0) + 1));
                float v2 = __bfloat162float(*reinterpret_cast<const bf16_t*>(&vw1));
                float v3 = __bfloat162float(*(reinterpret_cast<const bf16_t*>(&vw1) + 1));
                
                oa[0] = oa[0] * corr + p * v0;
                oa[1] = oa[1] * corr + p * v1;
                oa[2] = oa[2] * corr + p * v2;
                oa[3] = oa[3] * corr + p * v3;
                l_val = l_val * corr + p;
                m_val = nm;
            }
        }
        __syncthreads();
    }
    
    float inv_l = (l_val > 0.0f) ? __fdividef(1.0f, l_val) : 0.0f;
    float lse = (l_val > 0.0f) ? (m_val + __log2f(l_val)) : -INFINITY;
    
    if (lid == 0) pLSE[partial_idx * NQH + wid] = lse;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        pO[(partial_idx * NQH + wid) * HD + lid * 4 + i] = oa[i] * inv_l;
}

// Q>1 per-row combine kernel
__global__ void combine_kernel_mt_perrow(
    const float* __restrict__ pO, const float* __restrict__ pLSE,
    bf16_t* __restrict__ Out, int nsplits, int q_len
) {
    int qh = blockIdx.x;
    int q = qh / NQH;
    int h = qh % NQH;
    int d = threadIdx.x;
    if (q >= q_len || h >= NQH || d >= HD) return;
    
    float mx = -INFINITY;
    for (int s = 0; s < nsplits; s++)
        mx = fmaxf(mx, pLSE[(s * q_len + q) * NQH + h]);
    
    float so = 0, sw = 0;
    for (int s = 0; s < nsplits; s++) {
        float w = exp2f(pLSE[(s * q_len + q) * NQH + h] - mx);
        sw += w;
        so += w * pO[((s * q_len + q) * NQH + h) * HD + d];
    }
    Out[(q * NQH + h) * HD + d] = __float2bfloat16(sw > 0 ? __fdividef(so, sw) : 0.0f);
}

// =============================================================================
// PATH C: Q>1 fused atomic combine kernel (v009 design)
// Single launch: 2D grid (nsplits, q_len), last CTA per q_row does combine
// =============================================================================
static constexpr int KVC_FUSED = 8;

struct SmemFused {
    bf16_t kbuf[KVC_FUSED][HD];
    bf16_t vbuf[KVC_FUSED][HD];
};

__global__ void __launch_bounds__(BLK)
paged_gqa_fused(
    const bf16_t* __restrict__ Q,
    const bf16_t* __restrict__ paged_kv,
    const int* __restrict__ kv_indices,
    float* __restrict__ pO,
    float* __restrict__ pLSE,
    int* __restrict__ counters,
    bf16_t* __restrict__ Out,
    int seq_len, int q_len, int nsplits, float sm_scale_log2
) {
    __shared__ SmemFused sm;
    
    const int sid = blockIdx.x;    // split index
    const int q_row = blockIdx.y;  // query row
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lid = tid % 32;
    
    const int partial_idx = q_row * nsplits + sid;
    const int ts = sid * KVC_FUSED;
    const int causal_end = seq_len - q_len + 1 + q_row;
    const int te = min(min(ts + KVC_FUSED, seq_len), causal_end);
    
    if (ts >= te) {
        if (lid == 0) pLSE[partial_idx * NQH + wid] = -INFINITY;
        for (int d = lid; d < HD; d += 32)
            pO[(partial_idx * NQH + wid) * HD + d] = 0.0f;
    } else {
        float qr[4];
        #pragma unroll
        for (int i = 0; i < 4; i++)
            qr[i] = __bfloat162float(Q[q_row * NQH * HD + wid * HD + lid * 4 + i]);
        
        float m_val = -INFINITY, l_val = 0.0f;
        float oa[4] = {0, 0, 0, 0};
        
        // Load KV chunk to SMEM (with page lookup)
        {
            const int cl = te - ts;
            const int page_idx = ts / PS;
            const int tip = ts % PS;
            const int pp = __ldg(kv_indices + page_idx);
            const bf16_t* kbase = paged_kv + (int64_t)pp * KVS_PAGE + (int64_t)tip * KVS_TOK;
            const bf16_t* vbase = kbase + KVS_KV;
            
            const int u4_per_tok = HD / 8;
            const int total_u4 = cl * u4_per_tok;
            for (int idx = tid; idx < total_u4; idx += BLK) {
                int tok = idx / u4_per_tok;
                int u4i = idx % u4_per_tok;
                ((uint4*)sm.kbuf[tok])[u4i] = __ldg(((const uint4*)(kbase + (int64_t)tok * KVS_TOK)) + u4i);
                ((uint4*)sm.vbuf[tok])[u4i] = __ldg(((const uint4*)(vbase + (int64_t)tok * KVS_TOK)) + u4i);
            }
        }
        __syncthreads();
        
        {
            const int cl = te - ts;
            #pragma unroll
            for (int t = 0; t < KVC_FUSED; t++) {
                if (t >= cl) break;
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    dot += qr[i] * __bfloat162float(sm.kbuf[t][lid * 4 + i]);
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    dot += __shfl_xor_sync(0xFFFFFFFF, dot, o);
                float score = dot * sm_scale_log2;
                float nm = fmaxf(m_val, score);
                float corr = exp2f(m_val - nm);
                float p = exp2f(score - nm);
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    oa[i] = oa[i] * corr + p * __bfloat162float(sm.vbuf[t][lid * 4 + i]);
                l_val = l_val * corr + p;
                m_val = nm;
            }
        }
        
        float inv_l = (l_val > 0.0f) ? __fdividef(1.0f, l_val) : 0.0f;
        float lse = (l_val > 0.0f) ? (m_val + __log2f(l_val)) : -INFINITY;
        
        if (lid == 0) pLSE[partial_idx * NQH + wid] = lse;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            pO[(partial_idx * NQH + wid) * HD + lid * 4 + i] = oa[i] * inv_l;
    }
    
    __threadfence();
    
    __shared__ int is_last;
    if (tid == 0) {
        int old = atomicAdd(&counters[q_row], 1);
        is_last = (old == nsplits - 1);
    }
    __syncthreads();
    
    if (is_last) {
        int h = wid;
        float mx = -INFINITY;
        for (int s = 0; s < nsplits; s++) {
            int pi = q_row * nsplits + s;
            mx = fmaxf(mx, pLSE[pi * NQH + h]);
        }
        
        float out_val[4] = {0, 0, 0, 0};
        float total_w = 0;
        for (int s = 0; s < nsplits; s++) {
            int pi = q_row * nsplits + s;
            float w = exp2f(pLSE[pi * NQH + h] - mx);
            total_w += w;
            #pragma unroll
            for (int i = 0; i < 4; i++)
                out_val[i] += w * pO[(pi * NQH + h) * HD + lid * 4 + i];
        }
        float inv_w = (total_w > 0) ? __fdividef(1.0f, total_w) : 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            Out[q_row * NQH * HD + h * HD + lid * 4 + i] = __float2bfloat16(out_val[i] * inv_w);
        
        if (tid == 0) counters[q_row] = 0;
    }
}

// =============================================================================
// CPU reference
// =============================================================================
void cpu_ref(const bf16_t* Q, const bf16_t* kv, const int* ki,
             int q_len, int sl, float sms, float* out) {
    for (int qi = 0; qi < q_len; qi++) {
        int ce = (q_len == 1) ? sl : (sl - q_len + 1 + qi);
        for (int h = 0; h < NQH; h++) {
            float q[HD];
            for (int d = 0; d < HD; d++)
                q[d] = __bfloat162float(Q[qi * NQH * HD + h * HD + d]);
            float m = -INFINITY, l = 0;
            float o[HD] = {};
            for (int t = 0; t < ce; t++) {
                int pl = t / PS, tip = t % PS, pp = ki[pl];
                const bf16_t* k = kv + (int64_t)pp * KVS_PAGE + (int64_t)tip * KVS_TOK;
                float dot = 0;
                for (int d = 0; d < HD; d++) dot += q[d] * __bfloat162float(k[d]);
                float sc = dot * sms, nm = fmaxf(m, sc), c = expf(m - nm), p = expf(sc - nm);
                l = l * c + p;
                const bf16_t* v = kv + (int64_t)pp * KVS_PAGE + KVS_KV + (int64_t)tip * KVS_TOK;
                for (int d = 0; d < HD; d++) o[d] = o[d] * c + p * __bfloat162float(v[d]);
                m = nm;
            }
            float il = l > 0 ? 1.0f / l : 0;
            for (int d = 0; d < HD; d++) out[qi * NQH * HD + h * HD + d] = o[d] * il;
        }
    }
}

// =============================================================================
// Host helper
// =============================================================================
void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA(%s):%s\n", m, cudaGetErrorString(e)); exit(1); }
}

double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

struct Config {
    const char* name;
    int q_len;
    int seq_len;
};

double run_config(Config& c, float sms, void* df, size_t fs) {
    int ql = c.q_len, sl = c.seq_len;
    int np = (sl + PS - 1) / PS;
    size_t qs = (size_t)ql * NQH * HD;
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    
    std::vector<bf16_t> hq(qs);
    for (size_t i = 0; i < qs; i++) hq[i] = __float2bfloat16(dist(rng));
    
    size_t kvt = (size_t)np * 2 * PS * NKH * HD;
    std::vector<bf16_t> hkv(kvt);
    for (size_t i = 0; i < kvt; i++) hkv[i] = __float2bfloat16(dist(rng));
    
    std::vector<int> hki(np);
    for (int i = 0; i < np; i++) hki[i] = i;
    
    bf16_t *dq, *dkv, *dout;
    int *dki;
    chk(cudaMalloc(&dq, qs * 2), "");
    chk(cudaMalloc(&dkv, kvt * 2), "");
    chk(cudaMalloc(&dout, qs * 2), "");
    chk(cudaMalloc(&dki, np * 4), "");
    chk(cudaMemcpy(dq, hq.data(), qs * 2, cudaMemcpyHostToDevice), "");
    chk(cudaMemcpy(dkv, hkv.data(), kvt * 2, cudaMemcpyHostToDevice), "");
    chk(cudaMemcpy(dki, hki.data(), np * 4, cudaMemcpyHostToDevice), "");
    
    float sml2 = sms * 1.4426950408889634f;
    
    // Determine which path to use
    // Path A: Q=1 (decode)
    // Path B: Q>1, seq >= 4096 (split + separate combine)  
    // Path C: Q>1, seq < 4096 (fused atomic combine)
    
    // Allocate workspace
    int kvc, ns;
    bool use_fused = false;
    
    if (ql == 1) {
        // Path A: Q=1 decode
        if (sl <= 64) { ns = 1; kvc = sl; }
        else if (sl <= 512) { kvc = 32; ns = (sl + kvc - 1) / kvc; }
        else if (sl <= 4096) { kvc = 64; ns = (sl + kvc - 1) / kvc; }
        else { kvc = 256; ns = (sl + kvc - 1) / kvc; }  // seq32k: kvc=256, ns=128 (now with cp.async)
    } else if (sl <= 128) {
        // Path C: fused atomic combine (fast for short seq)
        use_fused = true;
        kvc = KVC_FUSED;
        ns = (sl + kvc - 1) / kvc;
    } else if (sl >= 8192) {
        // Path B: multitoken - use kvc=256 (fewer CTAs = less SM imbalance)
        kvc = 256;
        ns = (sl + kvc - 1) / kvc;
    } else {
        // Path B: multitoken split + separate combine (seq 512..4096)
        if (sl <= 512) {
            kvc = 32;  // seq512: kvc=32 -> ns=16
        } else {
            kvc = (ql <= 3) ? 64 : 32;
        }
        ns = (sl + kvc - 1) / kvc;
    }
    
    float *dpO = nullptr, *dpL = nullptr;
    int *dctr = nullptr;
    
    bool q1_fused = (ql == 1 && ns > 1 && ns <= 4);  // use fused path for Q=1 seq<=128 only
    
    if (ql == 1) {
        chk(cudaMalloc(&dpO, (size_t)ns * NQH * HD * 4), "");
        chk(cudaMalloc(&dpL, (size_t)ns * NQH * 4), "");
        if (q1_fused) {
            chk(cudaMalloc(&dctr, 4), "");
            chk(cudaMemset(dctr, 0, 4), "");
        }
    } else if (use_fused) {
        size_t total_partials = (size_t)ql * ns;
        chk(cudaMalloc(&dpO, total_partials * NQH * HD * 4), "");
        chk(cudaMalloc(&dpL, total_partials * NQH * 4), "");
        chk(cudaMalloc(&dctr, ql * 4), "");
        chk(cudaMemset(dctr, 0, ql * 4), "");
    } else {
        chk(cudaMalloc(&dpO, (size_t)ns * ql * NQH * HD * 4), "");
        chk(cudaMalloc(&dpL, (size_t)ns * ql * NQH * 4), "");
    }
    
    auto run = [&]() {
        if (ql == 1 && q1_fused) {
            // Single-launch fused decode+combine for Q=1 short seq
            paged_gqa_decode_fused<<<ns, BLK>>>(dq, dkv, dki, dpO, dpL, dctr, dout, sl, kvc, ns, sml2);
        } else if (ql == 1 && sl > 512) {
            // cp.async pipelined decode for long Q=1 sequences
            paged_gqa_decode_async<<<ns, BLK>>>(dq, dkv, dki, dpO, dpL, sl, kvc, sml2);
            combine_kernel_q1<<<NQH, HD>>>(dpO, dpL, dout, ns);
        } else if (ql == 1) {
            paged_gqa_decode<<<ns, BLK>>>(dq, dkv, dki, dpO, dpL, sl, kvc, sml2);
            combine_kernel_q1<<<NQH, HD>>>(dpO, dpL, dout, ns);
        } else if (use_fused) {
            dim3 grid(ns, ql);
            paged_gqa_fused<<<grid, BLK>>>(dq, dkv, dki, dpO, dpL, dctr, dout, sl, ql, ns, sml2);
        } else {
            if (ql == 2) {
                paged_gqa_multitoken<2><<<ns, BLK>>>(dq, dkv, dki, dpO, dpL, sl, kvc, sml2, ql);
                combine_kernel_mt<2><<<ql * NQH, HD>>>(dpO, dpL, dout, ns);
            } else if (ql == 3) {
                paged_gqa_multitoken<3><<<ns, BLK>>>(dq, dkv, dki, dpO, dpL, sl, kvc, sml2, ql);
                combine_kernel_mt<3><<<ql * NQH, HD>>>(dpO, dpL, dout, ns);
            } else {
                paged_gqa_multitoken<4><<<ns, BLK>>>(dq, dkv, dki, dpO, dpL, sl, kvc, sml2, ql);
                combine_kernel_mt<4><<<ql * NQH, HD>>>(dpO, dpL, dout, ns);
            }
        }
    };
    
    run();
    chk(cudaDeviceSynchronize(), "sync");
    
    // Validate
    std::vector<float> ref(qs);
    cpu_ref(hq.data(), hkv.data(), hki.data(), ql, sl, sms, ref.data());
    std::vector<bf16_t> gout(qs);
    chk(cudaMemcpy(gout.data(), dout, qs * 2, cudaMemcpyDeviceToHost), "");
    
    float mre = 0, mae = 0, sae = 0;
    for (size_t i = 0; i < qs; i++) {
        float g = __bfloat162float(gout[i]), r = ref[i];
        float ae = fabsf(g - r), re = ae / fmaxf(fabsf(r), 1e-6f);
        mre = fmaxf(mre, re);
        mae = fmaxf(mae, ae);
        sae += ae;
    }
    float meae = sae / qs;
    bool valid = (mre < 1e-2f) && (mae < 1.0f) && (meae < 1e-2f);
    fprintf(stderr, "%s (Q=%d,seq=%d): mre=%.6f mae=%.6f meae=%.6f %s\n",
            c.name, ql, sl, mre, mae, meae, valid ? "PASS" : "INVALID");
    
    if (!valid) {
        fprintf(stderr, "INVALID\n");
        cudaFree(dq); cudaFree(dkv); cudaFree(dout); cudaFree(dki);
        if (dpO) cudaFree(dpO);
        if (dpL) cudaFree(dpL);
        if (dctr) cudaFree(dctr);
        return 0.0;
    }
    
    // Warmup >= 2 seconds
    double t0 = get_time_sec();
    while (get_time_sec() - t0 < 2.0) {
        run();
        cudaDeviceSynchronize();
    }
    
    int NI = 300;
    std::vector<float> ts_vec(NI);
    cudaEvent_t ev_s, ev_e;
    cudaEventCreate(&ev_s);
    cudaEventCreate(&ev_e);
    for (int it = 0; it < NI; it++) {
        cudaMemset(df, 0, fs);
        cudaEventRecord(ev_s);
        run();
        cudaEventRecord(ev_e);
        cudaEventSynchronize(ev_e);
        float ms;
        cudaEventElapsedTime(&ms, ev_s, ev_e);
        ts_vec[it] = ms;
    }
    cudaEventDestroy(ev_s);
    cudaEventDestroy(ev_e);
    std::sort(ts_vec.begin(), ts_vec.end());
    float med = ts_vec[NI / 2];
    
    double total_attended;
    if (ql == 1) {
        total_attended = (double)sl;
    } else {
        total_attended = (double)ql * sl - (double)ql * (ql - 1) / 2.0;
    }
    double flops = 4.0 * NQH * total_attended * HD;
    double tflops = flops / (med / 1000.0) / 1e12;
    fprintf(stderr, "%s: %.4f TFLOPS, %.1f us\n", c.name, tflops, med * 1000);
    
    cudaFree(dq); cudaFree(dkv); cudaFree(dout); cudaFree(dki);
    if (dpO) cudaFree(dpO);
    if (dpL) cudaFree(dpL);
    if (dctr) cudaFree(dctr);
    return tflops;
}

int main() {
    chk(cudaSetDevice(0), "sd");
    float sms = 1.0f / sqrtf((float)HD);
    size_t fs = 128 * 1024 * 1024;
    void* df;
    chk(cudaMalloc(&df, fs), "f");
    
    Config configs[] = {
        {"q1_seq128",  1,   128},
        {"q1_seq512",  1,   512},
        {"q1_seq4k",   1,  4096},
        {"q1_seq32k",  1, 32768},
        {"q2_seq128",  2,   128},
        {"q2_seq512",  2,   512},
        {"q2_seq4k",   2,  4096},
        {"q2_seq32k",  2, 32768},
        {"q3_seq128",  3,   128},
        {"q3_seq512",  3,   512},
        {"q3_seq4k",   3,  4096},
        {"q3_seq32k",  3, 32768},
        {"q4_seq128",  4,   128},
        {"q4_seq512",  4,   512},
        {"q4_seq4k",   4,  4096},
        {"q4_seq32k",  4, 32768},
    };
    int ncfg = sizeof(configs) / sizeof(configs[0]);
    
    double results[16] = {};
    for (int ci = 0; ci < ncfg; ci++) {
        results[ci] = run_config(configs[ci], sms, df, fs);
    }
    
    // Print KERNEL_RESULT
    printf("KERNEL_RESULT {");
    for (int i = 0; i < ncfg; i++) {
        if (i) printf(", ");
        printf("\"%s\": %.4f", configs[i].name, results[i]);
    }
    printf("}\n");
    fflush(stdout);
    
    // Run FlashInfer baseline
    fflush(stderr);
    FILE* bp = popen("python3 ../baselines/paged-gqa-fused-qwen3/baseline.py 2>/dev/null", "r");
    double rv[16] = {};
    if (bp) {
        char line[1024];
        while (fgets(line, sizeof(line), bp)) {
            for (int i = 0; i < ncfg; i++) {
                if (strstr(line, configs[i].name)) {
                    // Parse "name: 0.123 TFLOPS ..."
                    char* p = strstr(line, configs[i].name);
                    if (p) {
                        p += strlen(configs[i].name);
                        // Skip to colon
                        while (*p && *p != ':') p++;
                        if (*p == ':') {
                            p++;
                            while (*p == ' ') p++;
                            double v = atof(p);
                            if (v > 0.001) rv[i] = v;
                        }
                    }
                }
            }
        }
        pclose(bp);
    }
    printf("KERNEL_RESULT_REFERENCE {");
    for (int i = 0; i < ncfg; i++) {
        if (i) printf(", ");
        printf("\"%s\": %.4f", configs[i].name, rv[i]);
    }
    printf("}\n");
    fflush(stdout);
    
    cudaFree(df);
    return 0;
}
