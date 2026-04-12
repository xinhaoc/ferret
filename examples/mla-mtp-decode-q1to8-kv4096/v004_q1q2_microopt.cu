// MLA Multi-Token Decode for DeepSeek V3 on B200 (SM100a)
// Compile: nvcc -O3 --use_fast_math -std=c++17 -gencode arch=compute_100a,code=sm_100a -o kernel kernel.cu -lcuda
// Supports Q_LEN=1,2,4,8 with split-K parallelism
// Based on v024 single-token decode architecture with multi-query extension
//
// Key idea: M=128 rows in MMA map to Q_LEN queries × (128/Q_LEN) heads per block
// KV is shared across all queries within a block (loaded once from SMEM)
// Grid: (num_head_groups × split_k, batch)
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static constexpr int NUM_HEADS = 128;
static constexpr int D_K = 576;
static constexpr int D_V = 512;
static constexpr int TILE_S = 128;     // KV tokens per tile
static constexpr int BK = 64;          // 128B swizzle tile width
static constexpr int MMA_K = 16;
static constexpr int K_ITERS = D_K / BK; // 9 for QK
static constexpr int V_CHUNKS = D_V / BK; // 8
static constexpr int TB = 128;

// Pipeline stages
static constexpr int NUM_QK_STAGES = 5;
static constexpr int NUM_PV_STAGES = 3;
static constexpr int MAX_STAGES = 5;

// SMEM tile: 128 rows × BK cols × 2 bytes = 16384
static constexpr int TILE_BYTES = 128 * BK * 2;  // always 128 rows for MMA M dim

// ============ PTX Helpers ============
__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t p = 0;
    asm volatile(
        "{\n\t.reg .pred %%px;\n\t"
        "elect.sync _|%%px, 0xFFFFFFFF;\n\t"
        "@%%px mov.s32 %0, 1;\n\t}"
        : "+r"(p));
    return p;
}

__device__ __forceinline__ void mbar_init(int addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count));
}

__device__ __forceinline__ void mbar_wait(int addr, int phase) {
    asm volatile(
        "{\n\t.reg .pred P;\n\t"
        "WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1, 0x989680;\n\t"
        "@P bra DONE;\n\t"
        "bra WAIT;\n\t"
        "DONE:\n\t}"
        :: "r"(addr), "r"(phase));
}

__device__ __forceinline__ void mbar_tx(int addr, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(addr), "r"(bytes) : "memory");
}

__device__ __forceinline__ constexpr uint64_t desc_enc(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4;
}

__device__ __forceinline__ uint64_t make_desc(int smem_addr) {
    constexpr uint64_t SBO = 8ULL * 128;  // stride between outer (8 rows × 128 bytes)
    return desc_enc(smem_addr) | (desc_enc(SBO) << 32) | (1ULL << 46) | (2ULL << 61);
}

__device__ __forceinline__ void tcgen05_mma(int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, int acc) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(acc));
}

__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(mbar_addr) : "memory");
}

// No reshape kernel — Q loaded directly from [Q_LEN, NUM_HEADS, D_K] layout via TMA

// ============ Main MLA Kernel ============
template<bool SINGLE_TILE>
__global__ __launch_bounds__(TB)
void mla_kernel(
    const __grid_constant__ CUtensorMap Q_tm,
    const __grid_constant__ CUtensorMap KV_tm,
    nv_bfloat16* __restrict__ Oa, float* __restrict__ La,
    float ss, int kv_len, int sk, int num_head_groups, int Q_LEN
) {
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    
    // Decompose blockIdx.x into (head_group, split_id)
    const int gi = blockIdx.x / sk;   // head group index
    const int si = blockIdx.x % sk;   // split index
    const int bi = blockIdx.y;         // batch index
    
    if (gi >= num_head_groups) return;
    
    const int kvt = (kv_len + TILE_S - 1) / TILE_S;
    const int tps = (kvt + sk - 1) / sk;
    const int t0 = si * tps;
    const int t1 = min(t0 + tps, kvt);
    if (t0 >= t1) return;
    
    const int hpb = NUM_HEADS / num_head_groups;  // heads per block
    
    // Causal limit per query: for query q, max KV position = kv_len - Q_LEN + q
    // Row r in block: q = r / hpb, h = r % hpb
    // Only apply causal when Q_LEN > 1
    
    // ===== SMEM layout =====
    extern __shared__ __align__(1024) char smem_buf[];
    const int smem_base = __cvta_generic_to_shared(smem_buf);
    
    // Q-K co-loading: each QK stage holds Q[ki]+K[ki] side-by-side (2*TILE_BYTES per stage)
    // This eliminates the serial Q preload phase, saving ~1.5us
    const int work_smem = smem_base;  // No separate Q area
    
    __shared__ uint64_t mbar_buf[12];
    __shared__ int tmem_addr_buf[1];
    const int tma_bar = __cvta_generic_to_shared(&mbar_buf[0]);
    const int mma_bar = __cvta_generic_to_shared(&mbar_buf[MAX_STAGES]);
    const int mainloop_bar = __cvta_generic_to_shared(&mbar_buf[2 * MAX_STAGES]);
    const int q_bar = __cvta_generic_to_shared(&mbar_buf[2 * MAX_STAGES + 1]);
    
    // Init barriers + alloc TMEM
    if (wid == 0 && elect_sync()) {
        for (int i = 0; i < MAX_STAGES; i++) {
            mbar_init(tma_bar + i * 8, 1);
            mbar_init(mma_bar + i * 8, 1);
        }
        mbar_init(mainloop_bar, 1);
        mbar_init(q_bar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    else if (wid == 1) {
        int addr_smem = __cvta_generic_to_shared(tmem_addr_buf);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(addr_smem), "r"(D_V));
    }
    __syncthreads();
    const int taddr = tmem_addr_buf[0];
    
    // Q is co-loaded with K per QK pipeline stage — no separate preload
    const int hpb_bytes = hpb * BK * 2;  // bytes per Q sub-chunk
    
    // Instruction descriptors (same tile sizes as v024)
    constexpr uint32_t idesc_qk = (1U << 4) | (1U << 7) | (1U << 10)
        | ((uint32_t)(TILE_S >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);
    
    constexpr uint32_t idesc_pv = (1U << 4) | (1U << 7) | (1U << 10)
        | (1U << 16)  // b_major = MN-major
        | ((uint32_t)(BK >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);
    
    // Output pointer: partial results for this (batch, group, split)
    int block_linear = bi * num_head_groups * sk + gi * sk + si;
    nv_bfloat16* Oout = Oa + block_linear * D_V * 128;
    float row_max = -1e30f;
    float row_sum = 0.0f;
    
    // Register buffer to save O[0:127] before QK overwrites TMEM (only for multi-tile)
    float o_save[SINGLE_TILE ? 1 : 128];
    
    // ===== Main loop over KV tiles =====
    for (int tile = t0; tile < t1; tile++) {
        const int kvs = tile * TILE_S;
        const int tlen = min(TILE_S, kv_len - kvs);
        
        // ===== Save O[0:127] from TMEM to registers (tile > t0 only) =====
        if (!SINGLE_TILE && tile > t0) {
            for (int c = 0; c < TILE_S; c += 16) {
                int addr = taddr + (tid << 16) + c;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                    : "=f"(o_save[c+0]), "=f"(o_save[c+1]), "=f"(o_save[c+2]), "=f"(o_save[c+3]),
                      "=f"(o_save[c+4]), "=f"(o_save[c+5]), "=f"(o_save[c+6]), "=f"(o_save[c+7]),
                      "=f"(o_save[c+8]), "=f"(o_save[c+9]), "=f"(o_save[c+10]), "=f"(o_save[c+11]),
                      "=f"(o_save[c+12]), "=f"(o_save[c+13]), "=f"(o_save[c+14]), "=f"(o_save[c+15])
                    : "r"(addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
            }
        }
        
        // ===== QK Phase =====
        __syncthreads();
        if (wid == 0 && elect_sync()) {
            for (int i = 0; i < NUM_QK_STAGES; i++) {
                mbar_init(tma_bar + i * 8, 1);
                mbar_init(mma_bar + i * 8, 1);
            }
            mbar_init(mainloop_bar, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        __syncthreads();
        
        if (wid == 0 && elect_sync()) {
            // TMA producer — co-load Q[ki] + K[ki] per stage
            int phase = 0;
            for (int ki = 0; ki < K_ITERS; ki++) {
                int stage = ki % NUM_QK_STAGES;
                mbar_wait(mma_bar + stage * 8, phase ^ 1);
                if (stage == NUM_QK_STAGES - 1) phase ^= 1;
                
                // Stage layout: [Q_tile(TILE_BYTES)] [K_tile(TILE_BYTES)]
                int q_stage = work_smem + stage * 2 * TILE_BYTES;
                int k_stage = q_stage + TILE_BYTES;
                
                mbar_tx(tma_bar + stage * 8, TILE_BYTES + hpb_bytes * Q_LEN);
                // Load K
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%2, %3, %4}], [%5];"
                    :: "r"(k_stage), "l"(&KV_tm),
                       "r"(0), "r"(bi * kv_len + kvs), "r"(ki), "r"(tma_bar + stage * 8) : "memory");
                // Load Q chunks for this ki
                for (int q = 0; q < Q_LEN; q++) {
                    int global_row = bi * Q_LEN * NUM_HEADS + q * NUM_HEADS + gi * hpb;
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1, {%2, %3, %4}], [%5];"
                        :: "r"(q_stage + q * hpb_bytes), "l"(&Q_tm),
                           "r"(0), "r"(global_row), "r"(ki), "r"(tma_bar + stage * 8) : "memory");
                }
            }
        }
        else if (wid == 1 && elect_sync()) {
            // MMA consumer — reads Q and K from same stage
            int phase = 0;
            for (int ki = 0; ki < K_ITERS; ki++) {
                int stage = ki % NUM_QK_STAGES;
                mbar_wait(tma_bar + stage * 8, phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (stage == NUM_QK_STAGES - 1) phase ^= 1;
                
                int q_stage = work_smem + stage * 2 * TILE_BYTES;
                int k_stage = q_stage + TILE_BYTES;
                
                for (int k2 = 0; k2 < BK / MMA_K; k2++) {
                    uint64_t a_desc = make_desc(q_stage + k2 * 32);
                    uint64_t b_desc = make_desc(k_stage + k2 * 32);
                    tcgen05_mma(taddr, a_desc, b_desc, idesc_qk, (ki == 0 && k2 == 0) ? 0 : 1);
                }
                tcgen05_commit(mma_bar + stage * 8);
            }
            tcgen05_commit(mainloop_bar);
        }
        
        __syncthreads();
        mbar_wait(mainloop_bar, 0);
        
        // ===== Softmax Phase =====
        asm volatile("tcgen05.fence::after_thread_sync;");
        
        // Reinit PV barriers and preload first V chunk
        if (wid == 0 && elect_sync()) {
            for (int i = 0; i < NUM_PV_STAGES; i++) {
                mbar_init(tma_bar + i * 8, 1);
                mbar_init(mma_bar + i * 8, 1);
            }
            mbar_init(mainloop_bar, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
            int v_smem0 = work_smem + 2 * TILE_BYTES;
            mbar_tx(tma_bar, TILE_BYTES);
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                "[%0], [%1, {%2, %3, %4}], [%5];"
                :: "r"(v_smem0), "l"(&KV_tm),
                   "r"(0), "r"(bi * kv_len + kvs), "r"(0), "r"(tma_bar) : "memory");
        }
        
        // === 2-pass TMEM softmax: pass 1 finds max, pass 2 computes exp and writes P ===
        int q_idx = tid / hpb;
        int causal_limit = kv_len;
        if (Q_LEN > 1) {
            causal_limit = kv_len - Q_LEN + q_idx + 1;
        }
        int effective_len = min(tlen, causal_limit - kvs);
        if (effective_len < 0) effective_len = 0;
        if (q_idx >= Q_LEN) effective_len = 0;
        
        int P0_smem = work_smem;
        int P1_smem = work_smem + TILE_BYTES;
        
        // Pass 1: Find global max across all 128 columns
        float tile_max = -1e30f;
        for (int c = 0; c < TILE_S; c += 16) {
            float t16[16];
            int addr = taddr + (tid << 16) + c;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                : "=f"(t16[0]), "=f"(t16[1]), "=f"(t16[2]), "=f"(t16[3]),
                  "=f"(t16[4]), "=f"(t16[5]), "=f"(t16[6]), "=f"(t16[7]),
                  "=f"(t16[8]), "=f"(t16[9]), "=f"(t16[10]), "=f"(t16[11]),
                  "=f"(t16[12]), "=f"(t16[13]), "=f"(t16[14]), "=f"(t16[15])
                : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                float v = (c + i < effective_len) ? t16[i] * ss : -1e30f;
                tile_max = fmaxf(tile_max, v);
            }
        }
        
        // Pass 2: Compute exp(val*ss - tile_max), write P to SMEM, accumulate sum
        float tile_sum = 0.0f;
        for (int half = 0; half < 2; half++) {
            int p_base = (half == 0) ? P0_smem : P1_smem;
            int row_base = p_base + tid * 128;
            
            for (int c = half * 64; c < half * 64 + 64; c += 16) {
                float t16[16];
                int addr = taddr + (tid << 16) + c;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                    : "=f"(t16[0]), "=f"(t16[1]), "=f"(t16[2]), "=f"(t16[3]),
                      "=f"(t16[4]), "=f"(t16[5]), "=f"(t16[6]), "=f"(t16[7]),
                      "=f"(t16[8]), "=f"(t16[9]), "=f"(t16[10]), "=f"(t16[11]),
                      "=f"(t16[12]), "=f"(t16[13]), "=f"(t16[14]), "=f"(t16[15])
                    : "r"(addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    float e = (c + i < effective_len) ? __expf(t16[i] * ss - tile_max) : 0.0f;
                    t16[i] = e;
                    tile_sum += e;
                }
                
                // Write 16 bf16 values to P in SMEM
                int g_start = (c % 64);
                #pragma unroll
                for (int gg = 0; gg < 16; gg += 8) {
                    int g = g_start + gg;
                    uint32_t w0, w1, w2, w3;
                    {
                        nv_bfloat16 b0 = __float2bfloat16(t16[gg+0]);
                        nv_bfloat16 b1 = __float2bfloat16(t16[gg+1]);
                        w0 = (uint32_t)(*(uint16_t*)&b0) | ((uint32_t)(*(uint16_t*)&b1) << 16);
                    }
                    {
                        nv_bfloat16 b0 = __float2bfloat16(t16[gg+2]);
                        nv_bfloat16 b1 = __float2bfloat16(t16[gg+3]);
                        w1 = (uint32_t)(*(uint16_t*)&b0) | ((uint32_t)(*(uint16_t*)&b1) << 16);
                    }
                    {
                        nv_bfloat16 b0 = __float2bfloat16(t16[gg+4]);
                        nv_bfloat16 b1 = __float2bfloat16(t16[gg+5]);
                        w2 = (uint32_t)(*(uint16_t*)&b0) | ((uint32_t)(*(uint16_t*)&b1) << 16);
                    }
                    {
                        nv_bfloat16 b0 = __float2bfloat16(t16[gg+6]);
                        nv_bfloat16 b1 = __float2bfloat16(t16[gg+7]);
                        w3 = (uint32_t)(*(uint16_t*)&b0) | ((uint32_t)(*(uint16_t*)&b1) << 16);
                    }
                    int byte_off = g * 2;
                    int swizzled = (byte_off & ~0xF) ^ ((tid & 7) << 4) | (byte_off & 0xF);
                    int saddr = row_base + swizzled;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                                 :: "r"(saddr), "r"(w0), "r"(w1), "r"(w2), "r"(w3));
                }
            }
        }
        
        // Compute global max and correction factor (for multi-tile accumulation)
        float nm = fmaxf(row_max, tile_max);
        float corr = __expf(row_max - nm);
        float ts = tile_sum * __expf(tile_max - nm);
        __syncthreads();
        
        // ===== Scale O[128:511] in TMEM for accumulation (tile > t0) =====
        if (!SINGLE_TILE && tile > t0) {
            for (int c = TILE_S; c < D_V; c += 16) {
                float t16[16];
                int addr = taddr + (tid << 16) + c;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                    : "=f"(t16[0]), "=f"(t16[1]), "=f"(t16[2]), "=f"(t16[3]),
                      "=f"(t16[4]), "=f"(t16[5]), "=f"(t16[6]), "=f"(t16[7]),
                      "=f"(t16[8]), "=f"(t16[9]), "=f"(t16[10]), "=f"(t16[11]),
                      "=f"(t16[12]), "=f"(t16[13]), "=f"(t16[14]), "=f"(t16[15])
                    : "r"(addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int i = 0; i < 16; i++) t16[i] *= corr;
                uint32_t* u = (uint32_t*)t16;
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
                    "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
                    :: "r"(addr),
                       "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]),
                       "r"(u[4]), "r"(u[5]), "r"(u[6]), "r"(u[7]),
                       "r"(u[8]), "r"(u[9]), "r"(u[10]), "r"(u[11]),
                       "r"(u[12]), "r"(u[13]), "r"(u[14]), "r"(u[15]));
            }
        }
        
        // Update running stats
        row_max = nm;
        row_sum = corr * row_sum + ts;
        
        // ===== PV Phase =====
        // For tile > t0: vc 0,1 use acc=0 (cols 0-127 had QK scores, need fresh)
        //                vc 2..7 use acc=1 (accumulate on scaled O[128:511])
        int V_buf_base = work_smem + 2 * TILE_BYTES;
        int pv_acc_base = (tile > t0) ? 1 : 0;
        
        __syncthreads();
        
        if (wid == 0 && elect_sync()) {
            int phase = 0;
            for (int vc = 1; vc < V_CHUNKS; vc++) {
                int stage = vc % NUM_PV_STAGES;
                mbar_wait(mma_bar + stage * 8, phase ^ 1);
                if (stage == NUM_PV_STAGES - 1) phase ^= 1;
                
                int v_smem = V_buf_base + stage * TILE_BYTES;
                mbar_tx(tma_bar + stage * 8, TILE_BYTES);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%2, %3, %4}], [%5];"
                    :: "r"(v_smem), "l"(&KV_tm),
                       "r"(0), "r"(bi * kv_len + kvs), "r"(vc), "r"(tma_bar + stage * 8) : "memory");
            }
        }
        else if (wid == 1 && elect_sync()) {
            int phase = 0;
            for (int vc = 0; vc < V_CHUNKS; vc++) {
                int stage = vc % NUM_PV_STAGES;
                mbar_wait(tma_bar + stage * 8, phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (stage == NUM_PV_STAGES - 1) phase ^= 1;
                
                int v_smem = V_buf_base + stage * TILE_BYTES;
                int out_taddr = taddr + vc * BK;
                
                // vc < 2: cols 0-127 (overlap with QK), always acc=0 for first k
                // vc >= 2: cols 128-511, acc=1 if tile > t0 (accumulate on scaled O)
                int vc_acc_base = (vc < 2) ? 0 : pv_acc_base;
                int first_in_vc = 1;
                for (int k1 = 0; k1 < 2; k1++) {
                    int p_addr = (k1 == 0) ? P0_smem : P1_smem;
                    int v_k1_off = k1 * 64 * 128;
                    for (int k2 = 0; k2 < BK / MMA_K; k2++) {
                        uint64_t a_desc = make_desc(p_addr + k2 * 32);
                        uint64_t b_desc = make_desc(v_smem + v_k1_off + k2 * 16 * 128);
                        int acc = (first_in_vc && !vc_acc_base) ? 0 : 1;
                        tcgen05_mma(out_taddr, a_desc, b_desc, idesc_pv, acc);
                        first_in_vc = 0;
                    }
                }
                tcgen05_commit(mma_bar + stage * 8);
            }
            tcgen05_commit(mainloop_bar);
        }
        
        __syncthreads();
        mbar_wait(mainloop_bar, 0);
        
        // ===== Merge saved O[0:127] with PV result (tile > t0) =====
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (!SINGLE_TILE && tile > t0) {
            for (int c = 0; c < TILE_S; c += 16) {
                float t16[16];
                int addr = taddr + (tid << 16) + c;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                    : "=f"(t16[0]), "=f"(t16[1]), "=f"(t16[2]), "=f"(t16[3]),
                      "=f"(t16[4]), "=f"(t16[5]), "=f"(t16[6]), "=f"(t16[7]),
                      "=f"(t16[8]), "=f"(t16[9]), "=f"(t16[10]), "=f"(t16[11]),
                      "=f"(t16[12]), "=f"(t16[13]), "=f"(t16[14]), "=f"(t16[15])
                    : "r"(addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    t16[i] += corr * o_save[c + i];
                uint32_t* u = (uint32_t*)t16;
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
                    "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
                    :: "r"(addr),
                       "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]),
                       "r"(u[4]), "r"(u[5]), "r"(u[6]), "r"(u[7]),
                       "r"(u[8]), "r"(u[9]), "r"(u[10]), "r"(u[11]),
                       "r"(u[12]), "r"(u[13]), "r"(u[14]), "r"(u[15]));
            }
        }
    }
    
    // ===== Epilogue: read O from TMEM, normalize, write to Oa =====
    asm volatile("tcgen05.fence::after_thread_sync;");
    float inv = (row_sum > 0) ? 1.0f / row_sum : 0.0f;
    for (int vc = 0; vc < V_CHUNKS; vc++) {
        int out_taddr = taddr + vc * BK;
        for (int c = 0; c < BK; c += 16) {
            float t16[16];
            int addr = out_taddr + (tid << 16) + c;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                : "=f"(t16[0]), "=f"(t16[1]), "=f"(t16[2]), "=f"(t16[3]),
                  "=f"(t16[4]), "=f"(t16[5]), "=f"(t16[6]), "=f"(t16[7]),
                  "=f"(t16[8]), "=f"(t16[9]), "=f"(t16[10]), "=f"(t16[11]),
                  "=f"(t16[12]), "=f"(t16[13]), "=f"(t16[14]), "=f"(t16[15])
                : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int base_d = (vc * BK + c) * 128 + tid;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                nv_bfloat16 val = __float2bfloat16(t16[i] * inv);
                asm volatile("st.global.cs.b16 [%0], %1;" :: "l"((nv_bfloat16*)(Oout + base_d + i * 128)), "h"(*(uint16_t*)&val) : "memory");
            }
        }
    }
    
    // Store LSE
    La[block_linear * 128 + tid] = logf(fmaxf(row_sum, 1e-30f)) + row_max;
    
    __syncthreads();
    if (wid == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(taddr), "r"(D_V));
    }
}

// ============ Reduction Kernel ============
// Coalesced memory access: row = tid % 128 so warp threads read consecutive rows
// This gives perfect 128-byte coalescing for Oa reads (row is innermost dimension)
static constexpr int RD_DV = 4;
static constexpr int RD_TB = 512;
static constexpr int MAX_SK = 32;

__global__ __launch_bounds__(RD_TB)
void mla_reduce(
    const nv_bfloat16* __restrict__ Oa,
    const float* __restrict__ La, 
    nv_bfloat16* __restrict__ O,
    int sk, int num_head_groups, int Q_LEN
) {
    const int dv_base = blockIdx.x * RD_DV;
    const int gi = blockIdx.y;
    const int bi = blockIdx.z;
    const int tid = threadIdx.x;
    
    // Coalesced mapping: row varies fastest within warp
    const int row = tid % 128;             // 0..127 (consecutive in warp)
    const int lane = tid / 128;            // 0..3 (RD_DV columns)
    const int d = dv_base + lane;
    
    int hpb = NUM_HEADS / num_head_groups;
    int q = row / hpb;
    int h_local = row % hpb;
    int h_global = gi * hpb + h_local;
    
    // Load La into shared memory cooperatively
    __shared__ float la_smem[MAX_SK * 128];
    int la_block_base = (bi * num_head_groups * sk + gi * sk) * 128;
    int la_total = sk * 128;
    for (int i = tid; i < la_total; i += RD_TB)
        la_smem[i] = La[la_block_base + i];
    __syncthreads();
    
    if (q >= Q_LEN || h_global >= NUM_HEADS) return;
    
    int o_base = (bi * Q_LEN + q) * NUM_HEADS * D_V + h_global * D_V;
    
    // Phase 1: Online softmax from shared memory La
    float lse_max = -1e30f;
    float sum_exp = 0.0f;
    for (int s = 0; s < sk; s++) {
        float la_val = la_smem[s * 128 + row];
        float new_max = fmaxf(lse_max, la_val);
        sum_exp = sum_exp * __expf(lse_max - new_max) + __expf(la_val - new_max);
        lse_max = new_max;
    }
    float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
    
    // Phase 2: Accumulate with La from shared memory
    // Oa layout: [split][D_V][128_rows] - row is innermost, coalesced within warp
    if (d < D_V) {
        float acc = 0.0f;
        int oa_base_gi = (bi * num_head_groups * sk + gi * sk) * D_V * 128;
        int oa_d = oa_base_gi + d * 128 + row;
        for (int s = 0; s < sk; s++) {
            float scale = __expf(la_smem[s * 128 + row] - lse_max) * inv_sum;
            acc += scale * __bfloat162float(Oa[oa_d + s * D_V * 128]);
        }
        O[o_base + d] = __float2bfloat16(acc);
    }
}

// ============ Reference Kernel (Q_LEN=1) ============
__global__ void ref_k(const nv_bfloat16* Q, const nv_bfloat16* KV, nv_bfloat16* O, 
                      float ss, int kl, int Q_LEN) {
    int h = blockIdx.x, b = blockIdx.y, q = blockIdx.z;
    int t = threadIdx.x;
    const nv_bfloat16* qr = Q + (q * NUM_HEADS + h) * D_K;
    const nv_bfloat16* kv = KV + b * kl * D_K;
    nv_bfloat16* o = O + (q * NUM_HEADS + h) * D_V;
    
    // Causal limit
    int causal_lim = (Q_LEN > 1) ? (kl - Q_LEN + q + 1) : kl;
    
    extern __shared__ float sc[];
    for (int i = t; i < kl; i += blockDim.x) {
        if (i < causal_lim) {
            float s = 0;
            for (int d = 0; d < D_K; d++)
                s += __bfloat162float(qr[d]) * __bfloat162float(kv[i * D_K + d]);
            sc[i] = s * ss;
        } else {
            sc[i] = -1e30f;
        }
    }
    __syncthreads();
    __shared__ float m2[2];
    if (t == 0) { float mx = -1e30f; for (int i = 0; i < causal_lim; i++) mx = fmaxf(mx, sc[i]); m2[0] = mx; }
    __syncthreads();
    for (int i = t; i < kl; i += blockDim.x) sc[i] = (i < causal_lim) ? expf(sc[i] - m2[0]) : 0.0f;
    __syncthreads();
    if (t == 0) { float s = 0; for (int i = 0; i < causal_lim; i++) s += sc[i]; m2[1] = 1.0f / s; }
    __syncthreads();
    for (int i = t; i < kl; i += blockDim.x) sc[i] *= m2[1];
    __syncthreads();
    for (int d = t; d < D_V; d += blockDim.x) {
        float v = 0; for (int i = 0; i < causal_lim; i++) v += sc[i] * __bfloat162float(kv[i * D_K + d]);
        o[d] = __float2bfloat16(v);
    }
}

// ============ Host code ============
void fill(nv_bfloat16* d, int n) {
    for (int i = 0; i < n; i++)
        d[i] = __float2bfloat16(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
}
void ck(CUresult e) {
    if (e != CUDA_SUCCESS) { const char* s; cuGetErrorString(e, &s); fprintf(stderr, "CU: %s\n", s); exit(1); }
}

int main(int argc, char** argv) {
    cuInit(0);
    int B = 1, KL = 4096;
    
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--b=", 4)) B = atoi(argv[i] + 4);
        if (!strncmp(argv[i], "--k=", 4)) KL = atoi(argv[i] + 4);
    }
    float ss = 1.0f / sqrtf((float)D_K);
    
    printf("=== MLA Multi-Token Decode (B200) ===\n");
    printf("DeepSeek V3: H=%d, D_K=%d, D_V=%d\n", NUM_HEADS, D_K, D_V);
    printf("B=%d, KV_LEN=%d\n\n", B, KL);
    
    // Test Q_LEN = 1,2,3,4
    for (int Q_LEN = 1; Q_LEN <= 4; Q_LEN++) {
        // Find largest hpb that divides NUM_HEADS and hpb * Q_LEN <= 128
        int hpb = 128 / Q_LEN;  // floor division
        // Round down to nearest divisor of NUM_HEADS
        while (NUM_HEADS % hpb != 0) hpb--;
        int num_head_groups = NUM_HEADS / hpb;
        
        int max_sk = (KL + TILE_S - 1) / TILE_S;
        // Auto-tune: try multiple sk values, pick best estimated total time
        // Reduce time ≈ 7 * num_head_groups * sk / 32 us (scales with Q_LEN and sk)
        // Main time: need enough blocks (~148) for parallelism
        // For Q_LEN=1,2,4: sk=32 (enough parallelism, reduce is manageable)  
        // For Q_LEN=8: sk=16 (128 blocks, reduce has 16 reads → half cost)
        // With two-pass softmax (64 regs, no spills), always use sk=max_sk (tps=1)
        // This maximizes parallelism and avoids multi-tile overhead
        // For Q8 with 8 groups: sk=16 gives 128 blocks (better reduce cost)
        // For Q1-Q4: sk=max (tps=1, best parallelism)
        int sk = max_sk;
        if (Q_LEN == 8 && num_head_groups * 16 >= 64) sk = 16;
        
        // Override from command line
        for (int i = 1; i < argc; i++) {
            if (!strncmp(argv[i], "--sk=", 5)) sk = min(atoi(argv[i] + 5), max_sk);
        }
        
        size_t Qs = Q_LEN * NUM_HEADS * D_K;
        size_t KVs = B * KL * D_K;
        size_t Os = Q_LEN * NUM_HEADS * D_V;
        
        nv_bfloat16 *hQ = new nv_bfloat16[Qs], *hKV = new nv_bfloat16[KVs];
        nv_bfloat16 *hO = new nv_bfloat16[Os], *hOr = new nv_bfloat16[Os];
        srand(42); fill(hQ, Qs); fill(hKV, KVs);
        
        nv_bfloat16 *dQ, *dKV, *dO, *dOr;
        nv_bfloat16 *dOa; float *dLa;
        cudaMalloc(&dQ, Qs * 2); cudaMalloc(&dKV, KVs * 2);
        cudaMalloc(&dO, Os * 2); cudaMalloc(&dOr, Os * 2);
        
        int total_blocks = B * num_head_groups * sk;
        cudaMalloc(&dOa, (size_t)total_blocks * D_V * 128 * 2);
        cudaMalloc(&dLa, (size_t)total_blocks * 128 * 4);
        
        cudaMemcpy(dQ, hQ, Qs * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dKV, hKV, KVs * 2, cudaMemcpyHostToDevice);
        
        // No reshape needed — Q loaded directly via TMA
        
        // TMA descriptors
        CUtensorMap Qtm, KVtm;
        // Q: [Q_LEN * NUM_HEADS, D_K] -> 3D [BK, Q_LEN*H, K_ITERS]
        // boxDim = {64, hpb, 1} — load hpb rows per TMA call
        {
            uint64_t gd[3] = {64, (uint64_t)B * Q_LEN * NUM_HEADS, (uint64_t)K_ITERS};
            uint64_t gs[2] = {(uint64_t)D_K * 2, 128};
            uint32_t bd[3] = {64, (uint32_t)hpb, 1};
            uint32_t es[3] = {1, 1, 1};
            ck(cuTensorMapEncodeTiled(&Qtm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)dQ,
                gd, gs, bd, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        }
        // KV: [B * KL, D_K] -> 3D [BK, B*KL, K_ITERS]
        {
            uint64_t gd[3] = {64, (uint64_t)B * KL, (uint64_t)K_ITERS};
            uint64_t gs[2] = {(uint64_t)D_K * 2, 128};
            uint32_t bd[3] = {64, (uint32_t)TILE_S, 1};
            uint32_t es[3] = {1, 1, 1};
            ck(cuTensorMapEncodeTiled(&KVtm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)dKV,
                gd, gs, bd, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        }
        
        // Reference
        {
            dim3 g(NUM_HEADS, B, Q_LEN);
            int sm = (KL + 1) * 4;
            if (sm > 48000) cudaFuncSetAttribute(ref_k, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
            ref_k<<<g, 256, sm>>>(dQ, dKV, dOr, ss, KL, Q_LEN);
            cudaDeviceSynchronize();
        }
        
        // SMEM: QK stages co-load Q+K (2*TILE_BYTES per stage), PV reuses same space
        constexpr int smem_size = NUM_QK_STAGES * 2 * TILE_BYTES;  // 160KB
        
        // Determine if single-tile mode (tps==1 for all splits)
        int kvt = (KL + TILE_S - 1) / TILE_S;
        int tps = (kvt + sk - 1) / sk;
        bool single_tile = (tps == 1);
        
        cudaFuncSetAttribute(mla_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        cudaFuncSetAttribute(mla_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        
        // Single stream version (default stream 0)
        auto run_main = [&]() {
            dim3 g(num_head_groups * sk, B);
            if (single_tile)
                mla_kernel<true><<<g, TB, smem_size>>>(Qtm, KVtm, dOa, dLa, ss, KL, sk, num_head_groups, Q_LEN);
            else
                mla_kernel<false><<<g, TB, smem_size>>>(Qtm, KVtm, dOa, dLa, ss, KL, sk, num_head_groups, Q_LEN);
        };
        auto run_reduce = [&]() {
            dim3 rg((D_V + RD_DV - 1) / RD_DV, num_head_groups, B);
            mla_reduce<<<rg, RD_TB>>>(dOa, dLa, dO, sk, num_head_groups, Q_LEN);
        };
        auto run = [&]() {
            run_main();
            run_reduce();
        };
        
        run();
        auto err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Q_LEN=%d: ERR: %s\n", Q_LEN, cudaGetErrorString(err)); 
            goto cleanup;
        }
        
        cudaMemcpy(hO, dO, Os * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(hOr, dOr, Os * 2, cudaMemcpyDeviceToHost);
        {
            float mx = 0;
            for (size_t i = 0; i < Os; i++) {
                float r = __bfloat162float(hOr[i]), g = __bfloat162float(hO[i]);
                mx = fmaxf(mx, fabsf(r - g) / fmaxf(fabsf(r), 1e-3f));
            }
            printf("Q_LEN=%d: Max rel err: %.6f (sk=%d, groups=%d)\n", Q_LEN, mx, sk, num_head_groups);
        }
        
        // Benchmark
        for (int i = 0; i < 3; i++) run();
        cudaDeviceSynchronize();
        
        {
            cudaEvent_t st, en;
            cudaEventCreate(&st); cudaEventCreate(&en);
            int N = 50;
            // Time main kernel only
            cudaEventRecord(st);
            for (int i = 0; i < N; i++) run_main();
            cudaEventRecord(en);
            cudaEventSynchronize(en);
            float ms_m;
            cudaEventElapsedTime(&ms_m, st, en);
            float us_m = (ms_m / N) * 1000;
            // Time full pipeline
            cudaDeviceSynchronize();
            cudaEventRecord(st);
            for (int i = 0; i < N; i++) run();
            cudaEventRecord(en);
            cudaEventSynchronize(en);
            float ms;
            cudaEventElapsedTime(&ms, st, en);
            float us = (ms / N) * 1000;
            double fl = (double)B * NUM_HEADS * Q_LEN * KL * (D_K - D_V + D_V + D_V);
            double tflops = fl / (us * 1e-6) / 1e12;
            printf("Q_LEN=%d: %.1f TFLOPS, %.1f us (main: %.1f, reduce: %.1f)\n",
                   Q_LEN, tflops, us, us_m, us - us_m);
            cudaEventDestroy(st); cudaEventDestroy(en);
        }

cleanup:
        cudaFree(dQ); cudaFree(dKV); cudaFree(dO); cudaFree(dOr);
        cudaFree(dOa); cudaFree(dLa);
        delete[] hQ; delete[] hKV; delete[] hO; delete[] hOr;
        printf("\n");
    }
    
    return 0;
}
