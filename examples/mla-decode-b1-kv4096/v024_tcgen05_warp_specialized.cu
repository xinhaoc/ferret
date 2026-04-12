// MLA Decode for DeepSeek V3 on B200 (SM100a)
// Warp-specialized TMA/MMA pipeline for both QK and PV GEMMs
// Based on example 04_warp_specialization pattern
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
static constexpr int NUM_QK_STAGES = 4;
static constexpr int NUM_PV_STAGES = 2;
static constexpr int MAX_STAGES = 4;  // max of QK and PV

// SMEM tile size: NUM_HEADS * BK * sizeof(bf16) = 128 * 64 * 2 = 16384
static constexpr int TILE_BYTES = NUM_HEADS * BK * 2;

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
    constexpr uint64_t SBO = 8ULL * 128;
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

// ============ Main Kernel ============
__global__ __launch_bounds__(TB)
void mla_kernel(
    const __grid_constant__ CUtensorMap Q_tm,
    const __grid_constant__ CUtensorMap KV_tm,
    float* __restrict__ Oa, float* __restrict__ La,
    float ss, int kv_len, int sk
) {
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int si = blockIdx.x;
    const int bi = blockIdx.y;

    const int kvt = (kv_len + TILE_S - 1) / TILE_S;
    const int tps = (kvt + sk - 1) / sk;
    const int t0 = si * tps;
    const int t1 = min(t0 + tps, kvt);
    if (t0 >= t1) return;

    // ===== SMEM layout =====
    // Q: K_ITERS * TILE_BYTES = 9 * 16384 = 147456
    // Work: MAX_STAGES * TILE_BYTES = 4 * 16384 = 65536
    // Total: 212992 bytes = 208 KB
    extern __shared__ __align__(1024) char smem_buf[];
    const int smem_base = __cvta_generic_to_shared(smem_buf);

    const int Q_smem = smem_base;
    const int work_smem = smem_base + K_ITERS * TILE_BYTES;
    // QK phase: K[stage] at work_smem + stage * TILE_BYTES (up to 4 stages)
    // PV phase: P[0] at work_smem, P[1] at work_smem + TILE_BYTES
    //           V[stage] at work_smem + 2*TILE_BYTES + stage*TILE_BYTES (up to 2 stages)

    // Barriers: tma[MAX_STAGES] + mma[MAX_STAGES] + mainloop + q_bar = 10
    __shared__ uint64_t mbar_buf[10];
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
        // Allocate 512 TMEM columns (for PV: 8 chunks * 64 cols)
        // QK uses first 128 columns, PV uses all 512
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(addr_smem), "r"(D_V));
    }
    __syncthreads();
    const int taddr = tmem_addr_buf[0];

    // ===== Load Q via TMA (bulk load all 9 tiles) =====
    if (wid == 0 && elect_sync()) {
        mbar_tx(q_bar, TILE_BYTES * K_ITERS);
        for (int ki = 0; ki < K_ITERS; ki++) {
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                "[%0], [%1, {%2, %3, %4}], [%5];"
                :: "r"(Q_smem + ki * TILE_BYTES), "l"(&Q_tm),
                   "r"(0), "r"(bi * NUM_HEADS), "r"(ki), "r"(q_bar) : "memory");
        }
    }
    mbar_wait(q_bar, 0);
    __syncthreads();

    // Instruction descriptors
    // QK: A=Q[128,K] K-major, B=K[128,K] K-major → S[128,128]
    constexpr uint32_t idesc_qk = (1U << 4) | (1U << 7) | (1U << 10)
        | ((uint32_t)(TILE_S >> 3) << 17) | ((uint32_t)(NUM_HEADS >> 4) << 24);

    // PV: A=P[128,128] K-major, B=V[128,64] MN-major → O[128,64]
    constexpr uint32_t idesc_pv = (1U << 4) | (1U << 7) | (1U << 10)
        | (1U << 16)  // b_major = MN-major
        | ((uint32_t)(BK >> 3) << 17) | ((uint32_t)(NUM_HEADS >> 4) << 24);

    float* Oout = Oa + (bi * sk + si) * D_V * NUM_HEADS;
    float row_max = -1e30f;
    float row_sum = 0.0f;

    // ===== Main loop over KV tiles =====
    for (int tile = t0; tile < t1; tile++) {
        const int kvs = tile * TILE_S;
        const int tlen = min(TILE_S, kv_len - kvs);

        // ===== QK Phase: Warp-specialized pipeline =====
        // Reinit barriers for QK phase
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
            // === TMA producer warp ===
            int phase = 0;
            for (int ki = 0; ki < K_ITERS; ki++) {
                int stage = ki % NUM_QK_STAGES;
                // Wait for MMA to finish with this buffer
                mbar_wait(mma_bar + stage * 8, phase ^ 1);
                if (stage == NUM_QK_STAGES - 1) phase ^= 1;

                int k_smem = work_smem + stage * TILE_BYTES;
                mbar_tx(tma_bar + stage * 8, TILE_BYTES);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%2, %3, %4}], [%5];"
                    :: "r"(k_smem), "l"(&KV_tm),
                       "r"(0), "r"(kvs), "r"(ki), "r"(tma_bar + stage * 8) : "memory");
            }
        }
        else if (wid == 1 && elect_sync()) {
            // === MMA consumer warp ===
            int phase = 0;
            for (int ki = 0; ki < K_ITERS; ki++) {
                int stage = ki % NUM_QK_STAGES;
                // Wait for TMA to load this buffer
                mbar_wait(tma_bar + stage * 8, phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (stage == NUM_QK_STAGES - 1) phase ^= 1;

                int k_smem = work_smem + stage * TILE_BYTES;

                // QK MMA for this K tile
                for (int k2 = 0; k2 < BK / MMA_K; k2++) {
                    uint64_t a_desc = make_desc(Q_smem + ki * TILE_BYTES + k2 * 32);
                    uint64_t b_desc = make_desc(k_smem + k2 * 32);
                    tcgen05_mma(taddr, a_desc, b_desc, idesc_qk, (ki == 0 && k2 == 0) ? 0 : 1);
                }
                tcgen05_commit(mma_bar + stage * 8);
            }
            // Signal mainloop completion
            tcgen05_commit(mainloop_bar);
        }

        // All threads wait for QK completion
        __syncthreads();
        mbar_wait(mainloop_bar, 0);

        // ===== Softmax Phase (warp 0 starts preloading V during softmax) =====
        asm volatile("tcgen05.fence::after_thread_sync;");

        // Reinit PV barriers and start first V load while other threads do softmax
        if (wid == 0 && elect_sync()) {
            for (int i = 0; i < NUM_PV_STAGES; i++) {
                mbar_init(tma_bar + i * 8, 1);
                mbar_init(mma_bar + i * 8, 1);
            }
            mbar_init(mainloop_bar, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
            // Preload first V chunk
            int v_smem0 = work_smem + 2 * TILE_BYTES;
            mbar_tx(tma_bar, TILE_BYTES);
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                "[%0], [%1, {%2, %3, %4}], [%5];"
                :: "r"(v_smem0), "l"(&KV_tm),
                   "r"(0), "r"(kvs), "r"(0), "r"(tma_bar) : "memory");
        }

        float sl[TILE_S];
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
            for (int i = 0; i < 16; i++) sl[c + i] = t16[i];
        }

        // Online softmax (fully unrolled assuming full tiles)
        float tm = -1e30f;
        #pragma unroll
        for (int t = 0; t < TILE_S; t++) {
            float v = sl[t] * ss;
            sl[t] = v;
            tm = fmaxf(tm, v);
        }
        // Mask out-of-bounds (only needed for partial tiles)
        if (tlen < TILE_S) {
            for (int t = tlen; t < TILE_S; t++) sl[t] = -1e30f;
            tm = -1e30f;
            for (int t = 0; t < tlen; t++) tm = fmaxf(tm, sl[t]);
        }
        float nm = fmaxf(row_max, tm);
        float corr = __expf(row_max - nm);
        float ts = 0;
        #pragma unroll
        for (int t = 0; t < TILE_S; t++) {
            float e = __expf(sl[t] - nm);
            sl[t] = e;
            ts += e;
        }
        if (tlen < TILE_S) {
            ts = 0;
            for (int t = 0; t < tlen; t++) ts += sl[t];
            for (int t = tlen; t < TILE_S; t++) sl[t] = 0;
        }

        // Write P to SMEM: P[0] at work_smem, P[1] at work_smem + TILE_BYTES
        // Vectorized: 8 bf16 values (16 bytes) per store using st.shared.v4.b32
        int P0_smem = work_smem;
        int P1_smem = work_smem + TILE_BYTES;
        #pragma unroll
        for (int p_tile = 0; p_tile < 2; p_tile++) {
            int p_base = (p_tile == 0) ? P0_smem : P1_smem;
            int row_base = p_base + tid * 128;
            #pragma unroll
            for (int g = 0; g < 64; g += 8) {
                // Pack 8 consecutive bf16 values into 4 uint32 (16 bytes)
                int t_off = p_tile * 64 + g;
                uint32_t w0, w1, w2, w3;
                {
                    nv_bfloat16 b0 = __float2bfloat16(sl[t_off+0]);
                    nv_bfloat16 b1 = __float2bfloat16(sl[t_off+1]);
                    w0 = (uint32_t)(*(uint16_t*)&b0) | ((uint32_t)(*(uint16_t*)&b1) << 16);
                }
                {
                    nv_bfloat16 b0 = __float2bfloat16(sl[t_off+2]);
                    nv_bfloat16 b1 = __float2bfloat16(sl[t_off+3]);
                    w1 = (uint32_t)(*(uint16_t*)&b0) | ((uint32_t)(*(uint16_t*)&b1) << 16);
                }
                {
                    nv_bfloat16 b0 = __float2bfloat16(sl[t_off+4]);
                    nv_bfloat16 b1 = __float2bfloat16(sl[t_off+5]);
                    w2 = (uint32_t)(*(uint16_t*)&b0) | ((uint32_t)(*(uint16_t*)&b1) << 16);
                }
                {
                    nv_bfloat16 b0 = __float2bfloat16(sl[t_off+6]);
                    nv_bfloat16 b1 = __float2bfloat16(sl[t_off+7]);
                    w3 = (uint32_t)(*(uint16_t*)&b0) | ((uint32_t)(*(uint16_t*)&b1) << 16);
                }
                // 8 consecutive bf16 at byte offsets g*2 .. g*2+15 are contiguous after swizzle
                // (XOR only affects bits 4-6, so 16-byte groups stay contiguous)
                int byte_off = g * 2;
                int swizzled = (byte_off & ~0xF) ^ ((tid & 7) << 4) | (byte_off & 0xF);
                int addr = row_base + swizzled;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                             :: "r"(addr), "r"(w0), "r"(w1), "r"(w2), "r"(w3));
            }
        }
        __syncthreads();

        // ===== PV Phase: Warp-specialized pipeline =====
        // Barriers already inited and first V chunk already loading from softmax phase

        int V_buf_base = work_smem + 2 * TILE_BYTES;  // After P tiles

        if (wid == 0 && elect_sync()) {
            // === TMA producer for V (vc=0 already loaded during softmax) ===
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
                       "r"(0), "r"(kvs), "r"(vc), "r"(tma_bar + stage * 8) : "memory");
            }
        }
        else if (wid == 1 && elect_sync()) {
            // === MMA consumer for PV ===
            int phase = 0;
            for (int vc = 0; vc < V_CHUNKS; vc++) {
                int stage = vc % NUM_PV_STAGES;
                mbar_wait(tma_bar + stage * 8, phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (stage == NUM_PV_STAGES - 1) phase ^= 1;

                int v_smem = V_buf_base + stage * TILE_BYTES;
                int out_taddr = taddr + vc * BK;

                // PV MMA: O[128,64] = P[128,128] * V[128,64]
                // P: 2 tiles (k1=0,1), each [128,64], K-major
                // V: MN-major, 128 rows of 64 dv values
                int first = 1;
                for (int k1 = 0; k1 < 2; k1++) {
                    int p_addr = (k1 == 0) ? P0_smem : P1_smem;
                    int v_k1_off = k1 * 64 * 128;  // skip 64 token rows
                    for (int k2 = 0; k2 < BK / MMA_K; k2++) {
                        uint64_t a_desc = make_desc(p_addr + k2 * 32);
                        uint64_t b_desc = make_desc(v_smem + v_k1_off + k2 * 16 * 128);
                        tcgen05_mma(out_taddr, a_desc, b_desc, idesc_pv, first ? 0 : 1);
                        first = 0;
                    }
                }
                tcgen05_commit(mma_bar + stage * 8);
            }
            // Signal PV mainloop completion
            tcgen05_commit(mainloop_bar);
        }

        // All threads wait for PV completion
        __syncthreads();
        mbar_wait(mainloop_bar, 0);

        // ===== Accumulate + Finalize Phase (fused) =====
        asm volatile("tcgen05.fence::after_thread_sync;");

        row_max = nm;
        row_sum = corr * row_sum + ts;
        float inv = (row_sum > 0) ? 1.0f / row_sum : 0.0f;
        float corr_inv = corr * inv;
        int is_first = (tile == t0);

        // Read O from TMEM and write scaled result to global (fused accumulate+finalize)
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
                int base_d = (vc * BK + c) * NUM_HEADS + tid;
                if (is_first) {
                    #pragma unroll
                    for (int i = 0; i < 16; i++)
                        Oout[base_d + i * NUM_HEADS] = t16[i] * inv;
                } else {
                    #pragma unroll
                    for (int i = 0; i < 16; i++) {
                        int gaddr = base_d + i * NUM_HEADS;
                        Oout[gaddr] = corr_inv * Oout[gaddr] + t16[i] * inv;
                    }
                }
            }
        }
    }

    // ===== Store LSE =====
    La[(bi * sk + si) * NUM_HEADS + tid] = logf(fmaxf(row_sum, 1e-30f)) + row_max;

    __syncthreads();
    if (wid == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(taddr), "r"(D_V));
    }
}

// ============ Reduction Kernel ============
// Oa layout: [split][dv][head]. Coalesced reads: threads map to heads.
// Grid: (D_V, B), Block: 128 threads (one per head)
// 2-pass streaming: first find lse_max+sum, then accumulate (less register pressure)
static constexpr int DV_PER_BLK = 1;
__global__ __launch_bounds__(128)
void mla_reduce(const float* Oa, const float* La, nv_bfloat16* O, int sk) {
    const int d = blockIdx.x;
    const int b = blockIdx.y;
    const int h = threadIdx.x;  // head index, 0..127
    
    // Pass 1: find lse_max and sum_exp (streaming, no array)
    float lse_max = -1e30f;
    for (int s = 0; s < sk; s++) {
        lse_max = fmaxf(lse_max, La[(b * sk + s) * NUM_HEADS + h]);
    }
    float sum_exp = 0.0f;
    for (int s = 0; s < sk; s++) {
        sum_exp += __expf(La[(b * sk + s) * NUM_HEADS + h] - lse_max);
    }
    float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
    
    // Pass 2: accumulate Oa values with scale (streaming)
    float acc = 0.0f;
    const float* oa_d = Oa + d * NUM_HEADS + h;
    for (int s = 0; s < sk; s++) {
        float scale = __expf(La[(b * sk + s) * NUM_HEADS + h] - lse_max) * inv_sum;
        acc += scale * oa_d[(b * sk + s) * D_V * NUM_HEADS];
    }
    
    // Write output
    O[(b * NUM_HEADS + h) * D_V + d] = __float2bfloat16(acc);
}

// ============ Reference Kernel ============
__global__ void ref_k(const nv_bfloat16* Q, const nv_bfloat16* KV, nv_bfloat16* O, float ss, int kl) {
    int h = blockIdx.x, b = blockIdx.y, t = threadIdx.x;
    const nv_bfloat16* q = Q + b * NUM_HEADS * D_K + h * D_K;
    const nv_bfloat16* kv = KV + b * kl * D_K;
    nv_bfloat16* o = O + b * NUM_HEADS * D_V + h * D_V;
    extern __shared__ float sc[];
    for (int i = t; i < kl; i += blockDim.x) {
        float s = 0;
        for (int d = 0; d < D_K; d++)
            s += __bfloat162float(q[d]) * __bfloat162float(kv[i * D_K + d]);
        sc[i] = s * ss;
    }
    __syncthreads();
    __shared__ float m2[2];
    if (t == 0) { float mx = -1e30f; for (int i = 0; i < kl; i++) mx = fmaxf(mx, sc[i]); m2[0] = mx; }
    __syncthreads();
    for (int i = t; i < kl; i += blockDim.x) sc[i] = expf(sc[i] - m2[0]);
    __syncthreads();
    if (t == 0) { float s = 0; for (int i = 0; i < kl; i++) s += sc[i]; m2[1] = 1.0f / s; }
    __syncthreads();
    for (int i = t; i < kl; i += blockDim.x) sc[i] *= m2[1];
    __syncthreads();
    for (int d = t; d < D_V; d += blockDim.x) {
        float v = 0; for (int i = 0; i < kl; i++) v += sc[i] * __bfloat162float(kv[i * D_K + d]);
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
    int max_sk = (KL + TILE_S - 1) / TILE_S;
    int sk = max_sk;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--sk=", 5)) sk = min(atoi(argv[i] + 5), max_sk);
    }
    printf("MLA: B=%d H=%d KL=%d SK=%d\n", B, NUM_HEADS, KL, sk);

    size_t Qs = B * NUM_HEADS * D_K;
    size_t KVs = B * KL * D_K;
    size_t Os = B * NUM_HEADS * D_V;
    nv_bfloat16 *hQ = new nv_bfloat16[Qs], *hKV = new nv_bfloat16[KVs];
    nv_bfloat16 *hO = new nv_bfloat16[Os], *hOr = new nv_bfloat16[Os];
    srand(42); fill(hQ, Qs); fill(hKV, KVs);

    nv_bfloat16 *dQ, *dKV, *dO, *dOr;
    float *dOa, *dLa;
    cudaMalloc(&dQ, Qs * 2); cudaMalloc(&dKV, KVs * 2);
    cudaMalloc(&dO, Os * 2); cudaMalloc(&dOr, Os * 2);
    cudaMalloc(&dOa, B * sk * NUM_HEADS * D_V * 4);
    cudaMalloc(&dLa, B * sk * NUM_HEADS * 4);
    cudaMemcpy(dQ, hQ, Qs * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dKV, hKV, KVs * 2, cudaMemcpyHostToDevice);

    // TMA descriptors
    CUtensorMap Qtm, KVtm;
    // Q: [B*NUM_HEADS, D_K] → 3D (64, B*NUM_HEADS, D_K/64)
    {
        uint64_t gd[3] = {64, (uint64_t)B * NUM_HEADS, (uint64_t)K_ITERS};
        uint64_t gs[2] = {(uint64_t)D_K * 2, 128};
        uint32_t bd[3] = {64, (uint32_t)NUM_HEADS, 1};
        uint32_t es[3] = {1, 1, 1};
        ck(cuTensorMapEncodeTiled(&Qtm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)dQ,
            gd, gs, bd, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }
    // KV: [B*KL, D_K] → 3D (64, B*KL, D_K/64)
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
        dim3 g(NUM_HEADS, B);
        int sm = (KL + 1) * 4;
        if (sm > 48000) cudaFuncSetAttribute(ref_k, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
        ref_k<<<g, 256, sm>>>(dQ, dKV, dOr, ss, KL);
        cudaDeviceSynchronize();
    }

    // SMEM: Q(147456) + Work(65536) = 212992
    constexpr int smem_size = K_ITERS * TILE_BYTES + MAX_STAGES * TILE_BYTES;
    printf("SMEM: %d bytes (%.1f KB)\n", smem_size, smem_size / 1024.0f);
    cudaFuncSetAttribute(mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    auto run_main = [&]() {
        dim3 g(sk, B);
        mla_kernel<<<g, TB, smem_size>>>(Qtm, KVtm, dOa, dLa, ss, KL, sk);
    };
    auto run_reduce = [&]() {
        dim3 rg(D_V / DV_PER_BLK, B);
        mla_reduce<<<rg, NUM_HEADS>>>(dOa, dLa, dO, sk);
    };
    auto run = [&]() {
        run_main();
        run_reduce();
    };

    run();
    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("ERR: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(hO, dO, Os * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hOr, dOr, Os * 2, cudaMemcpyDeviceToHost);
    float mx = 0;
    for (size_t i = 0; i < Os; i++) {
        float r = __bfloat162float(hOr[i]), g = __bfloat162float(hO[i]);
        mx = fmaxf(mx, fabsf(r - g) / fmaxf(fabsf(r), 1e-3f));
    }
    printf("Max rel err: %.6f\n", mx);

    // Benchmark
    cudaEvent_t st, en, mid;
    cudaEventCreate(&st); cudaEventCreate(&en); cudaEventCreate(&mid);
    for (int i = 0; i < 3; i++) run();
    cudaDeviceSynchronize();
    int N = 50;
    // Time main kernel only
    cudaEventRecord(st);
    for (int i = 0; i < N; i++) run_main();
    cudaEventRecord(en);
    cudaEventSynchronize(en);
    float ms_main;
    cudaEventElapsedTime(&ms_main, st, en);
    float us_main = (ms_main / N) * 1000;
    // Time full pipeline (no graph)
    cudaEventRecord(st);
    for (int i = 0; i < N; i++) run();
    cudaEventRecord(en);
    cudaEventSynchronize(en);
    float ms;
    cudaEventElapsedTime(&ms, st, en);
    float us = (ms / N) * 1000;
    printf("Main: %.2f us, Reduce: %.2f us\n", us_main, us - us_main);
    
    // CUDA Graph for reduced launch overhead
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    {
        dim3 g(sk, B);
        mla_kernel<<<g, TB, smem_size, stream>>>(Qtm, KVtm, dOa, dLa, ss, KL, sk);
        dim3 rg(D_V / DV_PER_BLK, B);
        mla_reduce<<<rg, NUM_HEADS, 0, stream>>>(dOa, dLa, dO, sk);
    }
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    // Warmup graph
    for (int i = 0; i < 3; i++) cudaGraphLaunch(graph_exec, stream);
    cudaStreamSynchronize(stream);
    // Benchmark graph
    cudaEventRecord(st, stream);
    for (int i = 0; i < N; i++) cudaGraphLaunch(graph_exec, stream);
    cudaEventRecord(en, stream);
    cudaStreamSynchronize(stream);
    float ms_graph;
    cudaEventElapsedTime(&ms_graph, st, en);
    float us_graph = (ms_graph / N) * 1000;
    printf("Graph: %.2f us\n", us_graph);
    
    double fl = 2.0 * B * NUM_HEADS * KL * ((double)D_K + D_V);
    printf("Perf: %.2f us, %.2f TFLOPS (baseline ~32)\n", us_graph, fl / (us_graph * 1e-6) / 1e12);
    printf("NoGraph: %.2f us, %.2f TFLOPS\n", us, fl / (us * 1e-6) / 1e12);
    
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    cudaFree(dQ); cudaFree(dKV); cudaFree(dO); cudaFree(dOr);
    cudaFree(dOa); cudaFree(dLa);
    delete[] hQ; delete[] hKV; delete[] hO; delete[] hOr;
    cudaEventDestroy(st); cudaEventDestroy(en);
    return 0;
}
