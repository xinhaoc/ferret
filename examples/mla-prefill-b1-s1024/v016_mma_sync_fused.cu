
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cfloat>

using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;

// ============================================================
// Constants
// ============================================================
static constexpr int D_CKV = 512;
static constexpr int D_KPE = 64;
static constexpr int D_QK  = 576;   // D_CKV + D_KPE
static constexpr int D_V   = 512;   // same as D_CKV
static constexpr int BM    = 64;    // Q tile rows
static constexpr int BN    = 64;    // KV tile size
static constexpr int NUM_THREADS = 256;
static constexpr int NUM_WARPS   = 8;
static constexpr int WARP_SIZE   = 32;
static constexpr int NUM_STAGES  = 2;

// MMA tile sizes
static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;   // m16n8k16
static constexpr int MMA_K = 16;

// Derived constants
static constexpr int NUM_MMA_KV   = BN / MMA_K;       // 4 (16×16 k-tiles along KV dim for PV)
static constexpr int NUM_MMA_N16  = BN / 16;           // 4 (16-wide N tiles in QK output)
static constexpr int NUM_MMA_D_CKV_K = D_CKV / MMA_K; // 32 (k-iters for QK nope part)
static constexpr int NUM_MMA_D_KPE_K = D_KPE / MMA_K; // 4  (k-iters for QK pe part)
static constexpr int NUM_MMA_D_V_HALF = (D_V / 2) / 16; // 16 (d-tiles per warp_d for PV)

// ============================================================
// Shared memory layout sizes (in bytes)
// ============================================================
static constexpr int Q_NOPE_SIZE = BM * D_CKV * sizeof(bf16);                   // 64KB
static constexpr int Q_PE_SIZE   = BM * D_KPE * sizeof(bf16);                   // 8KB
static constexpr int CKV_STAGE_SIZE = BN * D_CKV * sizeof(bf16);                // 64KB
static constexpr int KPE_STAGE_SIZE = BN * D_KPE * sizeof(bf16);                // 8KB
static constexpr int SMEM_SIZE = Q_NOPE_SIZE + Q_PE_SIZE + 
                                 NUM_STAGES * (CKV_STAGE_SIZE + KPE_STAGE_SIZE) +
                                 2 * BM * sizeof(float); // 216KB + 512B for m_wg sync

// ============================================================
// Swizzle function: XOR-based for bank-conflict-free ldmatrix access
// STRIDE: row stride in bytes, col: in units of 16 bytes
// ============================================================
template <int STRIDE>
__device__ __forceinline__ int swizzle(int row, int col) {
    if constexpr (STRIDE >= 128)
        col ^= (row % 8) / (128 / STRIDE > 1 ? 128 / STRIDE : 1);
    return row * STRIDE + col * 16;
}

// ============================================================
// PTX Intrinsics
// ============================================================
__device__ __forceinline__ void ldmatrix_x4(uint32_t reg[4], int addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
        : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t reg[4], int addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
        : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    const uint32_t A[4], const uint32_t B[2], float C[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]));
}

__device__ __forceinline__ void mma_m16n8k16_bf16_init(
    const uint32_t A[4], const uint32_t B[2], float C[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
}

// Composite m16n16k16 (two m16n8k16)
__device__ __forceinline__ void mma_m16n16k16_bf16(
    const uint32_t A[4], const uint32_t B[4], float C[8]) {
    mma_m16n8k16_bf16(A, &B[0], &C[0]);
    mma_m16n8k16_bf16(A, &B[2], &C[4]);
}

__device__ __forceinline__ void mma_m16n16k16_bf16_init(
    const uint32_t A[4], const uint32_t B[4], float C[8]) {
    mma_m16n8k16_bf16_init(A, &B[0], &C[0]);
    mma_m16n8k16_bf16_init(A, &B[2], &C[4]);
}

// Row sum using mma: computes rowsum of a 16×16 bf16 matrix
// d[0] += rowsum for row (lane/4), d[1] += rowsum for row (lane/4+8)
__device__ __forceinline__ void mma_rowsum_bf16(float* d, uint32_t* s_u32) {
    // 1065369472 = 0x3F803F80 = bf16(1.0) packed twice
    asm volatile(
        "{\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, _, %1, _}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7}, "
        "{%0, 0., %1, 0.};\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1])
        : "r"(s_u32[0]), "r"(s_u32[1]), "r"(s_u32[2]), "r"(s_u32[3]),
          "r"(1065369472u), "r"(1065369472u));
}

__device__ __forceinline__ void cp_async_128b(int dst_smem_addr, const void* src_gmem) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst_smem_addr), "l"(src_gmem));
}

__device__ __forceinline__ void cp_async_128b_pred(int dst_smem_addr, const void* src_gmem, bool pred) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @!p st.shared.v4.u32 [%0], {0, 0, 0, 0};\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
        "}\n"
        :: "r"(dst_smem_addr), "l"(src_gmem), "r"((int)pred));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ float shfl_xor(float val, int mask) {
    return __shfl_xor_sync(0xffffffff, val, mask);
}

// exp2f approximation
__device__ __forceinline__ float fast_exp2f(float x) {
    float r;
    asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
    return r;
}

// Convert float to bf16 packed pair
__device__ __forceinline__ uint32_t float2_to_bf16x2(float a, float b) {
    bf16_2 val = __float22bfloat162_rn(make_float2(a, b));
    return *reinterpret_cast<uint32_t*>(&val);
}

// ============================================================
// Helper: compute ceil_div
// ============================================================
__host__ __device__ __forceinline__ int cdiv(int a, int b) { return (a + b - 1) / b; }

// ============================================================
// Main MLA Prefill Attention Kernel
// ============================================================
template <bool REVERSE_BLOCKS>
__global__ __launch_bounds__(NUM_THREADS)
void mla_prefill_attention_kernel(
    const bf16* __restrict__ Q_nope,  // [S, H, D_CKV]
    const bf16* __restrict__ Q_pe,    // [S, H, D_KPE]
    const bf16* __restrict__ CKV,     // [S, D_CKV]
    const bf16* __restrict__ KPE,     // [S, D_KPE]
    bf16* __restrict__ O,             // [S, H, D_V]
    const int S, const int H, const float sm_scale_log2)
{
    // Grid: (H, num_q_blocks, B)
    const int head    = blockIdx.x;
    // Compile-time block order: reverse for small S (better tail), forward for large S (better DRAM)
    const int num_q_blocks_total = cdiv(S, BM);
    const int q_block = REVERSE_BLOCKS ? (num_q_blocks_total - 1 - blockIdx.y) : blockIdx.y;
    const int q_start = q_block * BM;
    
    // Thread indices
    const int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 128;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_m  = warp_id / 2;   // 0-3, handles 16 Q rows
    const int warp_d  = warp_id % 2;   // 0-1, handles D_V/2 columns in output
    
    // ============================================================
    // Shared memory setup
    // ============================================================
    extern __shared__ __align__(128) uint8_t smem_raw[];
    
    // Layout within shared memory
    int smem_base = static_cast<int>(__cvta_generic_to_shared(smem_raw));
    int q_nope_smem = smem_base;
    int q_pe_smem   = q_nope_smem + Q_NOPE_SIZE;
    int ckv_smem_base = q_pe_smem + Q_PE_SIZE;
    int kpe_smem_base = ckv_smem_base + NUM_STAGES * CKV_STAGE_SIZE;
    
    auto ckv_smem = [&](int stage) { return ckv_smem_base + stage * CKV_STAGE_SIZE; };
    auto kpe_smem = [&](int stage) { return kpe_smem_base + stage * KPE_STAGE_SIZE; };
    
    // ============================================================
    // Step 1: Load Q into shared memory (cooperative)
    // ============================================================
    {
        // Q_nope: [S, H, D_CKV] → load rows [q_start : q_start+BM] for head
        // Each row: D_CKV=512 bf16 = 1024 bytes = 64 × 16-byte loads
        // Total loads: BM × 64 = 4096 loads. With 256 threads: 16 loads per thread
        constexpr int Q_NOPE_LOADS = (BM * D_CKV) / 8; // 8 bf16 per 16-byte load
        constexpr int Q_PE_LOADS   = (BM * D_KPE) / 8;
        constexpr int STRIDE_NOPE  = D_CKV * sizeof(bf16); // 1024
        constexpr int STRIDE_PE    = D_KPE * sizeof(bf16);  // 128
        
        for (int i = tid; i < Q_NOPE_LOADS; i += NUM_THREADS) {
            int row = i / (D_CKV / 8);
            int col = i % (D_CKV / 8);
            int q_idx = q_start + row;
            int smem_addr = q_nope_smem + swizzle<STRIDE_NOPE>(row, col);
            const bf16* gmem_ptr = Q_nope + (long long)q_idx * H * D_CKV + (long long)head * D_CKV + col * 8;
            if (q_idx < S) {
                cp_async_128b(smem_addr, gmem_ptr);
            } else {
                // Zero-fill for out-of-bounds
                asm volatile("st.shared.v4.u32 [%0], {0, 0, 0, 0};\n" :: "r"(smem_addr));
            }
        }
        
        for (int i = tid; i < Q_PE_LOADS; i += NUM_THREADS) {
            int row = i / (D_KPE / 8);
            int col = i % (D_KPE / 8);
            int q_idx = q_start + row;
            int smem_addr = q_pe_smem + swizzle<STRIDE_PE>(row, col);
            const bf16* gmem_ptr = Q_pe + (long long)q_idx * H * D_KPE + (long long)head * D_KPE + col * 8;
            if (q_idx < S) {
                cp_async_128b(smem_addr, gmem_ptr);
            } else {
                asm volatile("st.shared.v4.u32 [%0], {0, 0, 0, 0};\n" :: "r"(smem_addr));
            }
        }
        
        cp_async_commit();
        cp_async_wait<0>();
        __syncthreads();
    }
    
    // ============================================================
    // Pre-compute ldmatrix base addresses for Q
    // ============================================================
    constexpr int STRIDE_NOPE_B = D_CKV * sizeof(bf16);  // 1024
    constexpr int STRIDE_PE_B   = D_KPE * sizeof(bf16);  // 128
    // For Q_nope (stride = 1024 bytes = D_CKV * 2):
    // A operand base: row = warp_m*16 + lane%16, col = lane/16
    int q_nope_ldm_base = q_nope_smem + swizzle<STRIDE_NOPE_B>(warp_m * 16 + (lane_id % 16), lane_id / 16);
    // For Q_pe (stride = 128 bytes = D_KPE * 2):
    int q_pe_ldm_base = q_pe_smem + swizzle<STRIDE_PE_B>(warp_m * 16 + (lane_id % 16), lane_id / 16);
    
    // ============================================================
    // Initialize accumulators
    // ============================================================
    // O accumulator: NUM_MMA_D_V_HALF × 8 floats per warp
    float o_frag[NUM_MMA_D_V_HALF][8]; // 16 tiles × 8 = 128 floats
    #pragma unroll
    for (int i = 0; i < NUM_MMA_D_V_HALF; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++)
            o_frag[i][j] = 0.0f;
    
    // Online softmax state: m (max), d (sum of exp)
    // j=0: row (lane/4), j=1: row (lane/4+8)
    float m_state[2] = {-INFINITY, -INFINITY};
    float d_state[2] = {1.0f, 1.0f};
    
    // S accumulator: NUM_N_SHARD × 8 floats (QK_SHARD: each warp_d handles half)
    constexpr int NUM_N_SHARD_GLOBAL = NUM_MMA_N16 / 2; // 2
    float s_frag[NUM_N_SHARD_GLOBAL][8];
    
    // ============================================================
    // Determine KV tile range with causal masking
    // ============================================================
    // For causal: kv_pos <= q_pos. Last valid kv_pos for this Q block = q_start + BM - 1
    int kv_end = S; // Without causal
    if (true /* causal */) {
        kv_end = min(S, q_start + BM); // Last Q row can attend to kv_pos = q_start + BM - 1
    }
    int num_kv_tiles = cdiv(kv_end, BN);
    // Safe tiles: all kv positions <= all q positions (no mask needed)
    // Tile kv_tile is safe if (kv_tile+1)*BN - 1 < q_start, i.e., (kv_tile+1)*BN <= q_start
    int num_safe_tiles = q_start / BN;
    
    // ============================================================
    // Prefetch first KV tile(s)
    // ============================================================
    auto load_kv_tile = [&](int kv_tile, int stage) {
        int kv_base = kv_tile * BN;
        constexpr int CKV_LOADS = (BN * D_CKV) / 8;
        constexpr int KPE_LOADS = (BN * D_KPE) / 8;
        constexpr int STRIDE_CKV = D_CKV * sizeof(bf16); // 1024
        constexpr int STRIDE_KPE_B = D_KPE * sizeof(bf16); // 128
        
        // Fast path: all rows valid (no boundary check needed)
        if (kv_base + BN <= S) {
            for (int i = tid; i < CKV_LOADS; i += NUM_THREADS) {
                int row = i / (D_CKV / 8);
                int col = i % (D_CKV / 8);
                int kv_idx = kv_base + row;
                int addr = ckv_smem(stage) + swizzle<STRIDE_CKV>(row, col);
                const bf16* ptr = CKV + (long long)kv_idx * D_CKV + col * 8;
                cp_async_128b(addr, ptr);
            }
            
            for (int i = tid; i < KPE_LOADS; i += NUM_THREADS) {
                int row = i / (D_KPE / 8);
                int col = i % (D_KPE / 8);
                int kv_idx = kv_base + row;
                int addr = kpe_smem(stage) + swizzle<STRIDE_KPE_B>(row, col);
                const bf16* ptr = KPE + (long long)kv_idx * D_KPE + col * 8;
                cp_async_128b(addr, ptr);
            }
        } else {
            // Slow path: boundary tile with validity checks
            for (int i = tid; i < CKV_LOADS; i += NUM_THREADS) {
                int row = i / (D_CKV / 8);
                int col = i % (D_CKV / 8);
                int kv_idx = kv_base + row;
                int addr = ckv_smem(stage) + swizzle<STRIDE_CKV>(row, col);
                const bf16* ptr = CKV + (long long)kv_idx * D_CKV + col * 8;
                bool valid = (kv_idx < S);
                cp_async_128b_pred(addr, ptr, valid);
            }
            
            for (int i = tid; i < KPE_LOADS; i += NUM_THREADS) {
                int row = i / (D_KPE / 8);
                int col = i % (D_KPE / 8);
                int kv_idx = kv_base + row;
                int addr = kpe_smem(stage) + swizzle<STRIDE_KPE_B>(row, col);
                const bf16* ptr = KPE + (long long)kv_idx * D_KPE + col * 8;
                bool valid = (kv_idx < S);
                cp_async_128b_pred(addr, ptr, valid);
            }
        }
        
        cp_async_commit();
    };
    
    // Load first tile only (single-stage look-ahead)
    if (num_kv_tiles > 0)
        load_kv_tile(0, 0);
    
    // ============================================================
    // Main loop over KV tiles
    // ============================================================
    constexpr int STRIDE_CKV_B  = D_CKV * sizeof(bf16);  // 1024
    constexpr int STRIDE_KPE_B2 = D_KPE * sizeof(bf16);  // 128
    
    // Pre-compute loop-invariant K address components
    const int k_row = (lane_id % 8) + (lane_id / 16) * 8;
    const int k_col_base = (lane_id % 16) / 8;
    constexpr int NUM_N_SHARD = NUM_MMA_N16 / 2; // 2
    const int n_offset = warp_d * NUM_N_SHARD; // 0 or 2
    
    #pragma unroll 1
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int stage = kv_tile % NUM_STAGES;
        int kv_base = kv_tile * BN;
        
        // Wait for current tile to be loaded
        cp_async_wait<0>();
        __syncthreads();
        
        // Prefetch next tile into the OTHER stage (overlaps with compute)
        {
            int next_tile = kv_tile + 1;
            if (next_tile < num_kv_tiles) {
                load_kv_tile(next_tile, next_tile % NUM_STAGES);
            }
        }
        
        // Fused QK: interleave Q_pe×KPE^T and Q_nope×CKV^T
        {
            int kpe_ldm_base_addr = kpe_smem(stage) + swizzle<STRIDE_KPE_B2>(k_row, k_col_base);
            int ckv_ldm_base_addr = ckv_smem(stage) + swizzle<STRIDE_CKV_B>(k_row, k_col_base);
            
            // First 4 iterations: compute both PE and CKV parts
            #pragma unroll
            for (int mma_k = 0; mma_k < NUM_MMA_D_KPE_K; mma_k++) {
                // PE part
                {
                    uint32_t q_reg[4];
                    int q_addr = q_pe_ldm_base ^ (mma_k * 32);
                    ldmatrix_x4(q_reg, q_addr);
                    
                    #pragma unroll
                    for (int nl = 0; nl < NUM_N_SHARD; nl++) {
                        uint32_t k_reg[4];
                        int mma_n = n_offset + nl;
                        int k_addr = kpe_ldm_base_addr + mma_n * 16 * STRIDE_KPE_B2;
                        ldmatrix_x4(k_reg, k_addr ^ (mma_k * 32));
                        
                        if (mma_k == 0) {
                            mma_m16n16k16_bf16_init(q_reg, k_reg, s_frag[nl]);
                        } else {
                            mma_m16n16k16_bf16(q_reg, k_reg, s_frag[nl]);
                        }
                    }
                }
                // CKV part (same mma_k index)
                {
                    uint32_t q_reg[4];
                    int q_addr = q_nope_ldm_base ^ (mma_k * 32);
                    ldmatrix_x4(q_reg, q_addr);
                    
                    #pragma unroll
                    for (int nl = 0; nl < NUM_N_SHARD; nl++) {
                        uint32_t k_reg[4];
                        int mma_n = n_offset + nl;
                        int k_addr = ckv_ldm_base_addr + mma_n * 16 * STRIDE_CKV_B;
                        ldmatrix_x4(k_reg, k_addr ^ (mma_k * 32));
                        
                        mma_m16n16k16_bf16(q_reg, k_reg, s_frag[nl]);
                    }
                }
            }
            
            // Remaining 28 iterations: CKV only
            #pragma unroll
            for (int mma_k = NUM_MMA_D_KPE_K; mma_k < NUM_MMA_D_CKV_K; mma_k++) {
                uint32_t q_reg[4];
                int q_addr = q_nope_ldm_base ^ (mma_k * 32);
                ldmatrix_x4(q_reg, q_addr);
                
                #pragma unroll
                for (int nl = 0; nl < NUM_N_SHARD; nl++) {
                    uint32_t k_reg[4];
                    int mma_n = n_offset + nl;
                    int k_addr = ckv_ldm_base_addr + mma_n * 16 * STRIDE_CKV_B;
                    ldmatrix_x4(k_reg, k_addr ^ (mma_k * 32));
                    
                    mma_m16n16k16_bf16(q_reg, k_reg, s_frag[nl]);
                }
            }
        }
        
        // ----------------------------------------------------------
        // Causal Masking (skip for fully safe tiles)
        // ----------------------------------------------------------
        if (kv_base + BN > q_start) {
            int q_row_base = q_start + warp_m * 16;
            #pragma unroll
            for (int nl = 0; nl < NUM_N_SHARD; nl++) {
                int mma_n_global = n_offset + nl;
                #pragma unroll
                for (int reg_id = 0; reg_id < 8; reg_id++) {
                    int row_in_tile = ((reg_id & 2) == 0) ? (lane_id / 4) : (lane_id / 4 + 8);
                    int kv_col = 2 * (lane_id % 4) + ((reg_id & 4) ? 8 : 0) + (reg_id & 1);
                    
                    int q_pos = q_row_base + row_in_tile;
                    int kv_pos = kv_base + mma_n_global * 16 + kv_col;
                    
                    if (!((kv_pos <= q_pos) && (kv_pos < S))) {
                        s_frag[nl][reg_id] = -INFINITY;
                    }
                }
            }
        }
        
        // ----------------------------------------------------------
        // Online Softmax Update (QK_SHARD: cross-warpgroup reduction via SMEM)
        // ----------------------------------------------------------
        // Use SMEM for cross-warpgroup max synchronization (m_wg buffer at end of base SMEM)
        constexpr int M_WG_OFF = Q_NOPE_SIZE + Q_PE_SIZE + NUM_STAGES * (CKV_STAGE_SIZE + KPE_STAGE_SIZE);
        float* m_wg = reinterpret_cast<float*>(smem_raw + M_WG_OFF);
        {
            float m_prev[2] = {m_state[0], m_state[1]};
            
            // Compute partial row max from local s_frag tiles
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                #pragma unroll
                for (int nl = 0; nl < NUM_N_SHARD; nl++) {
                    float local_max = fmaxf(
                        fmaxf(s_frag[nl][j * 2 + 0], s_frag[nl][j * 2 + 1]),
                        fmaxf(s_frag[nl][j * 2 + 4], s_frag[nl][j * 2 + 5]));
                    m_state[j] = fmaxf(m_state[j], local_max);
                }
                m_state[j] = fmaxf(m_state[j], shfl_xor(m_state[j], 0x2));
                m_state[j] = fmaxf(m_state[j], shfl_xor(m_state[j], 0x1));
                
                // Write partial max to SMEM: m_wg[warp_d][warp_m*16 + row]
                if (lane_id % 4 == 0) {
                    m_wg[warp_d * BM + warp_m * 16 + j * 8 + lane_id / 4] = m_state[j];
                }
            }
            
            // Partial barrier: only sync the 2 warps in this warp_m group (64 threads)
            // barrier_id = 1 + warp_m (barriers 1-4, barrier 0 is reserved for __syncthreads)
            asm volatile("bar.sync %0, 64;" :: "r"(1 + warp_m));
            
            // Read both warp_d maxes and compute global max
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                m_state[j] = fmaxf(
                    m_wg[0 * BM + warp_m * 16 + j * 8 + lane_id / 4],
                    m_wg[1 * BM + warp_m * 16 + j * 8 + lane_id / 4]);
                
                float neg_m_scaled = -(m_state[j] * sm_scale_log2);
                float scale = fast_exp2f(__fmaf_rn(m_prev[j], sm_scale_log2, neg_m_scaled));
                d_state[j] *= scale;
                #pragma unroll
                for (int mma_d = 0; mma_d < NUM_MMA_D_V_HALF; mma_d++) {
                    o_frag[mma_d][j * 2 + 0] *= scale;
                    o_frag[mma_d][j * 2 + 1] *= scale;
                    o_frag[mma_d][j * 2 + 4] *= scale;
                    o_frag[mma_d][j * 2 + 5] *= scale;
                }
                #pragma unroll
                for (int nl = 0; nl < NUM_N_SHARD; nl++) {
                    s_frag[nl][j * 2 + 0] = fast_exp2f(__fmaf_rn(s_frag[nl][j * 2 + 0], sm_scale_log2, neg_m_scaled));
                    s_frag[nl][j * 2 + 1] = fast_exp2f(__fmaf_rn(s_frag[nl][j * 2 + 1], sm_scale_log2, neg_m_scaled));
                    s_frag[nl][j * 2 + 4] = fast_exp2f(__fmaf_rn(s_frag[nl][j * 2 + 4], sm_scale_log2, neg_m_scaled));
                    s_frag[nl][j * 2 + 5] = fast_exp2f(__fmaf_rn(s_frag[nl][j * 2 + 5], sm_scale_log2, neg_m_scaled));
                }
            }
        }
        
        // ----------------------------------------------------------
        // Convert S to P (bf16) and compute row sum for d
        // Store P to SMEM for PV (using KPE buffer which is no longer needed)
        // ----------------------------------------------------------
        uint32_t p_f16_local[NUM_N_SHARD][4];
        #pragma unroll
        for (int nl = 0; nl < NUM_N_SHARD; nl++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                p_f16_local[nl][i] = float2_to_bf16x2(s_frag[nl][i * 2], s_frag[nl][i * 2 + 1]);
            }
            mma_rowsum_bf16(d_state, p_f16_local[nl]);
        }
        
        // Store P to KPE SMEM buffer (reused) as [BM, BN] bf16 with P swizzle
        // P layout: row = warp_m*16 + mma_row, col = n_offset * 16 + mma_col
        // Using 128B stride for BN=64 bf16 = 128 bytes
        constexpr int P_STRIDE = BN * sizeof(bf16); // 128
        int p_smem = kpe_smem(stage); // Reuse KPE buffer for P
        
        // Store local P tiles using st.shared (inline PTX for proper SMEM access)
        #pragma unroll
        for (int nl = 0; nl < NUM_N_SHARD; nl++) {
            int mma_n_global = n_offset + nl;
            int p_row = warp_m * 16 + lane_id / 4;
            int p_col = mma_n_global * 16 + 2 * (lane_id % 4);
            int p_col2 = mma_n_global * 16 + 2 * (lane_id % 4) + 8;
            
            // Use swizzled addresses for bank-conflict-free access
            // p_col and p_col2 are in bf16 elements; convert to 16-byte col units
            int col_unit = p_col / 8;  // 16-byte unit
            int col_unit2 = p_col2 / 8;
            int col_off = (p_col % 8) * (int)sizeof(bf16); // offset within 16-byte unit
            int col_off2 = (p_col2 % 8) * (int)sizeof(bf16);
            
            int a0 = p_smem + swizzle<P_STRIDE>(p_row, col_unit) + col_off;
            asm volatile("st.shared.u32 [%0], %1;" :: "r"(a0), "r"(p_f16_local[nl][0]));
            int a1 = p_smem + swizzle<P_STRIDE>(p_row + 8, col_unit) + col_off;
            asm volatile("st.shared.u32 [%0], %1;" :: "r"(a1), "r"(p_f16_local[nl][1]));
            int a2 = p_smem + swizzle<P_STRIDE>(p_row, col_unit2) + col_off2;
            asm volatile("st.shared.u32 [%0], %1;" :: "r"(a2), "r"(p_f16_local[nl][2]));
            int a3 = p_smem + swizzle<P_STRIDE>(p_row + 8, col_unit2) + col_off2;
            asm volatile("st.shared.u32 [%0], %1;" :: "r"(a3), "r"(p_f16_local[nl][3]));
        }
        
        // Partial barrier for P sync (same warp_m group, 64 threads)
        asm volatile("bar.sync %0, 64;" :: "r"(1 + warp_m));
        
        // ----------------------------------------------------------
        // PV Matmul: O += P × V (using P from SMEM, V from CKV SMEM)
        // ----------------------------------------------------------
        {
            int v_row = lane_id % 16;
            int v_col_base = warp_d * (D_CKV / 2 / 8) + lane_id / 16;
            
            // Load P from SMEM: P[warp_m*16 + ..., mma_kv*16 + ...]
            // P is stored as [BM, BN] without swizzle (simple row-major)
            // For ldmatrix to load P as A operand: 
            // A base: row = warp_m*16 + lane%16, col = lane/16 (in 16-byte = 8 bf16 units)
            int p_ldm_row = warp_m * 16 + (lane_id % 16);
            int p_ldm_col = lane_id / 16; // 0 or 1 (covers 16 bf16 = 32 bytes = 2 col-units)
            
            #pragma unroll
            for (int mma_kv = 0; mma_kv < NUM_MMA_KV; mma_kv++) {
                // Load P tile for this mma_kv with swizzle
                uint32_t p_frag[4];
                int p_addr = p_smem + swizzle<P_STRIDE>(p_ldm_row, mma_kv * 2 + p_ldm_col);
                ldmatrix_x4(p_frag, p_addr);
                
                #pragma unroll
                for (int mma_d = 0; mma_d < NUM_MMA_D_V_HALF; mma_d++) {
                    uint32_t v_frag[4];
                    int v_r = v_row + mma_kv * 16;
                    int v_c = v_col_base + mma_d * 2;
                    int v_addr = ckv_smem(stage) + swizzle<STRIDE_CKV_B>(v_r, v_c);
                    
                    ldmatrix_x4_trans(v_frag, v_addr);
                    mma_m16n16k16_bf16(p_frag, v_frag, o_frag[mma_d]);
                }
            }
        }
        
        // (prefetch is at the top of the loop, overlapping with compute)
    }
    
    // ============================================================
    // Normalize O by d (need cross-warpgroup d reduction)
    // ============================================================
    {
        // d_state needs to be summed across warp_d groups
        constexpr int DWG_OFF = Q_NOPE_SIZE + Q_PE_SIZE + NUM_STAGES * (CKV_STAGE_SIZE + KPE_STAGE_SIZE);
        float* d_wg_smem = reinterpret_cast<float*>(smem_raw + DWG_OFF);
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            if (lane_id % 4 == 0) {
                d_wg_smem[warp_d * BM + warp_m * 16 + j * 8 + lane_id / 4] = d_state[j];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            d_state[j] = d_wg_smem[0 * BM + warp_m * 16 + j * 8 + lane_id / 4] +
                          d_wg_smem[1 * BM + warp_m * 16 + j * 8 + lane_id / 4];
        }
        
        float d_rcp[2];
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            if (m_state[j] != -INFINITY) {
                asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(d_rcp[j]) : "f"(d_state[j]));
            } else {
                d_rcp[j] = 0.0f;
            }
        }
        
        #pragma unroll
        for (int mma_d = 0; mma_d < NUM_MMA_D_V_HALF; mma_d++) {
            #pragma unroll
            for (int reg_id = 0; reg_id < 8; reg_id++) {
                int j = (reg_id % 4) / 2;
                o_frag[mma_d][reg_id] *= d_rcp[j];
            }
        }
    }
    
    // ============================================================
    // Write O to global memory
    // ============================================================
    {
        // O is [S, H, D_V], each warp writes its 16 rows × 256 cols
        // Output layout: O[q_start + warp_m*16 + row, head, warp_d*256 + col]
        
        // From mma output C[8]: 
        // C[0] = O[g, 2t], C[1] = O[g, 2t+1], C[2] = O[g+8, 2t], C[3] = O[g+8, 2t+1]
        // C[4] = O[g, 2t+8], C[5] = O[g, 2t+9], C[6] = O[g+8, 2t+8], C[7] = O[g+8, 2t+9]
        
        int g = lane_id / 4;
        int t = lane_id % 4;
        
        #pragma unroll
        for (int mma_d = 0; mma_d < NUM_MMA_D_V_HALF; mma_d++) {
            int d_base = warp_d * (D_V / 2) + mma_d * 16;
            
            // Row g
            {
                int q_pos = q_start + warp_m * 16 + g;
                if (q_pos < S) {
                    long long base_offset = (long long)q_pos * H * D_V + (long long)head * D_V + d_base;
                    
                    // Cols 2*t, 2*t+1
                    bf16_2 val0 = __float22bfloat162_rn(make_float2(o_frag[mma_d][0], o_frag[mma_d][1]));
                    *reinterpret_cast<bf16_2*>(&O[base_offset + 2 * t]) = val0;
                    
                    // Cols 2*t+8, 2*t+9  
                    bf16_2 val4 = __float22bfloat162_rn(make_float2(o_frag[mma_d][4], o_frag[mma_d][5]));
                    *reinterpret_cast<bf16_2*>(&O[base_offset + 2 * t + 8]) = val4;
                }
            }
            
            // Row g+8
            {
                int q_pos = q_start + warp_m * 16 + g + 8;
                if (q_pos < S) {
                    long long base_offset = (long long)q_pos * H * D_V + (long long)head * D_V + d_base;
                    
                    bf16_2 val2 = __float22bfloat162_rn(make_float2(o_frag[mma_d][2], o_frag[mma_d][3]));
                    *reinterpret_cast<bf16_2*>(&O[base_offset + 2 * t]) = val2;
                    
                    bf16_2 val6 = __float22bfloat162_rn(make_float2(o_frag[mma_d][6], o_frag[mma_d][7]));
                    *reinterpret_cast<bf16_2*>(&O[base_offset + 2 * t + 8]) = val6;
                }
            }
        }
    }
}

// ============================================================
// Host launcher
// ============================================================
void launch_mla_prefill(
    const bf16* Q_nope, const bf16* Q_pe,
    const bf16* CKV, const bf16* KPE,
    bf16* O,
    int B, int S, int H, float sm_scale,
    cudaStream_t stream = 0)
{
    int num_q_blocks = cdiv(S, BM);
    // Interleave head and q_block dims: head varies fastest for L2 locality on KV cache
    // Use (num_q_blocks, H, B) grid with swapped head/q_block vs kernel indexing
    dim3 grid(H, num_q_blocks, B);
    dim3 block(32, 4, 2); // 256 threads = 8 warps
    
    int smem_size = SMEM_SIZE;
    float sm_scale_log2 = sm_scale * 1.44269504089f; // log2(e) = 1.4427
    
    // Use reverse block order for small S (better tail occupancy), forward for large S (better DRAM access)
    if (S <= 2048) {
        auto kernel = mla_prefill_attention_kernel<true>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        kernel<<<grid, block, smem_size, stream>>>(
            Q_nope, Q_pe, CKV, KPE, O, S, H, sm_scale_log2);
    } else {
        auto kernel = mla_prefill_attention_kernel<false>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        kernel<<<grid, block, smem_size, stream>>>(
            Q_nope, Q_pe, CKV, KPE, O, S, H, sm_scale_log2);
    }
}

// ============================================================
// File I/O helpers for validation
// ============================================================
bool load_file(const char* fname, void* dst, size_t bytes) {
    FILE* f = fopen(fname, "rb");
    if (!f) return false;
    size_t read = fread(dst, 1, bytes, f);
    fclose(f);
    return read == bytes;
}

bool save_file(const char* fname, const void* src, size_t bytes) {
    FILE* f = fopen(fname, "wb");
    if (!f) return false;
    fwrite(src, 1, bytes, f);
    fclose(f);
    return true;
}

// ============================================================
// Benchmark main
// ============================================================
int main(int argc, char** argv) {
    int B = 1, S = 1024, H = 128;
    bool validate = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--validate") == 0) {
            validate = true;
        } else if (i + 1 < argc && !validate) {
            S = atoi(argv[i]);
            H = atoi(argv[i + 1]);
            i++;
        } else {
            S = atoi(argv[i]);
        }
    }
    
    float sm_scale = 1.0f / sqrtf(576.0f);
    
    printf("=== MLA Prefill Attention Kernel (B200) ===\n");
    printf("B=%d, S=%d, H=%d, D_CKV=%d, D_KPE=%d, D_V=%d\n", B, S, H, D_CKV, D_KPE, D_V);
    
    // Allocate tensors
    bf16 *d_Q_nope, *d_Q_pe, *d_CKV, *d_KPE, *d_O;
    
    size_t q_nope_bytes = (size_t)B * S * H * D_CKV * sizeof(bf16);
    size_t q_pe_bytes   = (size_t)B * S * H * D_KPE * sizeof(bf16);
    size_t ckv_bytes    = (size_t)B * S * D_CKV * sizeof(bf16);
    size_t kpe_bytes    = (size_t)B * S * D_KPE * sizeof(bf16);
    size_t o_bytes      = (size_t)B * S * H * D_V * sizeof(bf16);
    
    cudaMalloc(&d_Q_nope, q_nope_bytes);
    cudaMalloc(&d_Q_pe, q_pe_bytes);
    cudaMalloc(&d_CKV, ckv_bytes);
    cudaMalloc(&d_KPE, kpe_bytes);
    cudaMalloc(&d_O, o_bytes);
    
    if (validate) {
        // Load from files
        bf16* h_buf;
        h_buf = (bf16*)malloc(q_nope_bytes);
        if (load_file("/tmp/q_nope.bin", h_buf, q_nope_bytes)) {
            cudaMemcpy(d_Q_nope, h_buf, q_nope_bytes, cudaMemcpyHostToDevice);
        }
        
        h_buf = (bf16*)realloc(h_buf, q_pe_bytes);
        if (load_file("/tmp/q_pe.bin", h_buf, q_pe_bytes)) {
            cudaMemcpy(d_Q_pe, h_buf, q_pe_bytes, cudaMemcpyHostToDevice);
        }
        
        h_buf = (bf16*)realloc(h_buf, ckv_bytes);
        if (load_file("/tmp/ckv.bin", h_buf, ckv_bytes)) {
            cudaMemcpy(d_CKV, h_buf, ckv_bytes, cudaMemcpyHostToDevice);
        }
        
        h_buf = (bf16*)realloc(h_buf, kpe_bytes);
        if (load_file("/tmp/kpe.bin", h_buf, kpe_bytes)) {
            cudaMemcpy(d_KPE, h_buf, kpe_bytes, cudaMemcpyHostToDevice);
        }
        free(h_buf);
        
        cudaMemset(d_O, 0, o_bytes);
        
        launch_mla_prefill(d_Q_nope, d_Q_pe, d_CKV, d_KPE, d_O, B, S, H, sm_scale);
        cudaDeviceSynchronize();
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return 1;
        }
        
        // Save output
        bf16* h_out = (bf16*)malloc(o_bytes);
        cudaMemcpy(h_out, d_O, o_bytes, cudaMemcpyDeviceToHost);
        save_file("/tmp/o_kernel.bin", h_out, o_bytes);
        
        // Load reference and compare
        bf16* h_ref = (bf16*)malloc(o_bytes);
        if (load_file("/tmp/o_ref.bin", h_ref, o_bytes)) {
            float max_err = 0, mean_err = 0;
            int count = 0;
            for (size_t i = 0; i < S * H * D_V; i++) {
                float a = __bfloat162float(h_out[i]);
                float b = __bfloat162float(h_ref[i]);
                float err_val = fabsf(a - b);
                max_err = fmaxf(max_err, err_val);
                mean_err += err_val;
                count++;
            }
            mean_err /= count;
            printf("Validation: max_err=%.6f, mean_err=%.6f\n", max_err, mean_err);
            
            // Print first few values
            printf("Kernel O[0,0,:8]: ");
            for (int i = 0; i < 8; i++) printf("%.4f ", __bfloat162float(h_out[i]));
            printf("\n");
            printf("Ref    O[0,0,:8]: ");
            for (int i = 0; i < 8; i++) printf("%.4f ", __bfloat162float(h_ref[i]));
            printf("\n");
        } else {
            printf("Could not load reference file\n");
        }
        free(h_out);
        free(h_ref);
    } else {
        // Benchmark mode
        cudaMemset(d_Q_nope, 0, q_nope_bytes);
        cudaMemset(d_Q_pe, 0, q_pe_bytes);
        cudaMemset(d_CKV, 0, ckv_bytes);
        cudaMemset(d_KPE, 0, kpe_bytes);
        cudaMemset(d_O, 0, o_bytes);
        
        // Warmup (extended for clock stability)
        for (int i = 0; i < 30; i++) {
            launch_mla_prefill(d_Q_nope, d_Q_pe, d_CKV, d_KPE, d_O, B, S, H, sm_scale);
        }
        cudaDeviceSynchronize();
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return 1;
        }
        
        int N_iter = 50;
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        
        cudaEventRecord(start);
        for (int i = 0; i < N_iter; i++) {
            launch_mla_prefill(d_Q_nope, d_Q_pe, d_CKV, d_KPE, d_O, B, S, H, sm_scale);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        
        float ms_total;
        cudaEventElapsedTime(&ms_total, start, end);
        float ms = ms_total / N_iter;
        
        double flops = (double)B * H * S * S * (D_CKV + D_KPE + D_CKV);
        double tflops = flops / (ms / 1000.0) / 1e12;
        
        printf("Latency: %.1f us, TFLOPS: %.1f\n", ms * 1000.0, tflops);
    }
    
    cudaFree(d_Q_nope);
    cudaFree(d_Q_pe);
    cudaFree(d_CKV);
    cudaFree(d_KPE);
    cudaFree(d_O);
    
    return 0;
}
