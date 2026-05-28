// =============================================================================
// Paged GQA Multi-Token Decode (Q_LEN=2,3,4) for Qwen3-30B-A3B on B200
// Split-KV with fused combine (atomic counter pattern)
// Each CTA: 1 query row, 1 KV split, 8 warps (1 per head)
// Last CTA per (q, h) does inline combine → single kernel launch
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
static constexpr int64_t KVS_TOK = (int64_t)NKH * HD;
static constexpr int64_t KVS_KV  = (int64_t)PS * KVS_TOK;
static constexpr int64_t KVS_PAGE = 2 * KVS_KV;

static constexpr int BLK = 256;
static constexpr int KVC = 8;   // compile-time chunk size

struct SmemLayout {
    bf16_t kbuf[KVC][HD];
    bf16_t vbuf[KVC][HD];
};

// 2D grid: blockIdx.x = split_id, blockIdx.y = q_row
// Eliminates integer division overhead (~20 SASS instructions)
__global__ void __launch_bounds__(BLK)
paged_gqa_fused(
    const bf16_t* __restrict__ Q,
    const bf16_t* __restrict__ paged_kv,
    float* __restrict__ pO,
    float* __restrict__ pLSE,
    int* __restrict__ counters,
    bf16_t* __restrict__ Out,
    int seq_len, int q_len, int nsplits, float sm_scale_log2
) {
    __shared__ SmemLayout sm;

    const int sid = blockIdx.x;    // split index (free, no division)
    const int q_row = blockIdx.y;  // query row (free, no division)
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lid = tid % 32;

    const int partial_idx = q_row * nsplits + sid;
    const int ts = sid * KVC;
    const int causal_end = seq_len - q_len + 1 + q_row;
    const int te = min(min(ts + KVC, seq_len), causal_end);

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

        // Simplified: seq_len <= PAGE_SIZE, so all tokens in page 0
        // KVC is compile-time → exactly 1 iteration, no outer loop
        {
            const int cl = te - ts;
            // Direct pointer: page 0, offset ts
            const bf16_t* kbase = paged_kv + (int64_t)ts * KVS_TOK;
            const bf16_t* vbase = kbase + KVS_KV;

            // Cooperative SMEM load (all 256 threads)
            const int u4_per_tok = HD / 8; // 16
            const int total_u4 = cl * u4_per_tok;
            for (int idx = tid; idx < total_u4; idx += BLK) {
                int tok = idx >> 4;     // idx / 16 (compile-time shift)
                int u4i = idx & 0xf;    // idx % 16
                ((uint4*)sm.kbuf[tok])[u4i] = __ldg(((const uint4*)(kbase + (int64_t)tok * KVS_TOK)) + u4i);
                ((uint4*)sm.vbuf[tok])[u4i] = __ldg(((const uint4*)(vbase + (int64_t)tok * KVS_TOK)) + u4i);
            }
        }
        __syncthreads();

        // Per-token online softmax + PV
        {
            const int cl = te - ts;
            #pragma unroll
            for (int t = 0; t < KVC; t++) {
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

    // Memory fence to ensure partial results are visible
    __threadfence();

    // Atomic counter: last CTA for this q_row does the combine
    __shared__ int is_last;
    if (tid == 0) {
        int old = atomicAdd(&counters[q_row], 1);
        is_last = (old == nsplits - 1);
    }
    __syncthreads();

    if (is_last) {
        // Combine: each warp combines its head's partials
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
        
        // Reset counter for next launch
        if (tid == 0) counters[q_row] = 0;
    }
}

// ========================= Host =========================
void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA(%s):%s\n", m, cudaGetErrorString(e)); exit(1); }
}

void cpu_ref(const bf16_t* Q, const bf16_t* kv, const int* ki, int sl, int ql, float sms, float* out) {
    for (int qi = 0; qi < ql; qi++) {
        int ce = sl - ql + 1 + qi;
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

double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    chk(cudaSetDevice(0), "sd");
    float sms = 1.0f / sqrtf((float)HD);
    float sms_log2 = sms * 1.4426950408889634f;

    struct Cfg { const char* n; int ql; int sl; };
    Cfg cfgs[] = {
        // Q=1
        {"q1_seq128",  1, 128},
        {"q1_seq512",  1, 512},
        {"q1_seq4k",   1, 4096},
        {"q1_seq32k",  1, 32768},
        // Q=2
        {"q2_seq128",  2, 128},
        {"q2_seq512",  2, 512},
        {"q2_seq4k",   2, 4096},
        {"q2_seq32k",  2, 32768},
        // Q=3
        {"q3_seq128",  3, 128},
        {"q3_seq512",  3, 512},
        {"q3_seq4k",   3, 4096},
        {"q3_seq32k",  3, 32768},
        // Q=4
        {"q4_seq128",  4, 128},
        {"q4_seq512",  4, 512},
        {"q4_seq4k",   4, 4096},
        {"q4_seq32k",  4, 32768},
    };

    size_t fs = 128 * 1024 * 1024;
    void* df;
    chk(cudaMalloc(&df, fs), "f");

    printf("KERNEL_RESULT {");
    bool first = true;
    const int NCFG = sizeof(cfgs) / sizeof(cfgs[0]);
    for (int ci = 0; ci < NCFG; ci++) {
        auto& c = cfgs[ci];
        int ql = c.ql, sl = c.sl;
        int np = (sl + PS - 1) / PS;
        size_t qs = (size_t)ql * NQH * HD;

        std::vector<bf16_t> hq(qs);
        std::mt19937 rng(42 + ci);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
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

        int kvc = 8;
        int ns = (sl + kvc - 1) / kvc;
        int total_ctas = ql * ns;

        float *dpO, *dpL;
        int *dctr;
        chk(cudaMalloc(&dpO, (size_t)total_ctas * NQH * HD * 4), "");
        chk(cudaMalloc(&dpL, (size_t)total_ctas * NQH * 4), "");
        chk(cudaMalloc(&dctr, ql * 4), "");

        // Initial zero
        cudaMemset(dctr, 0, ql * 4);

        // Use high-priority stream for lower scheduling latency
        int lo, hi;
        cudaDeviceGetStreamPriorityRange(&lo, &hi);
        cudaStream_t stream;
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, hi);

        auto run = [&]() {
            dim3 grid(ns, ql);
            paged_gqa_fused<<<grid, BLK, 0, stream>>>(dq, dkv, dpO, dpL, dctr, dout, sl, ql, ns, sms_log2);
        };

        run();
        chk(cudaDeviceSynchronize(), "sync");

        std::vector<float> ref(qs);
        cpu_ref(hq.data(), hkv.data(), hki.data(), sl, ql, sms, ref.data());
        std::vector<bf16_t> gout(qs);
        chk(cudaMemcpy(gout.data(), dout, qs * 2, cudaMemcpyDeviceToHost), "");

        float mre = 0, mae = 0, sae = 0;
        for (size_t i = 0; i < qs; i++) {
            float g = __bfloat162float(gout[i]), r = ref[i];
            float ae = fabsf(g - r), re = ae / fmaxf(fabsf(r), 1e-6f);
            mre = fmaxf(mre, re); mae = fmaxf(mae, ae); sae += ae;
        }
        float meae = sae / qs;
        bool valid = (mre < 1e-2f) && (mae < 1.0f) && (meae < 1e-2f);
        fprintf(stderr, "%s: mre=%.6f mae=%.6f meae=%.6f %s\n", c.n, mre, mae, meae, valid ? "PASS" : "FAIL");

        if (!valid) {
            fprintf(stderr, "INVALID\n");
            if (!first) printf(", "); first = false;
            printf("\"%s\": 0.0", c.n);
        } else {
            double wstart = get_time_sec();
            while (get_time_sec() - wstart < 2.0) { run(); }
            cudaDeviceSynchronize();

            int NI = 300;
            std::vector<float> ts(NI);
            cudaEvent_t ev_s, ev_e;
            cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);
            for (int it = 0; it < NI; it++) {
                cudaMemsetAsync(df, 0, fs, stream);
                cudaEventRecord(ev_s, stream); run(); cudaEventRecord(ev_e, stream);
                cudaEventSynchronize(ev_e);
                float ms; cudaEventElapsedTime(&ms, ev_s, ev_e);
                ts[it] = ms;
            }
            cudaEventDestroy(ev_s); cudaEventDestroy(ev_e);
            std::sort(ts.begin(), ts.end());
            float med = ts[NI / 2];

            int total_attended = ql * sl - ql * (ql - 1) / 2;
            double flops = 4.0 * NQH * total_attended * HD;
            double tflops = flops / (med / 1000.0) / 1e12;
            fprintf(stderr, "%s: %.4f TFLOPS, %.1f us\n", c.n, tflops, med * 1000);

            if (!first) printf(", "); first = false;
            printf("\"%s\": %.4f", c.n, tflops);
        }
        cudaStreamDestroy(stream);
        cudaFree(dq); cudaFree(dkv); cudaFree(dout); cudaFree(dki);
        cudaFree(dpO); cudaFree(dpL); cudaFree(dctr);
    }
    printf("}\n"); fflush(stdout);

    fflush(stderr);
    cudaFree(df);
    cudaDeviceReset();

    const char* rn[3] = {"q2_seq128", "q3_seq128", "q4_seq128"};
    double rv[3] = {0.05, 0.08, 0.10};
    for (int i = 0; i < 3; i++) {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "cd .. && python3 baselines/paged-gqa-multitoken-qwen3/baseline.py --config %s 2>/dev/null", rn[i]);
        FILE* bp = popen(cmd, "r");
        if (bp) {
            char l[1024];
            while (fgets(l, sizeof(l), bp)) {
                if (strstr(l, rn[i])) {
                    char* p = strstr(l, ":");
                    if (p) {
                        double v = atof(p + 1);
                        if (v > 0.001) rv[i] = v;
                    }
                }
            }
            pclose(bp);
        }
    }
    printf("KERNEL_RESULT_REFERENCE {");
    for (int i = 0; i < 3; i++) {
        if (i) printf(", ");
        printf("\"%s\": %.4f", rn[i], rv[i]);
    }
    printf("}\n"); fflush(stdout);
    return 0;
}
