//
//  MNNGemmInt8AccumOverH_RVV.cpp
//  MNN
//
//  M3' Plan B: accumulate-over-H int8 GEMM for RVV (W8A8, dynamic-quant, block-quantized weight), TILED.
//
//  RVV has no native int8 dot-product. The stock reduction kernel uses vwmul+vwredsum (latency-bound on
//  the vwredsum chain). This kernel removes the reduction by broadcasting each activation element a[l] as a
//  scalar (vwmul.vx) and accumulating with vwadd.wv over the output channels.
//
//  TILED layout: weight is repacked to [h_tile][bk][ble][TW] (TW = e8m1 vl = VLEN/8). The kernel processes one
//  TW-wide oc-tile at a time with the token loop INSIDE the tile, so the tile's weight is loaded cold once and
//  reused from cache across the (<=DST_XUNIT) tokens, and every weight read is CONTIGUOUS. This is essential:
//  the earlier [bk][d][H] (h-innermost) layout read weight strided and re-read it per token, which was ~3.6x
//  SLOWER end-to-end despite looking fast in a cache-resident microbench. Verified bit-exact vs the stock
//  reduction kernel on K1(X60)/K3(X100), including OC-split (decode) partitions.
//
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_USE_RVV
#include <riscv_vector.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <sys/time.h>
#include "../../compute/Int8FunctionsOpt.h"

namespace {
constexpr int kUnit = 4;   // GEMM_INT8_UNIT
constexpr int kLp   = 16;  // GEMM_INT8_SRC_UNIT
constexpr int kHdr  = 16;  // header bytes: [int32 Hfull, blockNum, blockL, TW]
// Read a uint64 field (e.g. "mvendorid"/"marchid") from /proc/cpuinfo. 0 if absent.
uint64_t readCpuId(const char* key) {
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (!f) return 0;
    char line[512];
    uint64_t v = 0;
    while (fgets(line, sizeof line, f)) {
        if (strstr(line, key)) {
            char* c = strchr(line, ':');
            if (c) v = strtoull(c + 1, nullptr, 0);
            break;
        }
    }
    fclose(f);
    return v;
}

// Grain (LMUL) planner: which tile width the accum-over-H kernel uses on THIS microarchitecture.
// Register-constrained optimum (cost-model §4.2 / analysis/rvv_grain_planner.py): spend the accumulator-
// register budget on LMUL up to the core's saturation knee, the rest on UNIT (token batching). Cores whose
// accum throughput keeps scaling past m4 (C920 +20%, X100 +9% end-to-end) take the wide tile (m8, TW=VLEN/4,
// 2-token); the narrow in-order X60 saturates by m4 so it keeps the 4-token batch (m4, TW=VLEN/8). X60/X100
// share mvendorid 0x710 and are told apart ONLY by marchid (X60 0x..58000001, X100 0x..58000002) -> the
// (vendor,marchid) joint key is mandatory. MNN_RVV_ACCUM_GRAIN={4,8} overrides for A/B experiments.
inline bool accumWantM8() {
    if (const char* g = getenv("MNN_RVV_ACCUM_GRAIN")) { if (g[0] == '8') return true; if (g[0] == '4') return false; }
    const uint64_t vend = readCpuId("mvendorid"), march = readCpuId("marchid");
    if (vend == 0x710 && (march & 0xffu) == 0x01u) return false; // SpacemiT X60/K1: LMUL saturates at m4 -> m4
    return true;                                                 // C920, X100, and unknown cores -> m8
}
// 3-way grain LMUL for the §五#1 full grid: {2,4,8}. Default = the deployed 2-way (accumWantM8?8:4);
// MNN_RVV_ACCUM_GRAIN=2 adds m2/U4 (a non-frontier point, measured dominated by m4/U4 -- pruning-safety).
inline int accumGrainLMUL() {
    if (const char* g = getenv("MNN_RVV_ACCUM_GRAIN")) { if (g[0]=='2') return 2; if (g[0]=='4') return 4; if (g[0]=='8') return 8; }
    return accumWantM8() ? 8 : 4;
}
thread_local int gAccumPlanOverride = -1;

inline int accumTW() {
    if (gAccumPlanOverride > 0) {
        const int lmul = (gAccumPlanOverride % 1000) / 10;
        return lmul == 8 ? (int)__riscv_vsetvlmax_e8m2()
             : lmul == 2 ? (int)__riscv_vsetvlmax_e8mf2()
                         : (int)__riscv_vsetvlmax_e8m1();
    }
    static const int tw = [](){ int L = accumGrainLMUL();
        return L == 8 ? (int)__riscv_vsetvlmax_e8m2()
             : L == 2 ? (int)__riscv_vsetvlmax_e8mf2()
                      : (int)__riscv_vsetvlmax_e8m1(); }();
    return tw;
}
// UNIT cap for the grid: MNN_RVV_ACCUM_UNIT=1 forces m8/U1 (non-frontier, dominated by m8/U2). 0 = default (max).
inline int accumUnitCap() { static int u = [](){ const char* e=getenv("MNN_RVV_ACCUM_UNIT"); return e?atoi(e):0; }(); return u; }
// Implementation variant: MNN_RVV_ACCUM_IMPL=strided forces the h-innermost [bk][d][H] layout (weight re-read
// per token, ~3.6x slower e2e) -- kept as a calibration candidate to MEASURE the tiled>strided pruning. Default tiled.
inline bool accumImplStrided() { static int s=-1; if(s<0){const char* e=getenv("MNN_RVV_ACCUM_IMPL"); s=(e&&(e[0]=='s'||e[0]=='S'))?1:0;} return s==1; }
} // namespace

// Byte size of the tiled accum weight buffer (executor uses this to allocate). Must agree with the repack.
extern "C" size_t MNNReorderWeightAccumH_Size_RVV(int oc, int ocUp4, int blockNum, int blockL) {
    (void)oc;
    const int TW     = accumTW();
    const int ble    = blockL * kLp;
    const int Htiles = (ocUp4 + TW - 1) / TW;
    const int Hpad   = Htiles * TW;
    size_t wtCount   = (size_t)Htiles * blockNum * ble * TW;
    return (size_t)kHdr + wtCount + (size_t)3 * blockNum * Hpad * sizeof(float);
}

// Repack stock packed weight -> tiled [h_tile][bk][ble][TW] + scaleC/wbiasC/wksC [bk][Hpad].
//   packedWeight : [hU][blockNum]{ [sz][u][16] int8 , scale[4] , wbias[4] }  (packWeightAndQuantInfo output)
//   weightKernelSum : [hU][blockNum][UNIT] float, or null (accum kernel only reads wksC on the asym-input path).
extern "C" void MNNReorderWeightAccumH_RVV(int8_t* dst, const int8_t* packedWeight, const float* weightKernelSum,
                                           int oc, int ocUp4, int blockNum, int blockL) {
    const int TW     = accumTW();
    const int ble    = blockL * kLp;
    const int hU     = ocUp4 / kUnit;
    const int s1     = blockL * kUnit * kLp;              // int8 weight bytes per (hU,bk)
    const int bb     = s1 + 2 * kUnit * (int)sizeof(float);
    const int Htiles = (ocUp4 + TW - 1) / TW;
    const int Hpad   = Htiles * TW;
    size_t wtCount   = (size_t)Htiles * blockNum * ble * TW;

    int32_t* hd = reinterpret_cast<int32_t*>(dst);
    hd[0] = ocUp4; hd[1] = blockNum; hd[2] = blockL; hd[3] = TW;
    int8_t* WT   = dst + kHdr;
    float* scaleC = reinterpret_cast<float*>(WT + wtCount);
    float* wbiasC = scaleC + (size_t)blockNum * Hpad;
    float* wksC   = wbiasC + (size_t)blockNum * Hpad;
    memset(dst + kHdr, 0, wtCount); // pad channels -> 0 weight

    for (int bk = 0; bk < blockNum; ++bk) {
        for (int i = 0; i < hU; ++i) {
            const int8_t* blk = packedWeight + ((size_t)i * blockNum + bk) * bb;
            const float*  sc  = reinterpret_cast<const float*>(blk + s1);
            const float*  wb  = sc + kUnit;
            for (int u = 0; u < kUnit; ++u) {
                int h = i * kUnit + u, ht = h / TW, t = h % TW;
                scaleC[(size_t)bk * Hpad + h] = sc[u];
                wbiasC[(size_t)bk * Hpad + h] = wb[u];
                wksC[(size_t)bk * Hpad + h] =
                    weightKernelSum ? weightKernelSum[(size_t)i * blockNum * kUnit + bk * kUnit + u] : 0.f;
                for (int sz = 0; sz < blockL; ++sz) {
                    for (int l = 0; l < kLp; ++l) {
                        int d = sz * kLp + l;
                        WT[(((size_t)ht * blockNum + bk) * ble + d) * TW + t] = blk[sz * kUnit * kLp + u * kLp + l];
                    }
                }
            }
        }
    }
}

extern "C" size_t MNNReorderWeightAccumH_Size_Plan_RVV(int oc, int ocUp4, int blockNum, int blockL, int planId) {
    const int previous = gAccumPlanOverride;
    gAccumPlanOverride = planId;
    const size_t result = MNNReorderWeightAccumH_Size_RVV(oc, ocUp4, blockNum, blockL);
    gAccumPlanOverride = previous;
    return result;
}

extern "C" void MNNReorderWeightAccumH_Plan_RVV(int8_t* dst, const int8_t* packedWeight,
                                                  const float* weightKernelSum, int oc, int ocUp4, int blockNum,
                                                  int blockL, int planId) {
    const int previous = gAccumPlanOverride;
    gAccumPlanOverride = planId;
    MNNReorderWeightAccumH_RVV(dst, packedWeight, weightKernelSum, oc, ocUp4, blockNum, blockL);
    gAccumPlanOverride = previous;
}
// Drop-in for CoreInt8Functions::Int8GemmKernel (requires post->useInt8 == 0; executor gates dstBytes == 4).
//   weight : MNNReorderWeightAccumH_RVV buffer (full, un-offset). dst_depth_quad = oc-groups THIS call produces.
//   post->accumHOcStart : absolute oc offset of this call's slice (0 token-split, ocIndex OC-split).
extern "C" void MNNGemmInt8AddBiasScale_16x4_AccumH_RVV(int8_t* dst, const int8_t* src, const int8_t* weight,
                                                        size_t src_depth_quad, size_t dst_step,
                                                        size_t dst_depth_quad, const QuanPostTreatParameters* post,
                                                        size_t realCount) {
    const int32_t* hd = reinterpret_cast<const int32_t*>(weight);
    const int Hfull = hd[0], blockNum = hd[1], blockL = hd[2], TW = hd[3];
    const int ble = blockL * kLp;
    const int Htiles = (Hfull + TW - 1) / TW, Hpad = Htiles * TW;
    const int8_t* WT = weight + kHdr;
    size_t wtCount = (size_t)Htiles * blockNum * ble * TW;
    const float* scaleC = reinterpret_cast<const float*>(WT + wtCount);
    const float* wbiasC = scaleC + (size_t)blockNum * Hpad;
    const float* wksC   = wbiasC + (size_t)blockNum * Hpad;

    const int ocStart = post->accumHOcStart;
    const int Hthread = (int)dst_depth_quad * kUnit;
    const int rc = (int)realCount;
    const float* biasPtr = post->biasFloat;              // relative to ocStart
    const float* sks = post->srcKernelSum;               // [bk][rc]
    const float* iS  = post->inputScale;                 // sym:[rc] asym:[bk][rc] (null -> 1)
    const float* iB  = post->inputBias;                  // asym:[bk][rc] else null
    float mn = 0.f, mx = 0.f;
    const bool clamp = (post->fp32minmax != nullptr);
    if (clamp) { mn = post->fp32minmax[0]; mx = post->fp32minmax[1]; }

    float chunk[256];
    if (TW == (int)__riscv_vsetvlmax_e8mf2()) { // ===== m2/U4 grain (non-frontier, §五#1 pruning-safety; TW=VLEN/16, i32m2 4-token) =====
    for (int tokenBase = 0; tokenBase < rc; tokenBase += 4) {
    int relOc = 0;
    while (relOc < Hthread) {
        int oc = ocStart + relOc, ht = oc / TW, off = oc % TW;
        int cw = TW - off; if (cw > Hthread - relOc) cw = Hthread - relOc;
        size_t vl = (size_t)cw;
        vfloat32m2_t r0 = __riscv_vfmv_v_f_f32m2(0.f, vl), r1 = r0, r2 = r0, r3 = r0;
        for (int bk = 0; bk < blockNum; ++bk) {
            const int8_t* wt = WT + (((size_t)ht * blockNum + bk) * ble) * TW + off;
            const int8_t* sb = src + (size_t)bk * blockL * rc * kLp;
            vint32m2_t a0 = __riscv_vmv_v_x_i32m2(0, vl), a1 = a0, a2 = a0, a3 = a0;
            for (int d = 0; d < ble; ++d) {
                int sz = d / kLp, l = d % kLp;
                vint8mf2_t vw = __riscv_vle8_v_i8mf2(wt + (size_t)d * TW, vl); // half-LMUL weight (VLEN/16 lanes)
                const int8_t* ar = sb + (size_t)sz * rc * kLp + l + (size_t)tokenBase * kLp;
                a0 = __riscv_vwadd_wv_i32m2(a0, __riscv_vwmul_vx_i16m1(vw, ar[0], vl), vl);
                if (rc - tokenBase > 1) a1 = __riscv_vwadd_wv_i32m2(a1, __riscv_vwmul_vx_i16m1(vw, ar[kLp], vl), vl);
                if (rc - tokenBase > 2) a2 = __riscv_vwadd_wv_i32m2(a2, __riscv_vwmul_vx_i16m1(vw, ar[2 * kLp], vl), vl);
                if (rc - tokenBase > 3) a3 = __riscv_vwadd_wv_i32m2(a3, __riscv_vwmul_vx_i16m1(vw, ar[3 * kLp], vl), vl);
            }
            vfloat32m2_t vsc0 = __riscv_vle32_v_f32m2(scaleC + (size_t)bk * Hpad + oc, vl);
            vfloat32m2_t vwb = __riscv_vle32_v_f32m2(wbiasC + (size_t)bk * Hpad + oc, vl);
            vfloat32m2_t vwks;
            if (iB) vwks = __riscv_vle32_v_f32m2(wksC + (size_t)bk * Hpad + oc, vl);
            auto req = [&](vfloat32m2_t r, vint32m2_t a, int e) -> vfloat32m2_t {
                vfloat32m2_t accf = __riscv_vfcvt_f_x_v_f32m2(a, vl);
                vfloat32m2_t vsc = iS ? __riscv_vfmul_vf_f32m2(vsc0, (iB ? iS[bk * rc + e] : iS[e]), vl) : vsc0;
                vfloat32m2_t val = __riscv_vfmul_vf_f32m2(vwb, sks[bk * rc + e], vl);
                if (iB) val = __riscv_vfmacc_vf_f32m2(val, iB[bk * rc + e], vwks, vl);
                val = __riscv_vfmacc_vv_f32m2(val, accf, vsc, vl);
                return __riscv_vfadd_vv_f32m2(r, val, vl);
            };
            r0 = req(r0, a0, tokenBase);
            if (rc - tokenBase > 1) r1 = req(r1, a1, tokenBase + 1);
            if (rc - tokenBase > 2) r2 = req(r2, a2, tokenBase + 2);
            if (rc - tokenBase > 3) r3 = req(r3, a3, tokenBase + 3);
        }
        auto fin = [&](vfloat32m2_t r, int e) {
            if (biasPtr) r = __riscv_vfadd_vv_f32m2(r, __riscv_vle32_v_f32m2(biasPtr + relOc, vl), vl);
            if (clamp) { r = __riscv_vfmax_vf_f32m2(r, mn, vl); r = __riscv_vfmin_vf_f32m2(r, mx, vl); }
            __riscv_vse32_v_f32m2(chunk, r, vl);
            for (int i = 0; i < cw; i += kUnit) {
                int dz = (relOc + i) / kUnit;
                float* d = reinterpret_cast<float*>(dst + (size_t)dz * dst_step) + (size_t)e * kUnit;
                d[0] = chunk[i]; d[1] = chunk[i + 1]; d[2] = chunk[i + 2]; d[3] = chunk[i + 3];
            }
        };
        fin(r0, tokenBase);
        if (rc - tokenBase > 1) fin(r1, tokenBase + 1);
        if (rc - tokenBase > 2) fin(r2, tokenBase + 2);
        if (rc - tokenBase > 3) fin(r3, tokenBase + 3);
        relOc += cw;
    }
    }
    } else if (TW <= (int)__riscv_vsetvlmax_e8m1()) { // ===== m4 grain (default), token-inner (4-token) =====
    for (int tokenBase = 0; tokenBase < rc; tokenBase += 4) {
    int relOc = 0;
    while (relOc < Hthread) {
        int oc = ocStart + relOc, ht = oc / TW, off = oc % TW;
        int cw = TW - off; if (cw > Hthread - relOc) cw = Hthread - relOc;
        size_t vl = (size_t)cw;
        // token-INNER: load each weight vector once per depth, apply to all rc(<=DST_XUNIT) tokens with named
        // accumulators (RVV vector types cannot be array elements), cutting hot-loop weight loads ~rc x.
        vfloat32m4_t r0 = __riscv_vfmv_v_f_f32m4(0.f, vl), r1 = r0, r2 = r0, r3 = r0;
        for (int bk = 0; bk < blockNum; ++bk) {
            const int8_t* wt = WT + (((size_t)ht * blockNum + bk) * ble) * TW + off;
            const int8_t* sb = src + (size_t)bk * blockL * rc * kLp;
            vint32m4_t a0 = __riscv_vmv_v_x_i32m4(0, vl), a1 = a0, a2 = a0, a3 = a0;
            for (int d = 0; d < ble; ++d) {
                int sz = d / kLp, l = d % kLp;
                vint8m1_t vw = __riscv_vle8_v_i8m1(wt + (size_t)d * TW, vl); // one load, reused across tokens
                const int8_t* ar = sb + (size_t)sz * rc * kLp + l + (size_t)tokenBase * kLp;
                a0 = __riscv_vwadd_wv_i32m4(a0, __riscv_vwmul_vx_i16m2(vw, ar[0], vl), vl);
                if (rc - tokenBase > 1) a1 = __riscv_vwadd_wv_i32m4(a1, __riscv_vwmul_vx_i16m2(vw, ar[kLp], vl), vl);
                if (rc - tokenBase > 2) a2 = __riscv_vwadd_wv_i32m4(a2, __riscv_vwmul_vx_i16m2(vw, ar[2 * kLp], vl), vl);
                if (rc - tokenBase > 3) a3 = __riscv_vwadd_wv_i32m4(a3, __riscv_vwmul_vx_i16m2(vw, ar[3 * kLp], vl), vl);
            }
            vfloat32m4_t vsc0 = __riscv_vle32_v_f32m4(scaleC + (size_t)bk * Hpad + oc, vl);
            vfloat32m4_t vwb = __riscv_vle32_v_f32m4(wbiasC + (size_t)bk * Hpad + oc, vl);
            vfloat32m4_t vwks;
            if (iB) vwks = __riscv_vle32_v_f32m4(wksC + (size_t)bk * Hpad + oc, vl);
            auto req = [&](vfloat32m4_t r, vint32m4_t a, int e) -> vfloat32m4_t {
                vfloat32m4_t accf = __riscv_vfcvt_f_x_v_f32m4(a, vl);
                vfloat32m4_t vsc = iS ? __riscv_vfmul_vf_f32m4(vsc0, (iB ? iS[bk * rc + e] : iS[e]), vl) : vsc0;
                vfloat32m4_t val = __riscv_vfmul_vf_f32m4(vwb, sks[bk * rc + e], vl);
                if (iB) val = __riscv_vfmacc_vf_f32m4(val, iB[bk * rc + e], vwks, vl);
                val = __riscv_vfmacc_vv_f32m4(val, accf, vsc, vl);
                return __riscv_vfadd_vv_f32m4(r, val, vl);
            };
            r0 = req(r0, a0, tokenBase);
            if (rc - tokenBase > 1) r1 = req(r1, a1, tokenBase + 1);
            if (rc - tokenBase > 2) r2 = req(r2, a2, tokenBase + 2);
            if (rc - tokenBase > 3) r3 = req(r3, a3, tokenBase + 3);
        }
        auto fin = [&](vfloat32m4_t r, int e) {
            if (biasPtr) r = __riscv_vfadd_vv_f32m4(r, __riscv_vle32_v_f32m4(biasPtr + relOc, vl), vl);
            if (clamp) { r = __riscv_vfmax_vf_f32m4(r, mn, vl); r = __riscv_vfmin_vf_f32m4(r, mx, vl); }
            __riscv_vse32_v_f32m4(chunk, r, vl);
            for (int i = 0; i < cw; i += kUnit) {
                int dz = (relOc + i) / kUnit;
                float* d = reinterpret_cast<float*>(dst + (size_t)dz * dst_step) + (size_t)e * kUnit;
                d[0] = chunk[i]; d[1] = chunk[i + 1]; d[2] = chunk[i + 2]; d[3] = chunk[i + 3];
            }
        };
        fin(r0, tokenBase);
        if (rc - tokenBase > 1) fin(r1, tokenBase + 1);
        if (rc - tokenBase > 2) fin(r2, tokenBase + 2);
        if (rc - tokenBase > 3) fin(r3, tokenBase + 3);
        relOc += cw;
    }
    }
    } else { // ===== m8 grain (MNN_RVV_ACCUM_GRAIN=8): 2-token batch, running kept in MEMORY =====
        // 2x i32m8 accumulators (16 regs). Running in run0/run1 (memory) frees the 16 regs a token-inner would
        // need for r0..r3, and the requant is split (t=accf*vsc; val+=t) so peak live m8 stays ~24 <= 32.
        float run0[256], run1[256];
        int relOc = 0;
        while (relOc < Hthread) {
            int oc = ocStart + relOc, ht = oc / TW, off = oc % TW;
            int cwv = TW - off; if (cwv > Hthread - relOc) cwv = Hthread - relOc;
            size_t vl = __riscv_vsetvl_e8m2((size_t)cwv); int cw = (int)vl;
            int ub = (accumUnitCap() == 1) ? 1 : 2;  // m8/U1 (§五#1 non-frontier) when MNN_RVV_ACCUM_UNIT=1; else U2
            for (int ep = 0; ep < rc; ep += ub) {
                int n = (rc - ep >= ub) ? ub : (rc - ep);
                for (int i = 0; i < cw; ++i) { run0[i] = 0.f; if (n > 1) run1[i] = 0.f; }
                for (int bk = 0; bk < blockNum; ++bk) {
                    const int8_t* wt = WT + (((size_t)ht * blockNum + bk) * ble) * TW + off;
                    const int8_t* sb = src + (size_t)bk * blockL * rc * kLp;
                    vint32m8_t a0 = __riscv_vmv_v_x_i32m8(0, vl), a1 = a0;
                    for (int d = 0; d < ble; ++d) {
                        int sz = d / kLp, l = d % kLp;
                        const int8_t* ar = sb + (size_t)sz * rc * kLp + l;
                        vint8m2_t vw = __riscv_vle8_v_i8m2(wt + (size_t)d * TW, vl);
                        a0 = __riscv_vwadd_wv_i32m8(a0, __riscv_vwmul_vx_i16m4(vw, ar[(size_t)ep * kLp], vl), vl);
                        if (n > 1) a1 = __riscv_vwadd_wv_i32m8(a1, __riscv_vwmul_vx_i16m4(vw, ar[(size_t)(ep + 1) * kLp], vl), vl);
                    }
                    for (int pass = 0; pass < n; ++pass) {
                        int e = ep + pass; float* run = pass ? run1 : run0; vint32m8_t a = pass ? a1 : a0;
                        vfloat32m8_t val = __riscv_vfmul_vf_f32m8(__riscv_vle32_v_f32m8(wbiasC + (size_t)bk * Hpad + oc, vl), sks[bk * rc + e], vl);
                        if (iB) val = __riscv_vfmacc_vf_f32m8(val, iB[bk * rc + e], __riscv_vle32_v_f32m8(wksC + (size_t)bk * Hpad + oc, vl), vl);
                        vfloat32m8_t vsc = __riscv_vle32_v_f32m8(scaleC + (size_t)bk * Hpad + oc, vl);
                        if (iS) vsc = __riscv_vfmul_vf_f32m8(vsc, (iB ? iS[bk * rc + e] : iS[e]), vl);
                        vfloat32m8_t t = __riscv_vfmul_vv_f32m8(__riscv_vfcvt_f_x_v_f32m8(a, vl), vsc, vl);
                        val = __riscv_vfadd_vv_f32m8(val, t, vl);
                        __riscv_vse32_v_f32m8(run, __riscv_vfadd_vv_f32m8(__riscv_vle32_v_f32m8(run, vl), val, vl), vl);
                    }
                }
                for (int pass = 0; pass < n; ++pass) {
                    int e = ep + pass; float* run = pass ? run1 : run0;
                    vfloat32m8_t vr = __riscv_vle32_v_f32m8(run, vl);
                    if (biasPtr) vr = __riscv_vfadd_vv_f32m8(vr, __riscv_vle32_v_f32m8(biasPtr + relOc, vl), vl);
                    if (clamp) { vr = __riscv_vfmax_vf_f32m8(vr, mn, vl); vr = __riscv_vfmin_vf_f32m8(vr, mx, vl); }
                    __riscv_vse32_v_f32m8(chunk, vr, vl);
                    for (int i = 0; i < cw; i += kUnit) {
                        int dz = (relOc + i) / kUnit;
                        float* d = reinterpret_cast<float*>(dst + (size_t)dz * dst_step) + (size_t)e * kUnit;
                        d[0] = chunk[i]; d[1] = chunk[i + 1]; d[2] = chunk[i + 2]; d[3] = chunk[i + 3];
                    }
                }
            }
            relOc += cw;
        }
    }
}

// ================= adaptive planner: accum-over-H vs stock reduction for this microarchitecture =================
namespace {
double nowSec() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}
double probeReduction(const int8_t* A, const int8_t* W, int32_t* C, int E, int H, int L) {
    int rep = 2; double t = 0, w2 = 2.0 * E * H * L;
    while (t < 0.02) { rep *= 2; double t0 = nowSec();
        for (int r = 0; r < rep; ++r)
            for (int e = 0; e < E; ++e)
                for (int hh = 0; hh < H; ++hh) {
                    const int8_t* a = A + (size_t)e * L; const int8_t* w = W + (size_t)hh * L; int32_t acc = 0; int l = 0;
                    while (l < L) { size_t vl = __riscv_vsetvl_e8m1((size_t)(L - l));
                        vint16m2_t p = __riscv_vwmul_vv_i16m2(__riscv_vle8_v_i8m1(a + l, vl), __riscv_vle8_v_i8m1(w + l, vl), vl);
                        acc += __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(p, __riscv_vmv_v_x_i32m1(0, 1), vl));
                        l += (int)vl; }
                    C[(size_t)e * H + hh] = acc; }
        t = nowSec() - t0; }
    return w2 * rep / t / 1e9;
}
double probeAccum(const int8_t* A, const int8_t* WT, int32_t* C, int E, int H, int L) {
    int rep = 2; double t = 0, w2 = 2.0 * E * H * L;
    while (t < 0.02) { rep *= 2; double t0 = nowSec();
        for (int r = 0; r < rep; ++r)
            for (int e = 0; e < E; ++e) {
                const int8_t* a = A + (size_t)e * L; int hh = 0;
                while (hh < H) { size_t vl = __riscv_vsetvl_e8m1((size_t)(H - hh));
                    vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);
                    for (int l = 0; l < L; ++l)
                        acc = __riscv_vwadd_wv_i32m4(acc, __riscv_vwmul_vx_i16m2(__riscv_vle8_v_i8m1(WT + (size_t)l * H + hh, vl), a[l], vl), vl);
                    __riscv_vse32_v_i32m4(C + (size_t)e * H + hh, acc, vl);
                    hh += (int)vl; }
            }
        t = nowSec() - t0; }
    return w2 * rep / t / 1e9;
}
bool computeUseAccum() {
    if (const char* f = getenv("MNN_RVV_FORCE_REDUCTION")) { if (f[0] == '1') return false; }
    if (const char* f = getenv("MNN_RVV_FORCE_ACCUM")) { if (f[0] == '1') return true; }
    uint64_t vend = readCpuId("mvendorid");
    if (vend == 0x5b7 || vend == 0x710) return true; // T-Head SG2044; SpacemiT X60/X100
    const int E = 8, H = 64, L = 512;
    static int8_t A[E * L], W[H * L], WT[L * H];
    static int32_t C[E * H];
    for (int i = 0; i < E * L; ++i) A[i] = (int8_t)((i * 7) & 0x7f);
    for (int i = 0; i < H * L; ++i) W[i] = (int8_t)((i * 3) & 0x7f);
    for (int hh = 0; hh < H; ++hh) for (int l = 0; l < L; ++l) WT[(size_t)l * H + hh] = W[(size_t)hh * L + l];
    double rred = probeReduction(A, W, C, E, H, L);
    double racc = probeAccum(A, WT, C, E, H, L);
    return racc > rred * 1.1;
}
} // namespace

extern "C" bool MNNRVVUseAccumOverH() {
    static bool cached = computeUseAccum();
    return cached;
}

// W4-specific accum gate. The int4 deinterleave accum (m4 grain) beats the reduction W4 kernel END-TO-END on
// SpacemiT (X60 1.88x prefill / X100 1.56x) but LOSES on C920/SG2044 (0x5b7): its superscalar 64-core reduction
// is already fast (pp ~37) and the m4-only W4 accum (C920's W8 accum uses the wider m8 tile) can't beat it.
// So gate W4 accum to SpacemiT 0x710 -- a case where the kernel microbench (biggest accum win 2.90x on C920)
// mispredicts the end-to-end winner. MNN_RVV_W4_ACCUM=0/1 overrides for A/B.
namespace {
bool computeUseAccumW4() {
    if (const char* e = getenv("MNN_RVV_W4_ACCUM")) { if (e[0] == '0') return false; if (e[0] == '1') return true; }
    if (!MNNRVVUseAccumOverH()) return false;
    return readCpuId("mvendorid") == 0x710;   // SpacemiT X60/X100 win; T-Head C920 keeps the fast reduction W4
}
}
extern "C" bool MNNRVVUseAccumOverH_W4() {
    static bool cached = computeUseAccumW4();
    return cached;
}

// ============================================================================
// M3' E-f: W4A8 accum-over-H (int4 weight). Mirrors the W8 AccumH path (token-inner, full
// requant, m4 AND m8 grain via accumTW()); the ONLY difference is the weight is int4-packed
// at HALF the bytes and deinterleaved to full TW lanes on the fly.
//
// Tiled int4 weight (split-half): byte b of tile ht at depth d holds
//   hi nibble = channel (ht*TW + b) ,  lo nibble = channel (ht*TW + b + TW/2).
// Kernel: load TW/2 bytes -> vsrl>>4 = hi (ch[0..TW/2)) / vand&0xf = lo (ch[TW/2..TW)),
// vslideup lo into the upper half -> a full TW-lane weight vector (for m8, vlmul_ext the
// e8m1 halves to e8m2 first), then the SAME vwmul.vx + vwadd.wv accumulation as W8. Nibbles
// are UNSIGNED 0..15 (int4 zero-point folded into requant), matching the scalar/reduction W4.
// Grain follows accumWantM8() like W8: X60 m4, X100/C920 m8. End-to-end prefill (SpacemiT, gated):
// X60 1.84x (m4), X100 1.74x (m8, +10% over m4); C920 gate-off (its fast reduction beats W4 accum at
// any grain -- m4 0.71x / m8 0.45x prefill; only C920 decode would win, but the per-op kernel choice keeps
// reduction to avoid the big prefill regression). See MNNRVVUseAccumOverH_W4 + docs/RVV_W4A8_results.md §9.
// ============================================================================

extern "C" size_t MNNReorderWeightAccumH_W4_Size_RVV(int oc, int ocUp4, int blockNum, int blockL) {
    (void)oc;
    const int TW  = accumTW();                          // grain-adaptive (m4 e8m1 / m8 e8m2)
    const int TWh = TW / 2;
    const int ble = blockL * kLp;
    const int Htiles = (ocUp4 + TW - 1) / TW;
    const int Hpad   = Htiles * TW;
    size_t wtCount   = (size_t)Htiles * blockNum * ble * TWh;   // int4-packed => HALF of W8
    return (size_t)kHdr + wtCount + (size_t)3 * blockNum * Hpad * sizeof(float);
}

// Stock W4 packed weight -> tiled int4 [h_tile][bk][ble][TW/2] (split-half) + scaleC/wbiasC/wksC [bk][Hpad].
//   packedWeight : [hU][blockNum]{ [blockL][32 bytes int4] , scale[4] , wbias[4] }  (MNNReorderWeightInt4 output)
//   nibble(u,l)  : byte = wsz[((u&1)?16:0)+l] ; value = (u<2)? byte>>4 : byte&0xf   (matches scalar w4 oracle)
extern "C" void MNNReorderWeightAccumH_W4_RVV(int8_t* dst, const int8_t* packedWeight, const float* weightKernelSum,
                                              int oc, int ocUp4, int blockNum, int blockL) {
    const int TW   = accumTW();                         // grain-adaptive (m4 / m8)
    const int TWh  = TW / 2;
    const int ble  = blockL * kLp;
    const int hU   = ocUp4 / kUnit;
    const int s1_4 = blockL * 32;                       // int4 bytes per (hU,bk): blockL * (4*16/2)
    const int bb4  = s1_4 + 2 * kUnit * (int)sizeof(float);
    const int Htiles = (ocUp4 + TW - 1) / TW;
    const int Hpad   = Htiles * TW;
    size_t wtCount   = (size_t)Htiles * blockNum * ble * TWh;

    int32_t* hd = reinterpret_cast<int32_t*>(dst);
    hd[0] = ocUp4; hd[1] = blockNum; hd[2] = blockL; hd[3] = TW;
    uint8_t* WT   = reinterpret_cast<uint8_t*>(dst + kHdr);
    float* scaleC = reinterpret_cast<float*>(WT + wtCount);
    float* wbiasC = scaleC + (size_t)blockNum * Hpad;
    float* wksC   = wbiasC + (size_t)blockNum * Hpad;
    memset(dst + kHdr, 0, wtCount);   // pad channels -> 0 weight; also int4 OR-accumulation base

    for (int bk = 0; bk < blockNum; ++bk) {
        for (int i = 0; i < hU; ++i) {
            const uint8_t* blk = reinterpret_cast<const uint8_t*>(packedWeight) + ((size_t)i * blockNum + bk) * bb4;
            const float* sc = reinterpret_cast<const float*>(blk + s1_4);
            const float* wb = sc + kUnit;
            for (int u = 0; u < kUnit; ++u) {
                int h = i * kUnit + u, ht = h / TW, t = h % TW;
                scaleC[(size_t)bk * Hpad + h] = sc[u];
                wbiasC[(size_t)bk * Hpad + h] = wb[u];
                wksC[(size_t)bk * Hpad + h] =
                    weightKernelSum ? weightKernelSum[(size_t)i * blockNum * kUnit + bk * kUnit + u] : 0.f;
                const int b = (t < TWh) ? t : (t - TWh);
                const bool hiHalf = (t < TWh);
                for (int sz = 0; sz < blockL; ++sz) {
                    const uint8_t* wsz = blk + (size_t)sz * 32;
                    for (int l = 0; l < kLp; ++l) {
                        int d = sz * kLp + l;
                        uint8_t byte = wsz[((u & 1) ? 16 : 0) + l];
                        uint8_t nib  = (u < 2) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0f);   // 0..15
                        size_t idx = (((size_t)ht * blockNum + bk) * ble + d) * TWh + b;
                        WT[idx] |= hiHalf ? (uint8_t)(nib << 4) : (uint8_t)(nib & 0x0f);
                    }
                }
            }
        }
    }
}

extern "C" size_t MNNReorderWeightAccumH_W4_Size_Plan_RVV(int oc, int ocUp4, int blockNum, int blockL, int planId) {
    const int previous = gAccumPlanOverride;
    gAccumPlanOverride = planId;
    const size_t result = MNNReorderWeightAccumH_W4_Size_RVV(oc, ocUp4, blockNum, blockL);
    gAccumPlanOverride = previous;
    return result;
}

extern "C" void MNNReorderWeightAccumH_W4_Plan_RVV(int8_t* dst, const int8_t* packedWeight,
                                                     const float* weightKernelSum, int oc, int ocUp4, int blockNum,
                                                     int blockL, int planId) {
    const int previous = gAccumPlanOverride;
    gAccumPlanOverride = planId;
    MNNReorderWeightAccumH_W4_RVV(dst, packedWeight, weightKernelSum, oc, ocUp4, blockNum, blockL);
    gAccumPlanOverride = previous;
}
extern "C" void MNNGemmInt8AddBiasScale_16x4_AccumH_W4_RVV(int8_t* dst, const int8_t* src, const int8_t* weight,
                                                           size_t src_depth_quad, size_t dst_step,
                                                           size_t dst_depth_quad, const QuanPostTreatParameters* post,
                                                           size_t realCount) {
    const int32_t* hd = reinterpret_cast<const int32_t*>(weight);
    const int Hfull = hd[0], blockNum = hd[1], blockL = hd[2], TW = hd[3];
    const int TWh = TW / 2;
    const int ble = blockL * kLp;
    const int Htiles = (Hfull + TW - 1) / TW, Hpad = Htiles * TW;
    const uint8_t* WT = reinterpret_cast<const uint8_t*>(weight) + kHdr;
    size_t wtCount = (size_t)Htiles * blockNum * ble * TWh;
    const float* scaleC = reinterpret_cast<const float*>(WT + wtCount);
    const float* wbiasC = scaleC + (size_t)blockNum * Hpad;
    const float* wksC   = wbiasC + (size_t)blockNum * Hpad;

    const int ocStart = post->accumHOcStart;
    const int Hthread = (int)dst_depth_quad * kUnit;
    const int rc = (int)realCount;
    const float* biasPtr = post->biasFloat;
    const float* sks = post->srcKernelSum;
    const float* iS  = post->inputScale;
    const float* iB  = post->inputBias;
    float mn = 0.f, mx = 0.f;
    const bool clamp = (post->fp32minmax != nullptr);
    if (clamp) { mn = post->fp32minmax[0]; mx = post->fp32minmax[1]; }

    float chunk[256];
    if (TW == (int)__riscv_vsetvlmax_e8mf2()) { // ===== m2/U4 grain (non-frontier, §五#1; int4 deinterleave to e8mf2) =====
    for (int tokenBase = 0; tokenBase < rc; tokenBase += 4) {
    int relOc = 0;
    while (relOc < Hthread) {
        int oc = ocStart + relOc, ht = oc / TW, off = oc % TW;
        int cw = TW - off; if (cw > Hthread - relOc) cw = Hthread - relOc;
        size_t vl = (size_t)cw;
        size_t vlh = (size_t)TWh;
        vfloat32m2_t r0 = __riscv_vfmv_v_f_f32m2(0.f, vl), r1 = r0, r2 = r0, r3 = r0;
        for (int bk = 0; bk < blockNum; ++bk) {
            const uint8_t* wt = WT + (((size_t)ht * blockNum + bk) * ble) * TWh;
            const int8_t* sb = src + (size_t)bk * blockL * rc * kLp;
            vint32m2_t a0 = __riscv_vmv_v_x_i32m2(0, vl), a1 = a0, a2 = a0, a3 = a0;
            for (int d = 0; d < ble; ++d) {
                int sz = d / kLp, l = d % kLp;
                vuint8mf2_t bb = __riscv_vle8_v_u8mf2(wt + (size_t)d * TWh, vlh);
                vuint8mf2_t hi = __riscv_vsrl_vx_u8mf2(bb, 4, vlh);
                vuint8mf2_t lo = __riscv_vand_vx_u8mf2(bb, 0x0f, vlh);
                vuint8mf2_t full = __riscv_vslideup_vx_u8mf2(hi, lo, (size_t)TWh, (size_t)TW);
                vint8mf2_t vw = (off == 0)
                    ? __riscv_vreinterpret_v_u8mf2_i8mf2(full)
                    : __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vslidedown_vx_u8mf2(full, (size_t)off, vl));
                const int8_t* ar = sb + (size_t)sz * rc * kLp + l + (size_t)tokenBase * kLp;
                a0 = __riscv_vwadd_wv_i32m2(a0, __riscv_vwmul_vx_i16m1(vw, ar[0], vl), vl);
                if (rc - tokenBase > 1) a1 = __riscv_vwadd_wv_i32m2(a1, __riscv_vwmul_vx_i16m1(vw, ar[kLp], vl), vl);
                if (rc - tokenBase > 2) a2 = __riscv_vwadd_wv_i32m2(a2, __riscv_vwmul_vx_i16m1(vw, ar[2 * kLp], vl), vl);
                if (rc - tokenBase > 3) a3 = __riscv_vwadd_wv_i32m2(a3, __riscv_vwmul_vx_i16m1(vw, ar[3 * kLp], vl), vl);
            }
            vfloat32m2_t vsc0 = __riscv_vle32_v_f32m2(scaleC + (size_t)bk * Hpad + oc, vl);
            vfloat32m2_t vwb = __riscv_vle32_v_f32m2(wbiasC + (size_t)bk * Hpad + oc, vl);
            vfloat32m2_t vwks;
            if (iB) vwks = __riscv_vle32_v_f32m2(wksC + (size_t)bk * Hpad + oc, vl);
            auto req = [&](vfloat32m2_t r, vint32m2_t a, int e) -> vfloat32m2_t {
                vfloat32m2_t accf = __riscv_vfcvt_f_x_v_f32m2(a, vl);
                vfloat32m2_t vsc = iS ? __riscv_vfmul_vf_f32m2(vsc0, (iB ? iS[bk * rc + e] : iS[e]), vl) : vsc0;
                vfloat32m2_t val = __riscv_vfmul_vf_f32m2(vwb, sks[bk * rc + e], vl);
                if (iB) val = __riscv_vfmacc_vf_f32m2(val, iB[bk * rc + e], vwks, vl);
                val = __riscv_vfmacc_vv_f32m2(val, accf, vsc, vl);
                return __riscv_vfadd_vv_f32m2(r, val, vl);
            };
            r0 = req(r0, a0, tokenBase);
            if (rc - tokenBase > 1) r1 = req(r1, a1, tokenBase + 1);
            if (rc - tokenBase > 2) r2 = req(r2, a2, tokenBase + 2);
            if (rc - tokenBase > 3) r3 = req(r3, a3, tokenBase + 3);
        }
        auto fin = [&](vfloat32m2_t r, int e) {
            if (biasPtr) r = __riscv_vfadd_vv_f32m2(r, __riscv_vle32_v_f32m2(biasPtr + relOc, vl), vl);
            if (clamp) { r = __riscv_vfmax_vf_f32m2(r, mn, vl); r = __riscv_vfmin_vf_f32m2(r, mx, vl); }
            __riscv_vse32_v_f32m2(chunk, r, vl);
            for (int i = 0; i < cw; i += kUnit) {
                int dz = (relOc + i) / kUnit;
                float* d = reinterpret_cast<float*>(dst + (size_t)dz * dst_step) + (size_t)e * kUnit;
                d[0] = chunk[i]; d[1] = chunk[i + 1]; d[2] = chunk[i + 2]; d[3] = chunk[i + 3];
            }
        };
        fin(r0, tokenBase);
        if (rc - tokenBase > 1) fin(r1, tokenBase + 1);
        if (rc - tokenBase > 2) fin(r2, tokenBase + 2);
        if (rc - tokenBase > 3) fin(r3, tokenBase + 3);
        relOc += cw;
    }
    }
    } else if (TW <= (int)__riscv_vsetvlmax_e8m1()) { // ===== m4 grain (token-inner, 4-token) =====
    for (int tokenBase = 0; tokenBase < rc; tokenBase += 4) {
    int relOc = 0;
    while (relOc < Hthread) {
        int oc = ocStart + relOc, ht = oc / TW, off = oc % TW;
        int cw = TW - off; if (cw > Hthread - relOc) cw = Hthread - relOc;
        size_t vl = (size_t)cw;
        size_t vlh = (size_t)TWh;
        vfloat32m4_t r0 = __riscv_vfmv_v_f_f32m4(0.f, vl), r1 = r0, r2 = r0, r3 = r0;
        for (int bk = 0; bk < blockNum; ++bk) {
            const uint8_t* wt = WT + (((size_t)ht * blockNum + bk) * ble) * TWh;
            const int8_t* sb = src + (size_t)bk * blockL * rc * kLp;
            vint32m4_t a0 = __riscv_vmv_v_x_i32m4(0, vl), a1 = a0, a2 = a0, a3 = a0;
            for (int d = 0; d < ble; ++d) {
                int sz = d / kLp, l = d % kLp;
                vuint8m1_t bb = __riscv_vle8_v_u8m1(wt + (size_t)d * TWh, vlh);  // HALF the bytes; full tile
                vuint8m1_t hi = __riscv_vsrl_vx_u8m1(bb, 4, vlh);               // ch[0..TWh)
                vuint8m1_t lo = __riscv_vand_vx_u8m1(bb, 0x0f, vlh);            // ch[TWh..TW)
                vuint8m1_t full = __riscv_vslideup_vx_u8m1(hi, lo, (size_t)TWh, (size_t)TW);
                vint8m1_t vw = (off == 0)
                    ? __riscv_vreinterpret_v_u8m1_i8m1(full)
                    : __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vslidedown_vx_u8m1(full, (size_t)off, vl));
                const int8_t* ar = sb + (size_t)sz * rc * kLp + l + (size_t)tokenBase * kLp;
                a0 = __riscv_vwadd_wv_i32m4(a0, __riscv_vwmul_vx_i16m2(vw, ar[0], vl), vl);
                if (rc - tokenBase > 1) a1 = __riscv_vwadd_wv_i32m4(a1, __riscv_vwmul_vx_i16m2(vw, ar[kLp], vl), vl);
                if (rc - tokenBase > 2) a2 = __riscv_vwadd_wv_i32m4(a2, __riscv_vwmul_vx_i16m2(vw, ar[2 * kLp], vl), vl);
                if (rc - tokenBase > 3) a3 = __riscv_vwadd_wv_i32m4(a3, __riscv_vwmul_vx_i16m2(vw, ar[3 * kLp], vl), vl);
            }
            vfloat32m4_t vsc0 = __riscv_vle32_v_f32m4(scaleC + (size_t)bk * Hpad + oc, vl);
            vfloat32m4_t vwb = __riscv_vle32_v_f32m4(wbiasC + (size_t)bk * Hpad + oc, vl);
            vfloat32m4_t vwks;
            if (iB) vwks = __riscv_vle32_v_f32m4(wksC + (size_t)bk * Hpad + oc, vl);
            auto req = [&](vfloat32m4_t r, vint32m4_t a, int e) -> vfloat32m4_t {
                vfloat32m4_t accf = __riscv_vfcvt_f_x_v_f32m4(a, vl);
                vfloat32m4_t vsc = iS ? __riscv_vfmul_vf_f32m4(vsc0, (iB ? iS[bk * rc + e] : iS[e]), vl) : vsc0;
                vfloat32m4_t val = __riscv_vfmul_vf_f32m4(vwb, sks[bk * rc + e], vl);
                if (iB) val = __riscv_vfmacc_vf_f32m4(val, iB[bk * rc + e], vwks, vl);
                val = __riscv_vfmacc_vv_f32m4(val, accf, vsc, vl);
                return __riscv_vfadd_vv_f32m4(r, val, vl);
            };
            r0 = req(r0, a0, tokenBase);
            if (rc - tokenBase > 1) r1 = req(r1, a1, tokenBase + 1);
            if (rc - tokenBase > 2) r2 = req(r2, a2, tokenBase + 2);
            if (rc - tokenBase > 3) r3 = req(r3, a3, tokenBase + 3);
        }
        auto fin = [&](vfloat32m4_t r, int e) {
            if (biasPtr) r = __riscv_vfadd_vv_f32m4(r, __riscv_vle32_v_f32m4(biasPtr + relOc, vl), vl);
            if (clamp) { r = __riscv_vfmax_vf_f32m4(r, mn, vl); r = __riscv_vfmin_vf_f32m4(r, mx, vl); }
            __riscv_vse32_v_f32m4(chunk, r, vl);
            for (int i = 0; i < cw; i += kUnit) {
                int dz = (relOc + i) / kUnit;
                float* d = reinterpret_cast<float*>(dst + (size_t)dz * dst_step) + (size_t)e * kUnit;
                d[0] = chunk[i]; d[1] = chunk[i + 1]; d[2] = chunk[i + 2]; d[3] = chunk[i + 3];
            }
        };
        fin(r0, tokenBase);
        if (rc - tokenBase > 1) fin(r1, tokenBase + 1);
        if (rc - tokenBase > 2) fin(r2, tokenBase + 2);
        if (rc - tokenBase > 3) fin(r3, tokenBase + 3);
        relOc += cw;
    }
    }
    } else { // ===== m8 grain: 2-token batch, running kept in MEMORY (int4 deinterleave to e8m2) =====
        float run0[256], run1[256];
        int relOc = 0;
        while (relOc < Hthread) {
            int oc = ocStart + relOc, ht = oc / TW, off = oc % TW;
            int cwv = TW - off; if (cwv > Hthread - relOc) cwv = Hthread - relOc;
            size_t vl = __riscv_vsetvl_e8m2((size_t)cwv); int cw = (int)vl;
            for (int ep = 0; ep < rc; ep += 2) {
                int n = (rc - ep >= 2) ? 2 : 1;
                for (int i = 0; i < cw; ++i) { run0[i] = 0.f; if (n > 1) run1[i] = 0.f; }
                for (int bk = 0; bk < blockNum; ++bk) {
                    const uint8_t* wt = WT + (((size_t)ht * blockNum + bk) * ble) * TWh;
                    const int8_t* sb = src + (size_t)bk * blockL * rc * kLp;
                    vint32m8_t a0 = __riscv_vmv_v_x_i32m8(0, vl), a1 = a0;
                    for (int d = 0; d < ble; ++d) {
                        int sz = d / kLp, l = d % kLp;
                        const int8_t* ar = sb + (size_t)sz * rc * kLp + l;
                        vuint8m1_t bb = __riscv_vle8_v_u8m1(wt + (size_t)d * TWh, (size_t)TWh);
                        vuint8m1_t hi = __riscv_vsrl_vx_u8m1(bb, 4, (size_t)TWh);
                        vuint8m1_t lo = __riscv_vand_vx_u8m1(bb, 0x0f, (size_t)TWh);
                        vuint8m2_t fu = __riscv_vslideup_vx_u8m2(__riscv_vlmul_ext_v_u8m1_u8m2(hi),
                                                                 __riscv_vlmul_ext_v_u8m1_u8m2(lo), (size_t)TWh, (size_t)TW);
                        vint8m2_t vwF = __riscv_vreinterpret_v_u8m2_i8m2(fu);
                        vint8m2_t vw = (off == 0) ? vwF : __riscv_vslidedown_vx_i8m2(vwF, (size_t)off, vl);
                        a0 = __riscv_vwadd_wv_i32m8(a0, __riscv_vwmul_vx_i16m4(vw, ar[(size_t)ep * kLp], vl), vl);
                        if (n > 1) a1 = __riscv_vwadd_wv_i32m8(a1, __riscv_vwmul_vx_i16m4(vw, ar[(size_t)(ep + 1) * kLp], vl), vl);
                    }
                    for (int pass = 0; pass < n; ++pass) {
                        int e = ep + pass; float* run = pass ? run1 : run0; vint32m8_t a = pass ? a1 : a0;
                        vfloat32m8_t val = __riscv_vfmul_vf_f32m8(__riscv_vle32_v_f32m8(wbiasC + (size_t)bk * Hpad + oc, vl), sks[bk * rc + e], vl);
                        if (iB) val = __riscv_vfmacc_vf_f32m8(val, iB[bk * rc + e], __riscv_vle32_v_f32m8(wksC + (size_t)bk * Hpad + oc, vl), vl);
                        vfloat32m8_t vsc = __riscv_vle32_v_f32m8(scaleC + (size_t)bk * Hpad + oc, vl);
                        if (iS) vsc = __riscv_vfmul_vf_f32m8(vsc, (iB ? iS[bk * rc + e] : iS[e]), vl);
                        vfloat32m8_t t = __riscv_vfmul_vv_f32m8(__riscv_vfcvt_f_x_v_f32m8(a, vl), vsc, vl);
                        val = __riscv_vfadd_vv_f32m8(val, t, vl);
                        __riscv_vse32_v_f32m8(run, __riscv_vfadd_vv_f32m8(__riscv_vle32_v_f32m8(run, vl), val, vl), vl);
                    }
                }
                for (int pass = 0; pass < n; ++pass) {
                    int e = ep + pass; float* run = pass ? run1 : run0;
                    vfloat32m8_t vr = __riscv_vle32_v_f32m8(run, vl);
                    if (biasPtr) vr = __riscv_vfadd_vv_f32m8(vr, __riscv_vle32_v_f32m8(biasPtr + relOc, vl), vl);
                    if (clamp) { vr = __riscv_vfmax_vf_f32m8(vr, mn, vl); vr = __riscv_vfmin_vf_f32m8(vr, mx, vl); }
                    __riscv_vse32_v_f32m8(chunk, vr, vl);
                    for (int i = 0; i < cw; i += kUnit) {
                        int dz = (relOc + i) / kUnit;
                        float* d = reinterpret_cast<float*>(dst + (size_t)dz * dst_step) + (size_t)e * kUnit;
                        d[0] = chunk[i]; d[1] = chunk[i + 1]; d[2] = chunk[i + 2]; d[3] = chunk[i + 3];
                    }
                }
            }
            relOc += cw;
        }
    }
}

// ============================================================================
// M3' §五#1: accum-over-H STRIDED implementation variant (h-innermost [bk][d][H], token-OUTER).
// The pre-tiled layout the production tiled kernel replaced: weight read H-strided and RE-READ per token (no
// tile reuse) -> ~3.6x slower e2e despite looking fast cache-resident. Kept as a calibration candidate
// (MNN_RVV_ACCUM_IMPL=strided) to MEASURE the tiled>strided performance-pruning. Requant + NC4HW4 scatter are
// identical to the tiled kernel; only the weight layout (h-innermost) and token loop (outer) differ.
// Buffer: [i32 hdr: ocUp4, blockNum, blockL, 0(=strided)][WT[bk][d][ocUp4] int8][scaleC][wbiasC][wksC] (bk*ocUp4).
// ============================================================================
extern "C" size_t MNNReorderWeightAccumH_Strided_Size_RVV(int oc, int ocUp4, int blockNum, int blockL) {
    (void)oc;
    const int ble = blockL * kLp;
    size_t wtCount = (size_t)blockNum * ble * ocUp4;
    return (size_t)kHdr + wtCount + (size_t)3 * blockNum * ocUp4 * sizeof(float);
}
extern "C" void MNNReorderWeightAccumH_Strided_RVV(int8_t* dst, const int8_t* packedWeight, const float* weightKernelSum,
                                                   int oc, int ocUp4, int blockNum, int blockL) {
    (void)oc;
    const int ble = blockL * kLp;
    const int hU  = ocUp4 / kUnit;
    const int s1  = blockL * kUnit * kLp;
    const int bb  = s1 + 2 * kUnit * (int)sizeof(float);
    size_t wtCount = (size_t)blockNum * ble * ocUp4;
    int32_t* hd = reinterpret_cast<int32_t*>(dst);
    hd[0] = ocUp4; hd[1] = blockNum; hd[2] = blockL; hd[3] = 0; // TW=0 marks strided
    int8_t* WT = dst + kHdr;
    float* scaleC = reinterpret_cast<float*>(WT + wtCount);
    float* wbiasC = scaleC + (size_t)blockNum * ocUp4;
    float* wksC   = wbiasC + (size_t)blockNum * ocUp4;
    memset(dst + kHdr, 0, wtCount);
    for (int bk = 0; bk < blockNum; ++bk) {
        for (int i = 0; i < hU; ++i) {
            const int8_t* blk = packedWeight + ((size_t)i * blockNum + bk) * bb;
            const float*  sc  = reinterpret_cast<const float*>(blk + s1);
            const float*  wb  = sc + kUnit;
            for (int u = 0; u < kUnit; ++u) {
                int h = i * kUnit + u;
                scaleC[(size_t)bk * ocUp4 + h] = sc[u];
                wbiasC[(size_t)bk * ocUp4 + h] = wb[u];
                wksC[(size_t)bk * ocUp4 + h] =
                    weightKernelSum ? weightKernelSum[(size_t)i * blockNum * kUnit + bk * kUnit + u] : 0.f;
                for (int sz = 0; sz < blockL; ++sz)
                    for (int l = 0; l < kLp; ++l) {
                        int d = sz * kLp + l;
                        WT[((size_t)bk * ble + d) * ocUp4 + h] = blk[sz * kUnit * kLp + u * kLp + l];
                    }
            }
        }
    }
}
extern "C" size_t MNNReorderWeightAccumH_Strided_Size_Plan_RVV(int oc, int ocUp4, int blockNum, int blockL, int planId) {
    (void)planId;
    return MNNReorderWeightAccumH_Strided_Size_RVV(oc, ocUp4, blockNum, blockL);
}

extern "C" void MNNReorderWeightAccumH_Strided_Plan_RVV(int8_t* dst, const int8_t* packedWeight,
                                                          const float* weightKernelSum, int oc, int ocUp4,
                                                          int blockNum, int blockL, int planId) {
    (void)planId;
    MNNReorderWeightAccumH_Strided_RVV(dst, packedWeight, weightKernelSum, oc, ocUp4, blockNum, blockL);
}
extern "C" void MNNGemmInt8AddBiasScale_16x4_AccumH_Strided_RVV(int8_t* dst, const int8_t* src, const int8_t* weight,
        size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post,
        size_t realCount) {
    (void)src_depth_quad;
    const int32_t* hd = reinterpret_cast<const int32_t*>(weight);
    const int Hpad = hd[0], blockNum = hd[1], blockL = hd[2]; // Hpad = ocUp4 (h-contiguous, no tiling)
    const int ble = blockL * kLp;
    const int8_t* WT = weight + kHdr;
    size_t wtCount = (size_t)blockNum * ble * Hpad;
    const float* scaleC = reinterpret_cast<const float*>(WT + wtCount);
    const float* wbiasC = scaleC + (size_t)blockNum * Hpad;
    const float* wksC   = wbiasC + (size_t)blockNum * Hpad;
    const int ocStart = post->accumHOcStart;
    const int Hthread = (int)dst_depth_quad * kUnit;
    const int rc = (int)realCount;
    const float* biasPtr = post->biasFloat;
    const float* sks = post->srcKernelSum;
    const float* iS  = post->inputScale;
    const float* iB  = post->inputBias;
    float mn = 0.f, mx = 0.f; const bool clamp = (post->fp32minmax != nullptr);
    if (clamp) { mn = post->fp32minmax[0]; mx = post->fp32minmax[1]; }
    float chunk[256];
    for (int e = 0; e < rc; ++e) {          // token-OUTER: weight re-read per token (the "slow" pattern)
        int relOc = 0;
        while (relOc < Hthread) {
            int oc = ocStart + relOc;
            size_t vl = __riscv_vsetvl_e8m1((size_t)(Hthread - relOc));
            int cw = (int)vl;
            vfloat32m4_t running = __riscv_vfmv_v_f_f32m4(0.f, vl);
            for (int bk = 0; bk < blockNum; ++bk) {
                const int8_t* wblk = WT + (size_t)bk * ble * Hpad;   // [d][h]
                const int8_t* sb = src + (size_t)bk * blockL * rc * kLp;
                vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);
                for (int d = 0; d < ble; ++d) {
                    int sz = d / kLp, l = d % kLp;
                    vint8m1_t vw = __riscv_vle8_v_i8m1(wblk + (size_t)d * Hpad + oc, vl); // STRIDED: stride Hpad per d
                    const int8_t* ar = sb + (size_t)sz * rc * kLp + (size_t)e * kLp + l;
                    acc = __riscv_vwadd_wv_i32m4(acc, __riscv_vwmul_vx_i16m2(vw, ar[0], vl), vl);
                }
                vfloat32m4_t accf = __riscv_vfcvt_f_x_v_f32m4(acc, vl);
                vfloat32m4_t vsc = __riscv_vle32_v_f32m4(scaleC + (size_t)bk * Hpad + oc, vl);
                if (iS) vsc = __riscv_vfmul_vf_f32m4(vsc, (iB ? iS[bk * rc + e] : iS[e]), vl);
                vfloat32m4_t vwb = __riscv_vle32_v_f32m4(wbiasC + (size_t)bk * Hpad + oc, vl);
                vfloat32m4_t val = __riscv_vfmul_vf_f32m4(vwb, sks[bk * rc + e], vl);
                if (iB) { vfloat32m4_t vwks = __riscv_vle32_v_f32m4(wksC + (size_t)bk * Hpad + oc, vl);
                          val = __riscv_vfmacc_vf_f32m4(val, iB[bk * rc + e], vwks, vl); }
                val = __riscv_vfmacc_vv_f32m4(val, accf, vsc, vl);
                running = __riscv_vfadd_vv_f32m4(running, val, vl);
            }
            if (biasPtr) running = __riscv_vfadd_vv_f32m4(running, __riscv_vle32_v_f32m4(biasPtr + relOc, vl), vl);
            if (clamp) { running = __riscv_vfmax_vf_f32m4(running, mn, vl); running = __riscv_vfmin_vf_f32m4(running, mx, vl); }
            __riscv_vse32_v_f32m4(chunk, running, vl);
            for (int i = 0; i < cw; i += kUnit) {
                int dz = (relOc + i) / kUnit;
                float* d = reinterpret_cast<float*>(dst + (size_t)dz * dst_step) + (size_t)e * kUnit;
                d[0] = chunk[i]; d[1] = chunk[i + 1]; d[2] = chunk[i + 2]; d[3] = chunk[i + 3];
            }
            relOc += cw;
        }
    }
}
extern "C" bool MNNRVVAccumImplStrided() { return accumImplStrided(); }

#endif // MNN_USE_RVV
