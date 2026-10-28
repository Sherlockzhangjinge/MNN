//
//  MNNGemmInt8AddBiasScale_16x4_w4_Unit_RVV.cpp
//  MNN
//
//  RVV int4-weight x int8-activation (W4A8) GEMM microkernel.
//  Mirrors the W8 kernel MNNGemmInt8AddBiasScale_16x4_Unit_RVV exactly (per-(sz,j)
//  widening MAC + full requant); the ONLY difference is the weight load: each sz
//  block is 32 bytes of packed int4 (half of the 64-byte int8 block), unpacked into
//  the same [j*16+i] layout the scalar oracle MNNGemmInt8AddBiasScale_16x4_w4_Unit
//  consumes (Int8FunctionsOpt.cpp): w8[k]=byte[k]>>4 (hi, k=0..31) / w8[k+32]=byte[k]&0xf
//  (lo). => weight for (channel-in-quad j, src-lane i):
//    j=0: hi(byte[i])     j=1: hi(byte[16+i])
//    j=2: lo(byte[i])     j=3: lo(byte[16+i])
//  Nibbles are UNSIGNED 0..15 (the int4 zero-point offset is folded into the requant
//  via weightBias / weightKernelSum, exactly as in the scalar path).
//
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <riscv_vector.h>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "../../compute/Int8FunctionsOpt.h"

void MNNGemmInt8AddBiasScale_16x4_w4_Unit_RVV(
    int8_t* dst,
    const int8_t* src,
    const int8_t* weight,
    size_t src_depth_quad,
    size_t dst_step,
    size_t dst_depth_quad,
    const QuanPostTreatParameters* post,
    size_t realCount) {

    // Dispatch marker (opt-in via MNN_RVV_W4_DEBUG) to confirm the W4A8 path reaches this kernel E2E.
    static const bool s_marker = [] {
        if (getenv("MNN_RVV_W4_DEBUG")) { fprintf(stderr, "[MNN] RVV W4A8 (int4) kernel INVOKED\n"); }
        return true;
    }();
    (void)s_marker;

    // W4A8 float output only (int4 zero-point folded into requant); matches scalar oracle.
    const int bytes = 4;

    float fp32min = 0.f, fp32max = 0.f;
    if (post->fp32minmax) {
        fp32min = post->fp32minmax[0];
        fp32max = post->fp32minmax[1];
    }

    // int4: each sz block is HALF the int8 size (32 bytes = 4*16 nibbles).
    const int weight_step_Y = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) / 2;
    const int weight_step_Z = weight_step_Y * src_depth_quad + 4 * 2 * GEMM_INT8_UNIT;

    float* biasPtr = (float*)post->biasFloat;
    auto accumbuff = post->accumBuffer;
    auto blockNum = post->blockNum;

    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        auto dst_z = dst + dz * dst_step;

        for (int bk = 0; bk < blockNum; ++bk) {

            const auto weight_dz =
                weight + dz * blockNum * weight_step_Z + bk * weight_step_Z;

            const float* scale_dz =
                reinterpret_cast<const float*>(
                    weight_dz + src_depth_quad * weight_step_Y);

            const auto weightBias_dz = scale_dz + GEMM_INT8_UNIT;
            const auto bias_dz = biasPtr + dz * GEMM_INT8_UNIT;

            const auto srcSumPtr = post->srcKernelSum + bk * realCount;

            const auto inputScalePtr =
                post->inputBias ? post->inputScale + bk * realCount
                                : post->inputScale;

            for (int w = 0; w < realCount; ++w) {

                const auto src_x =
                    src + bk * src_depth_quad * GEMM_INT8_SRC_UNIT * realCount
                    + w * GEMM_INT8_SRC_UNIT;

                auto dst_x = dst_z + w * GEMM_INT8_UNIT * bytes;
                auto accum_x = accumbuff + w * GEMM_INT8_UNIT;

                int32_t acc[4] = {0, 0, 0, 0};

                // ===============================
                // RVV core: W4A8 int4xint8 accumulate
                // ===============================
                for (int sz = 0; sz < src_depth_quad; ++sz) {

                    const uint8_t* weight_sz =
                        reinterpret_cast<const uint8_t*>(weight_dz) + weight_step_Y * sz;
                    const auto src_z =
                        src_x + sz * realCount * GEMM_INT8_SRC_UNIT;

                    size_t vl = __riscv_vsetvl_e8m1(GEMM_INT8_SRC_UNIT);

                    // src: 16 int8 lanes shared by all 4 output channels
                    vint8m1_t vsrc = __riscv_vle8_v_i8m1(src_z, vl);

                    // 32 packed bytes -> two 16-byte halves
                    vuint8m1_t b0 = __riscv_vle8_v_u8m1(weight_sz + 0, vl);       // bytes 0..15
                    vuint8m1_t b1 = __riscv_vle8_v_u8m1(weight_sz + GEMM_INT8_SRC_UNIT, vl); // bytes 16..31

                    // nibble unpack (unsigned 0..15, same as scalar w8[])
                    vint8m1_t w_j0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(b0, 4, vl));    // hi(byte[i])
                    vint8m1_t w_j1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(b1, 4, vl));    // hi(byte[16+i])
                    vint8m1_t w_j2 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(b0, 0x0f, vl)); // lo(byte[i])
                    vint8m1_t w_j3 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(b1, 0x0f, vl)); // lo(byte[16+i])

                    const vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);

                    vint16m2_t p0 = __riscv_vwmul_vv_i16m2(vsrc, w_j0, vl);
                    acc[0] += __riscv_vmv_x_s_i32m1_i32(
                        __riscv_vwredsum_vs_i16m2_i32m1(p0, vzero, vl));

                    vint16m2_t p1 = __riscv_vwmul_vv_i16m2(vsrc, w_j1, vl);
                    acc[1] += __riscv_vmv_x_s_i32m1_i32(
                        __riscv_vwredsum_vs_i16m2_i32m1(p1, vzero, vl));

                    vint16m2_t p2 = __riscv_vwmul_vv_i16m2(vsrc, w_j2, vl);
                    acc[2] += __riscv_vmv_x_s_i32m1_i32(
                        __riscv_vwredsum_vs_i16m2_i32m1(p2, vzero, vl));

                    vint16m2_t p3 = __riscv_vwmul_vv_i16m2(vsrc, w_j3, vl);
                    acc[3] += __riscv_vmv_x_s_i32m1_i32(
                        __riscv_vwredsum_vs_i16m2_i32m1(p3, vzero, vl));
                }

                // ===============================
                // Requant (verbatim from the W8 RVV kernel == scalar W4 path)
                // ===============================
                for (int j = 0; j < GEMM_INT8_UNIT; ++j) {

                    float value = acc[j] * scale_dz[j]
                                  + srcSumPtr[w] * weightBias_dz[j];

                    if (post->inputScale) {
                        value = acc[j] * scale_dz[j] * inputScalePtr[w]
                                + srcSumPtr[w] * weightBias_dz[j];
                    }

                    if (post->inputBias) {
                        auto weightKernelSum =
                            post->weightKernelSum
                            + dz * (blockNum * GEMM_INT8_UNIT)
                            + bk * GEMM_INT8_UNIT;

                        value += (post->inputBias[bk * realCount + w]
                                  * weightKernelSum[j]);
                    }

                    if (bk > 0) {
                        value += ((float*)accum_x)[j];
                    }

                    if (bk == blockNum - 1) {
                        if (biasPtr) {
                            value += bias_dz[j];
                        }
                        if (post->fp32minmax) {
                            value = std::min(std::max(fp32min, value), fp32max);
                        }
                        ((float*)dst_x)[j] = value;
                    } else {
                        ((float*)accum_x)[j] = value;
                    }
                }
            }
        }
    }
}
