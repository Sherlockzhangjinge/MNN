#include <stddef.h>
#include <riscv_vector.h>

void MNNPackC8_RVV(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    const size_t srcAreaStride = (size_t)areaOffset[0];
    const size_t dstAreaStride = (size_t)areaOffset[1];
    const size_t depthC8 = depth / 8;
    const size_t remain = depth - depthC8 * 8;

    for (size_t z = 0; z < depthC8; ++z) {
        const float* s0 = src + (z * 8 + 0) * srcAreaStride;
        const float* s1 = src + (z * 8 + 1) * srcAreaStride;
        const float* s2 = src + (z * 8 + 2) * srcAreaStride;
        const float* s3 = src + (z * 8 + 3) * srcAreaStride;
        const float* s4 = src + (z * 8 + 4) * srcAreaStride;
        const float* s5 = src + (z * 8 + 5) * srcAreaStride;
        const float* s6 = src + (z * 8 + 6) * srcAreaStride;
        const float* s7 = src + (z * 8 + 7) * srcAreaStride;
        float* d = dst + z * dstAreaStride * 8;

        size_t x = 0;
        while (x < area) {
            size_t vl = __riscv_vsetvl_e32m1(area - x);
            vfloat32m1_t v0 = __riscv_vle32_v_f32m1(s0 + x, vl);
            vfloat32m1_t v1 = __riscv_vle32_v_f32m1(s1 + x, vl);
            vfloat32m1_t v2 = __riscv_vle32_v_f32m1(s2 + x, vl);
            vfloat32m1_t v3 = __riscv_vle32_v_f32m1(s3 + x, vl);
            vfloat32m1_t v4 = __riscv_vle32_v_f32m1(s4 + x, vl);
            vfloat32m1_t v5 = __riscv_vle32_v_f32m1(s5 + x, vl);
            vfloat32m1_t v6 = __riscv_vle32_v_f32m1(s6 + x, vl);
            vfloat32m1_t v7 = __riscv_vle32_v_f32m1(s7 + x, vl);
            vfloat32m1x8_t v = __riscv_vundefined_f32m1x8();
            v = __riscv_vset_v_f32m1_f32m1x8(v, 0, v0);
            v = __riscv_vset_v_f32m1_f32m1x8(v, 1, v1);
            v = __riscv_vset_v_f32m1_f32m1x8(v, 2, v2);
            v = __riscv_vset_v_f32m1_f32m1x8(v, 3, v3);
            v = __riscv_vset_v_f32m1_f32m1x8(v, 4, v4);
            v = __riscv_vset_v_f32m1_f32m1x8(v, 5, v5);
            v = __riscv_vset_v_f32m1_f32m1x8(v, 6, v6);
            v = __riscv_vset_v_f32m1_f32m1x8(v, 7, v7);
            __riscv_vsseg8e32_v_f32m1x8(d + x * 8, v, vl);
            x += vl;
        }
    }

    if (remain > 0) {
        float* d = dst + depthC8 * dstAreaStride * 8;
        for (size_t y = 0; y < remain; ++y) {
            const float* s = src + (depthC8 * 8 + y) * srcAreaStride;
            for (size_t x = 0; x < area; ++x) {
                d[x * 8 + y] = s[x];
            }
        }
        for (size_t y = remain; y < 8; ++y) {
            for (size_t x = 0; x < area; ++x) {
                d[x * 8 + y] = 0.0f;
            }
        }
    }
}

void MNNUnpackC8_RVV(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    const size_t srcAreaStride = (size_t)areaOffset[0];
    const size_t dstAreaStride = (size_t)areaOffset[1];
    const size_t depthC8 = depth / 8;
    const size_t remain = depth - depthC8 * 8;

    for (size_t z = 0; z < depthC8; ++z) {
        const float* s = src + z * srcAreaStride * 8;
        float* d0 = dst + (z * 8 + 0) * dstAreaStride;
        float* d1 = dst + (z * 8 + 1) * dstAreaStride;
        float* d2 = dst + (z * 8 + 2) * dstAreaStride;
        float* d3 = dst + (z * 8 + 3) * dstAreaStride;
        float* d4 = dst + (z * 8 + 4) * dstAreaStride;
        float* d5 = dst + (z * 8 + 5) * dstAreaStride;
        float* d6 = dst + (z * 8 + 6) * dstAreaStride;
        float* d7 = dst + (z * 8 + 7) * dstAreaStride;

        size_t x = 0;
        while (x < area) {
            size_t vl = __riscv_vsetvl_e32m1(area - x);
            vfloat32m1x8_t v = __riscv_vlseg8e32_v_f32m1x8(s + x * 8, vl);
            __riscv_vse32_v_f32m1(d0 + x, __riscv_vget_v_f32m1x8_f32m1(v, 0), vl);
            __riscv_vse32_v_f32m1(d1 + x, __riscv_vget_v_f32m1x8_f32m1(v, 1), vl);
            __riscv_vse32_v_f32m1(d2 + x, __riscv_vget_v_f32m1x8_f32m1(v, 2), vl);
            __riscv_vse32_v_f32m1(d3 + x, __riscv_vget_v_f32m1x8_f32m1(v, 3), vl);
            __riscv_vse32_v_f32m1(d4 + x, __riscv_vget_v_f32m1x8_f32m1(v, 4), vl);
            __riscv_vse32_v_f32m1(d5 + x, __riscv_vget_v_f32m1x8_f32m1(v, 5), vl);
            __riscv_vse32_v_f32m1(d6 + x, __riscv_vget_v_f32m1x8_f32m1(v, 6), vl);
            __riscv_vse32_v_f32m1(d7 + x, __riscv_vget_v_f32m1x8_f32m1(v, 7), vl);
            x += vl;
        }
    }

    if (remain > 0) {
        const float* s = src + depthC8 * srcAreaStride * 8;
        for (size_t y = 0; y < remain; ++y) {
            float* d = dst + (depthC8 * 8 + y) * dstAreaStride;
            for (size_t x = 0; x < area; ++x) {
                d[x] = s[x * 8 + y];
            }
        }
    }
}
