#include <riscv_vector.h>
#include <stdint.h>
#include <cstdint>

void MNN4BitcopyFast_RVV(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    uint32_t* src = (uint32_t*)srcO;
    uint32_t* dst = (uint32_t*)dstO;

    if (ds == 1) {
        if (stride == 1) {
            // 模式 A: 纯连续拷贝 (类似 memcpy)
            for (size_t vl; size > 0; size -= vl) {
                vl = __riscv_vsetvl_e32m8(size);
                vuint32m8_t v = __riscv_vle32_v_u32m8(src, vl);
                __riscv_vse32_v_u32m8(dst, v, vl);
                src += vl;
                dst += vl;
            }
        } else if (stride == 0) {
            // 模式 B: 广播填充 (类似 memset/fill)
            uint32_t val = *src;
            for (size_t vl; size > 0; size -= vl) {
                vl = __riscv_vsetvl_e32m8(size);
                vuint32m8_t v = __riscv_vmv_v_x_u32m8(val, vl);
                __riscv_vse32_v_u32m8(dst, v, vl);
                dst += vl;
            }
        } else {
            // 模式 C: 源跨步，目标连续
            ptrdiff_t b_stride = (ptrdiff_t)stride * sizeof(uint32_t);
            for (size_t vl; size > 0; size -= vl) {
                vl = __riscv_vsetvl_e32m8(size);
                vuint32m8_t v = __riscv_vlse32_v_u32m8(src, b_stride, vl);
                __riscv_vse32_v_u32m8(dst, v, vl);
                src += vl * stride;
                dst += vl;
            }
        }
    } else {
        // 模式 D: 通用跨步拷贝 (目标也不连续)
        ptrdiff_t b_stride = (ptrdiff_t)stride * sizeof(uint32_t);
        ptrdiff_t b_ds = (ptrdiff_t)ds * sizeof(uint32_t);
        for (size_t vl; size > 0; size -= vl) {
            vl = __riscv_vsetvl_e32m8(size);
            vuint32m8_t v = __riscv_vlse32_v_u32m8(src, b_stride, vl);
            __riscv_vsse32_v_u32m8(dst, b_ds, v, vl);
            src += vl * stride;
            dst += vl * ds;
        }
    }
}
