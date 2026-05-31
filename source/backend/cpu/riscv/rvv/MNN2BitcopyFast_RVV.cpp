#include <riscv_vector.h>
#include <stdint.h>
#include <cstdint>
void MNN2BitcopyFast_RVV(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    uint16_t* src = (uint16_t*)srcO;
    uint16_t* dst = (uint16_t*)dstO;

    if (ds == 1) {
        if (stride == 1) {
            // 模式 A: 连续 16位 拷贝 (memcpy)
            for (size_t vl; size > 0; size -= vl) {
                vl = __riscv_vsetvl_e16m8(size);
                vuint16m8_t v = __riscv_vle16_v_u16m8(src, vl);
                __riscv_vse16_v_u16m8(dst, v, vl);
                src += vl;
                dst += vl;
            }
        } else if (stride == 0) {
            // 模式 B: 16位 广播填充 (fill)
            uint16_t val = *src;
            for (size_t vl; size > 0; size -= vl) {
                vl = __riscv_vsetvl_e16m8(size);
                vuint16m8_t v = __riscv_vmv_v_x_u16m8(val, vl);
                __riscv_vse16_v_u16m8(dst, v, vl);
                dst += vl;
            }
        } else {
            // 模式 C: 源跨步，目标连续 (16-bit)
            ptrdiff_t b_stride = (ptrdiff_t)stride * sizeof(uint16_t);
            for (size_t vl; size > 0; size -= vl) {
                vl = __riscv_vsetvl_e16m8(size);
                vuint16m8_t v = __riscv_vlse16_v_u16m8(src, b_stride, vl);
                __riscv_vse16_v_u16m8(dst, v, vl);
                src += vl * stride;
                dst += vl;
            }
        }
    } else {
        // 模式 D: 全跨步 16位 拷贝
        ptrdiff_t b_stride = (ptrdiff_t)stride * sizeof(uint16_t);
        ptrdiff_t b_ds = (ptrdiff_t)ds * sizeof(uint16_t);
        for (size_t vl; size > 0; size -= vl) {
            vl = __riscv_vsetvl_e16m8(size);
            vuint16m8_t v = __riscv_vlse16_v_u16m8(src, b_stride, vl);
            __riscv_vsse16_v_u16m8(dst, b_ds, v, vl);
            src += vl * stride;
            dst += vl * ds;
        }
    }
}
