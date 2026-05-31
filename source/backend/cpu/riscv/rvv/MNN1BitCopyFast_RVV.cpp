#include <riscv_vector.h>
#include <stdint.h>

void MNN1BitCopyFast_RVV(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    if (size <= 0) return;

    if (ds == 1) {
        if (stride == 1) {
            // 模式 A: 连续字节拷贝 (memcpy 优化)
            for (size_t vl; size > 0; size -= vl) {
                vl = __riscv_vsetvl_e8m8(size);
                vuint8m8_t v = __riscv_vle8_v_u8m8(srcO, vl);
                __riscv_vse8_v_u8m8(dstO, v, vl);
                srcO += vl;
                dstO += vl;
            }
        } else if (stride == 0) {
            // 模式 B: 字节广播填充 (memset 优化)
            uint8_t val = *srcO;
            for (size_t vl; size > 0; size -= vl) {
                vl = __riscv_vsetvl_e8m8(size);
                vuint8m8_t v = __riscv_vmv_v_x_u8m8(val, vl);
                __riscv_vse8_v_u8m8(dstO, v, vl);
                dstO += vl;
            }
        } else {
            // 模式 C: 源跨步，目标连续 (多用于通道剥离)
            ptrdiff_t b_stride = (ptrdiff_t)stride;
            for (size_t vl; size > 0; size -= vl) {
                vl = __riscv_vsetvl_e8m8(size);
                vuint8m8_t v = __riscv_vlse8_v_u8m8(srcO, b_stride, vl);
                __riscv_vse8_v_u8m8(dstO, v, vl);
                srcO += vl * stride;
                dstO += vl;
            }
        }
    } else {
        // 模式 D: 全跨步拷贝
        ptrdiff_t b_stride = (ptrdiff_t)stride;
        ptrdiff_t b_ds = (ptrdiff_t)ds;
        for (size_t vl; size > 0; size -= vl) {
            vl = __riscv_vsetvl_e8m8(size);
            vuint8m8_t v = __riscv_vlse8_v_u8m8(srcO, b_stride, vl);
            __riscv_vsse8_v_u8m8(dstO, b_ds, v, vl);
            srcO += vl * stride;
            dstO += vl * ds;
        }
    }
}
