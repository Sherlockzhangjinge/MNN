#include <riscv_vector.h>
#include <stdint.h>
#include <cstdint>
void MNN1BitcopyWithStride_RVV(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    // 步长在 uint8_t 下直接使用，无需乘以 sizeof
    ptrdiff_t b_stride = (ptrdiff_t)stride;
    ptrdiff_t b_ds = (ptrdiff_t)ds;

    for (size_t vl; size > 0; size -= vl) {
        // 使用 8 位元素宽度 (e8)，最大寄存器组 (m8)
        vl = __riscv_vsetvl_e8m8(size);
        
        // 跨步加载 8 位数据
        vuint8m8_t v_data = __riscv_vlse8_v_u8m8(srcO, b_stride, vl);
        
        // 跨步存储 8 位数据
        __riscv_vsse8_v_u8m8(dstO, b_ds, v_data, vl);

        // 推进指针
        srcO += vl * stride;
        dstO += vl * ds;
    }
}
