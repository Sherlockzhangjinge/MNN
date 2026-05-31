#include <riscv_vector.h>
#include <stdint.h>
#include <cstdint>
void MNN2BitcopyWithStride_RVV(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    uint16_t* src = (uint16_t*)srcO;
    uint16_t* dst = (uint16_t*)dstO;

    // 转换为字节步长 (Byte Stride)
    ptrdiff_t b_stride = (ptrdiff_t)stride * sizeof(uint16_t);
    ptrdiff_t b_ds = (ptrdiff_t)ds * sizeof(uint16_t);

    for (size_t vl; size > 0; size -= vl) {
        // 设置 16 位元素宽度，使用 m8 最大寄存器组
        vl = __riscv_vsetvl_e16m8(size);
        
        // 跨步加载 16 位数据
        vuint16m8_t v_data = __riscv_vlse16_v_u16m8(src, b_stride, vl);
        
        // 跨步存储 16 位数据
        __riscv_vsse16_v_u16m8(dst, b_ds, v_data, vl);

        // 指针推移
        src += vl * stride;
        dst += vl * ds;
    }
}
