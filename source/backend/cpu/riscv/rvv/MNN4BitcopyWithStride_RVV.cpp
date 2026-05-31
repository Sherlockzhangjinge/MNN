#include <riscv_vector.h>
#include <stdint.h>
#include <cstdint>
void MNN4BitcopyWithStride_RVV(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    uint32_t* src = (uint32_t*)srcO;
    uint32_t* dst = (uint32_t*)dstO;

    // 转换为字节步长
    ptrdiff_t b_stride = stride * sizeof(uint32_t);
    ptrdiff_t b_ds = ds * sizeof(uint32_t);

    for (size_t vl; size > 0; size -= vl) {
        vl = __riscv_vsetvl_e32m8(size); // 使用最大寄存器组(m8)以提高吞吐
        
        // 跨步加载：从 src 开始，每隔 b_stride 字节取一个 32bit 值
        vuint32m8_t v_data = __riscv_vlse32_v_u32m8(src, b_stride, vl);
        
        // 跨步存储：存入 dst，每隔 b_ds 字节存一个 32bit 值
        __riscv_vsse32_v_u32m8(dst, b_ds, v_data, vl);

        // 更新指针：移动 vl 个元素对应的物理距离
        src += vl * stride;
        dst += vl * ds;
    }
}
