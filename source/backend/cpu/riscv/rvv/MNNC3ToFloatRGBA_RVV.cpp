#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __riscv_vector
#include <riscv_vector.h>
#endif
void MNNC3ToFloatRGBA_RVV(const unsigned char *source, float *dest,
                          const float *mean, const float *normal,
                          size_t count) {
  const float neg_mean0 = -mean[0], neg_mean1 = -mean[1], neg_mean2 = -mean[2];
  const float normal0 = normal[0], normal1 = normal[1], normal2 = normal[2];

  size_t n = count;

  while (n > 0) {
    /* SEW=8, LMUL=mf2；两次拓宽后 float32 落在 m2 */
    size_t vl = __riscv_vsetvl_e8mf2(n);

    /* ---- 1. 交错加载 RGB u8，解交错为 R/G/B 三条向量 ---- */
    vuint8mf2x3_t seg3 = __riscv_vlseg3e8_v_u8mf2x3(source, vl);
    vuint8mf2_t r8 = __riscv_vget_v_u8mf2x3_u8mf2(seg3, 0);
    vuint8mf2_t g8 = __riscv_vget_v_u8mf2x3_u8mf2(seg3, 1);
    vuint8mf2_t b8 = __riscv_vget_v_u8mf2x3_u8mf2(seg3, 2);

    /* ---- 2. uint8(mf2) → uint16(m1) ---- */
    vuint16m1_t r16 = __riscv_vwcvtu_x_x_v_u16m1(r8, vl);
    vuint16m1_t g16 = __riscv_vwcvtu_x_x_v_u16m1(g8, vl);
    vuint16m1_t b16 = __riscv_vwcvtu_x_x_v_u16m1(b8, vl);

    vuint32m2_t r32 = __riscv_vwcvtu_x_x_v_u32m2(r16, vl);
    vuint32m2_t g32 = __riscv_vwcvtu_x_x_v_u32m2(g16, vl);
    vuint32m2_t b32 = __riscv_vwcvtu_x_x_v_u32m2(b16, vl);

    /* ---- 4. uint32(m2) → float32(m2) ---- */
    vfloat32m2_t rf = __riscv_vfcvt_f_xu_v_f32m2(r32, vl);
    vfloat32m2_t gf = __riscv_vfcvt_f_xu_v_f32m2(g32, vl);
    vfloat32m2_t bf = __riscv_vfcvt_f_xu_v_f32m2(b32, vl);

    /* ---- 5. pixel + (-mean) ---- */
    rf = __riscv_vfadd_vf_f32m2(rf, neg_mean0, vl);
    gf = __riscv_vfadd_vf_f32m2(gf, neg_mean1, vl);
    bf = __riscv_vfadd_vf_f32m2(bf, neg_mean2, vl);

    /* ---- 6. × normal ---- */
    rf = __riscv_vfmul_vf_f32m2(rf, normal0, vl);
    gf = __riscv_vfmul_vf_f32m2(gf, normal1, vl);
    bf = __riscv_vfmul_vf_f32m2(bf, normal2, vl);

    /* ---- 7. Alpha 通道：广播 0.0f ---- */
    vfloat32m2_t af = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    /* ---- 8. 交错存储4通道 RGBA float（m2×4=8≤8，合法）---- */
    vfloat32m2x4_t out = __riscv_vcreate_v_f32m2x4(rf, gf, bf, af);
    __riscv_vsseg4e32_v_f32m2x4(dest, out, vl);

    /* ---- 9. 指针推进（注意步长不同）---- */
    source += vl * 3; /* 输入：每像素 3 字节 */
    dest += vl * 4;   /* 输出：每像素 4 个 float */
    n -= vl;
  }
}
