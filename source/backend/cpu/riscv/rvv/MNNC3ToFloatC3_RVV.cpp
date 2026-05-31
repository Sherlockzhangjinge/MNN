#include "core/Macro.h"
#include <algorithm>
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
void MNNC3ToFloatC3_RVV(const unsigned char *source, float *dest,
                        const float *mean, const float *normal, size_t count) {
  /*
   * RVV segment 指令约束：LMUL × NFIELDS ≤ 8
   * 本函数 NFIELDS = 3（RGB三段），因此 LMUL ≤ 2。
   *
   * LMUL 选择链：
   *   源 uint8  → mf2  (加载后拓宽到 m1)
   *   uint8→u16 →  m1  (拓宽到 m2)
   *   u16→u32   →  m2  (拓宽到... 超限？)
   *
   * 正确的拓宽链（保持最终 float32 在 m2）：
   *   vsetvl_e8mf2  → vl 个 uint8，用 mf2
   *   vlseg3e8 mf2x3 → 解交错
   *   vwcvtu mf2→m1  → uint16
   *   vwcvtu m1→m2   → uint32
   *   vfcvt  m2      → float32  (m2 × 3 = 6 ≤ 8，合法)
   *   vsseg3e32 m2x3 → 交错写回
   */
  float neg_mean0 = -mean[0], neg_mean1 = -mean[1], neg_mean2 = -mean[2];
  float normal0 = normal[0], normal1 = normal[1], normal2 = normal[2];

  size_t n = count;

  while (n > 0) {
    /*
     * SEW=8, LMUL=mf2：每个向量寄存器存 VLEN/16 个 uint8
     * 这样拓宽两次后（mf2→m1→m2）float32 恰好用 m2，
     * 满足 segment store m2×3=6≤8 的约束。
     */
    size_t vl = __riscv_vsetvl_e8mf2(n);

    /* ---- 1. 交错加载 uint8，解交错为 R/G/B ---- */
    vuint8mf2x3_t seg = __riscv_vlseg3e8_v_u8mf2x3(source, vl);
    vuint8mf2_t r8 = __riscv_vget_v_u8mf2x3_u8mf2(seg, 0);
    vuint8mf2_t g8 = __riscv_vget_v_u8mf2x3_u8mf2(seg, 1);
    vuint8mf2_t b8 = __riscv_vget_v_u8mf2x3_u8mf2(seg, 2);

    /* ---- 2. uint8(mf2) → uint16(m1)，零扩展拓宽 ---- */
    vuint16m1_t r16 = __riscv_vwcvtu_x_x_v_u16m1(r8, vl);
    vuint16m1_t g16 = __riscv_vwcvtu_x_x_v_u16m1(g8, vl);
    vuint16m1_t b16 = __riscv_vwcvtu_x_x_v_u16m1(b8, vl);

    /* ---- 3. uint16(m1) → uint32(m2)，零扩展拓宽 ---- */
    vuint32m2_t r32 = __riscv_vwcvtu_x_x_v_u32m2(r16, vl);
    vuint32m2_t g32 = __riscv_vwcvtu_x_x_v_u32m2(g16, vl);
    vuint32m2_t b32 = __riscv_vwcvtu_x_x_v_u32m2(b16, vl);

    /* ---- 4. uint32(m2) → float32(m2) ---- */
    vfloat32m2_t rf = __riscv_vfcvt_f_xu_v_f32m2(r32, vl);
    vfloat32m2_t gf = __riscv_vfcvt_f_xu_v_f32m2(g32, vl);
    vfloat32m2_t bf = __riscv_vfcvt_f_xu_v_f32m2(b32, vl);

    /* ---- 5. pixel + (-mean)，向量+标量广播 ---- */
    rf = __riscv_vfadd_vf_f32m2(rf, neg_mean0, vl);
    gf = __riscv_vfadd_vf_f32m2(gf, neg_mean1, vl);
    bf = __riscv_vfadd_vf_f32m2(bf, neg_mean2, vl);

    /* ---- 6. × normal，向量×标量广播 ---- */
    rf = __riscv_vfmul_vf_f32m2(rf, normal0, vl);
    gf = __riscv_vfmul_vf_f32m2(gf, normal1, vl);
    bf = __riscv_vfmul_vf_f32m2(bf, normal2, vl);

    /* ---- 7. 交错存储 float32：m2×3=6≤8，合法 ---- */
    vfloat32m2x3_t out = __riscv_vcreate_v_f32m2x3(rf, gf, bf);
    __riscv_vsseg3e32_v_f32m2x3(dest, out, vl);

    source += vl * 3;
    dest += vl * 3;
    n -= vl;
  }
}
