// Native arm64 builds only (the browser uses the wasm build): lin_p16 and lin_i8 with more rows per block than the wasm
// kernels, for AArch64's 32 vector registers. x is read 4 inputs at a time and each one is used by lane (FMUL by element is
// the same single multiply as a splat): RW rows x 16 outputs per block. Each output still sums mul then add (no FMA) over
// i = 0..I-1 in order, so y is bit-identical to the wasm kernels. RW: rows per block (4 was fastest on Apple M3 Max: 4 rows x 16 outputs = 16 accumulators + 4 x + 4 weights).
#ifndef RW
#define RW 4
#endif

/** One step i of an RW x 16 block: a[r][q] += x[r][i] * w_q (lane l of xv[r] holds x[r][i]). */
#define STEP(l)                                                                                                        \
  _Pragma("clang loop unroll(full)") for (int r = 0; r < RW; r++) {                                                    \
    a[r][0] = vaddq_f32(a[r][0], vmulq_laneq_f32(w0, xv[r], l)); a[r][1] = vaddq_f32(a[r][1], vmulq_laneq_f32(w1, xv[r], l)); \
    a[r][2] = vaddq_f32(a[r][2], vmulq_laneq_f32(w2, xv[r], l)); a[r][3] = vaddq_f32(a[r][3], vmulq_laneq_f32(w3, xv[r], l)); \
  }

/** The RW x 16 block over i = 0..I-1: W16(i) sets w0..w3 to the 16 weights of input i. */
#define BLOCK16(W16)                                                                                                   \
  float32x4_t a[RW][4];                                                                                                \
  _Pragma("clang loop unroll(full)") for (int r = 0; r < RW; r++) a[r][0] = a[r][1] = a[r][2] = a[r][3] = vdupq_n_f32(0); \
  int i = 0;                                                                                                           \
  for (; i + 4 <= I; i += 4) {                                                                                         \
    float32x4_t xv[RW], w0, w1, w2, w3;                                                                                \
    _Pragma("clang loop unroll(full)") for (int r = 0; r < RW; r++) xv[r] = vld1q_f32(xr[r] + i);                      \
    W16(i) STEP(0) W16(i + 1) STEP(1) W16(i + 2) STEP(2) W16(i + 3) STEP(3)                                            \
  }                                                                                                                    \
  for (; i < I; i++) {                                                                                                 \
    float32x4_t xv[RW], w0, w1, w2, w3;                                                                                \
    _Pragma("clang loop unroll(full)") for (int r = 0; r < RW; r++) xv[r] = vdupq_n_f32(xr[r][i]);                     \
    W16(i) STEP(0)                                                                                                     \
  }

static void lin_p16n(const f32 *x, int T, int I, const f32 *w, int O4, const f32 *scale, const f32 *bias, f32 *y, int act, int res) {
  const int O = O4 * 4, O16 = O & ~15;
  for (int t0 = 0; t0 < T; t0 += RW) {
    const int rows = T - t0 < RW ? T - t0 : RW;
    const f32 *xr[RW];
    for (int r = 0; r < RW; r++) xr[r] = x + (t0 + (r < rows ? r : 0)) * I;
    for (int o = 0; o < O16; o += 16) {
      const f32 *wp = w + (o / 16) * I * 16;
#define W16P(k) w0 = vld1q_f32(wp + (k) * 16); w1 = vld1q_f32(wp + (k) * 16 + 4); w2 = vld1q_f32(wp + (k) * 16 + 8); w3 = vld1q_f32(wp + (k) * 16 + 12);
      BLOCK16(W16P)
#undef W16P
      for (int r = 0; r < rows; r++) for (int q = 0; q < 4; q++) finish(y + (t0 + r) * O + o + 4 * q, a[r][q], scale + o + 4 * q, bias + o + 4 * q, act, res);
    }
    for (int o = O16; o < O; o += 4) {
      const f32 *wp0 = w + O16 * I + (o - O16) * I;
      for (int r = 0; r < rows; r++) {
        float32x4_t acc = vdupq_n_f32(0);
        const f32 *wp = wp0;
        for (int i = 0; i < I; i++, wp += 4) acc = vaddq_f32(acc, vmulq_f32(vdupq_n_f32(xr[r][i]), vld1q_f32(wp)));
        finish(y + (t0 + r) * O + o, acc, scale + o, bias + o, act, res);
      }
    }
  }
}

/** 16 int8 weights -> 4 float vectors (exact per lane, as w4_i8). */
static inline void i8x16_f32(const int8_t *p, float32x4_t *w0, float32x4_t *w1, float32x4_t *w2, float32x4_t *w3) {
  int8x16_t b = vld1q_s8(p);
  int16x8_t lo = vmovl_s8(vget_low_s8(b)), hi = vmovl_high_s8(b);
  *w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))); *w1 = vcvtq_f32_s32(vmovl_high_s16(lo));
  *w2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))); *w3 = vcvtq_f32_s32(vmovl_high_s16(hi));
}

/** lin_i8 on the stored int8 layout [I, O4 * 4]: RW rows x 16 outputs per block, the int8 -> float of each weight shared by the rows. */
static void lin_i8n(const f32 *x, int T, int I, const int8_t *w, int O4, const f32 *scale, const f32 *bias, f32 *y, int act, int res) {
  const int O = O4 * 4, O16 = O & ~15;
  for (int t0 = 0; t0 < T; t0 += RW) {
    const int rows = T - t0 < RW ? T - t0 : RW;
    const f32 *xr[RW];
    for (int r = 0; r < RW; r++) xr[r] = x + (t0 + (r < rows ? r : 0)) * I;
    for (int o = 0; o < O16; o += 16) {
      const int8_t *wp = w + o;
#define W16I(k) i8x16_f32(wp + (size_t)(k) * O, &w0, &w1, &w2, &w3);
      BLOCK16(W16I)
#undef W16I
      for (int r = 0; r < rows; r++) for (int q = 0; q < 4; q++) finish(y + (t0 + r) * O + o + 4 * q, a[r][q], scale + o + 4 * q, bias + o + 4 * q, act, res);
    }
    for (int o = O16; o < O; o += 4) {
      for (int r = 0; r < rows; r++) {
        float32x4_t acc = vdupq_n_f32(0);
        const int8_t *wp = w + o;
        for (int i = 0; i < I; i++, wp += O) acc = vaddq_f32(acc, vmulq_f32(vdupq_n_f32(xr[r][i]), w4_i8(wp)));
        finish(y + (t0 + r) * O + o, acc, scale + o, bias + o, act, res);
      }
    }
  }
}
#undef STEP
#undef BLOCK16
