// Native x86 builds only (the browser uses the wasm build): lin_p16 with 256-bit (WIDE=256, AVX2) or 512-bit (WIDE=512, AVX-512F)
// vectors and, for 512, lin_i8 too. Wider vectors cover more OUTPUTS per instruction; each output still sums mul then add
// (no FMA) over i = 0..I-1 in order, so y is bit-identical to the wasm kernels. RW: rows per block (12 is best for 512).
#include <immintrin.h>
#if WIDE == 512
#ifndef RW
#define RW 12
#endif
__attribute__((target("avx512f"))) static void lin_p16w(const f32 *x, int T, int I, const f32 *w, int O4, const f32 *scale, const f32 *bias, f32 *y, int act, int res) {
  const int O = O4 * 4, O16 = O & ~15;
  for (int t0 = 0; t0 < T; t0 += RW) {
    const int rows = T - t0 < RW ? T - t0 : RW;
    const f32 *xr[RW];
    for (int r = 0; r < RW; r++) xr[r] = x + (t0 + (r < rows ? r : 0)) * I;
    for (int o = 0; o < O16; o += 16) {
      __m512 a[RW];
      for (int r = 0; r < RW; r++) a[r] = _mm512_setzero_ps();
      const f32 *wp = w + (o / 16) * I * 16;
      for (int i = 0; i < I; i++, wp += 16) {
        __m512 w0 = _mm512_loadu_ps(wp);
                _Pragma("clang loop unroll(full)") for (int r = 0; r < RW; r++) a[r] = _mm512_add_ps(a[r], _mm512_mul_ps(_mm512_set1_ps(xr[r][i]), w0));
      }
      for (int r = 0; r < rows; r++) {
        f32 t[16]; _mm512_storeu_ps(t, a[r]);
        for (int q = 0; q < 4; q++) finish(y + (t0 + r) * O + o + 4 * q, _mm_loadu_ps(t + 4 * q), scale + o + 4 * q, bias + o + 4 * q, act, res);
      }
    }
    for (int o = O16; o < O; o += 4) {
      const f32 *wp0 = w + O16 * I + (o - O16) * I;
      for (int r = 0; r < rows; r++) {
        __m128 acc = _mm_setzero_ps();
        const f32 *wp = wp0;
        for (int i = 0; i < I; i++, wp += 4) acc = _mm_add_ps(acc, _mm_mul_ps(_mm_set1_ps(xr[r][i]), _mm_loadu_ps(wp)));
        finish(y + (t0 + r) * O + o, acc, scale + o, bias + o, act, res);
      }
    }
  }
}


/**
 * lin_i8 with 512-bit vectors on the stored int8 layout [I, O4 * 4] (16 outputs per zmm; int8 -> int32 -> float exact
 * per lane), RW rows per block; every output sums x * float(w) for i = 0..I-1 in order, no FMA: bit-identical to lin_i8.
 */
__attribute__((target("avx512f"))) static void lin_i8w(const f32 *x, int T, int I, const int8_t *w, int O4, const f32 *scale, const f32 *bias, f32 *y, int act, int res) {
  const int O = O4 * 4, O16 = O & ~15;
  for (int t0 = 0; t0 < T; t0 += RW) {
    const int rows = T - t0 < RW ? T - t0 : RW;
    const f32 *xr[RW];
    for (int r = 0; r < RW; r++) xr[r] = x + (t0 + (r < rows ? r : 0)) * I;
    for (int o = 0; o < O16; o += 16) {
      __m512 a[RW];
      for (int r = 0; r < RW; r++) a[r] = _mm512_setzero_ps();
      const int8_t *wp = w + o;
      for (int i = 0; i < I; i++, wp += O) {
        __m512 w0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)wp)));
        _Pragma("clang loop unroll(full)") for (int r = 0; r < RW; r++) a[r] = _mm512_add_ps(a[r], _mm512_mul_ps(_mm512_set1_ps(xr[r][i]), w0));
      }
      for (int r = 0; r < rows; r++) {
        f32 t[16]; _mm512_storeu_ps(t, a[r]);
        for (int q = 0; q < 4; q++) finish(y + (t0 + r) * O + o + 4 * q, _mm_loadu_ps(t + 4 * q), scale + o + 4 * q, bias + o + 4 * q, act, res);
      }
    }
    for (int o = O16; o < O; o += 4) {
      for (int r = 0; r < rows; r++) {
        __m128 acc = _mm_setzero_ps();
        const int8_t *wp = w + o;
        for (int i = 0; i < I; i++, wp += O) acc = _mm_add_ps(acc, _mm_mul_ps(_mm_set1_ps(xr[r][i]), w4_i8(wp)));
        finish(y + (t0 + r) * O + o, acc, scale + o, bias + o, act, res);
      }
    }
  }
}
#else
#define RW 6
__attribute__((target("avx2"))) static void lin_p16w(const f32 *x, int T, int I, const f32 *w, int O4, const f32 *scale, const f32 *bias, f32 *y, int act, int res) {
  const int O = O4 * 4, O16 = O & ~15;
  for (int t0 = 0; t0 < T; t0 += RW) {
    const int rows = T - t0 < RW ? T - t0 : RW;
    const f32 *xr[RW];
    for (int r = 0; r < RW; r++) xr[r] = x + (t0 + (r < rows ? r : 0)) * I;
    for (int o = 0; o < O16; o += 16) {
      __m256 a[RW][2];
      for (int r = 0; r < RW; r++) a[r][0] = a[r][1] = _mm256_setzero_ps();
      const f32 *wp = w + (o / 16) * I * 16;
      for (int i = 0; i < I; i++, wp += 16) {
        __m256 w0 = _mm256_loadu_ps(wp), w1 = _mm256_loadu_ps(wp + 8);
        _Pragma("clang loop unroll(full)") for (int r = 0; r < RW; r++) {
          __m256 s = _mm256_set1_ps(xr[r][i]);
          a[r][0] = _mm256_add_ps(a[r][0], _mm256_mul_ps(s, w0)); a[r][1] = _mm256_add_ps(a[r][1], _mm256_mul_ps(s, w1));
        }
      }
      for (int r = 0; r < rows; r++) {
        f32 t[16]; _mm256_storeu_ps(t, a[r][0]); _mm256_storeu_ps(t + 8, a[r][1]);
        for (int q = 0; q < 4; q++) finish(y + (t0 + r) * O + o + 4 * q, _mm_loadu_ps(t + 4 * q), scale + o + 4 * q, bias + o + 4 * q, act, res);
      }
    }
    for (int o = O16; o < O; o += 4) {
      const f32 *wp0 = w + O16 * I + (o - O16) * I;
      for (int r = 0; r < rows; r++) {
        __m128 acc = _mm_setzero_ps();
        const f32 *wp = wp0;
        for (int i = 0; i < I; i++, wp += 4) acc = _mm_add_ps(acc, _mm_mul_ps(_mm_set1_ps(xr[r][i]), _mm_loadu_ps(wp)));
        finish(y + (t0 + r) * O + o, acc, scale + o, bias + o, act, res);
      }
    }
  }
}
#endif
