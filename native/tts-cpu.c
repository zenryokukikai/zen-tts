// The TTS's matrix and signal kernels on the CPU (WebAssembly SIMD128, freestanding: no libc, no runtime glue).
// Each kernel is the CPU form of a WGSL kernel of src/runtime/tts-wgsl.ts / vits-attn.ts / nsf-wgsl.ts with the same
// math (only the order of float additions differs): lin_* = linear(), im2col = IM2COL, dw_norm = DW_NORM, vattn = VITS_ATTN.
// Memory is imported (JS lays out every buffer from __heap_base); pointers are byte offsets into it.
// Build: pnpm build:wasm (scripts/build-cpu-wasm.mjs -> cpu/tts-cpu.wasm).
#include <wasm_simd128.h>
#include <stdint.h>

#define EXPORT(n) __attribute__((export_name(n)))

typedef float f32;

/** exp (Cephes expf: |rel error| ~ 2 ulp), inputs clamped to the float range. */
static inline f32 exp_(f32 x) {
  if (x > 88.72f) x = 88.72f;
  if (x < -87.33f) x = -87.33f;
  f32 n = __builtin_floorf(x * 1.44269504088896341f + 0.5f);
  f32 r = x - n * 0.693359375f + n * 2.12194440e-4f;
  f32 p = 1.9875691500e-4f;
  p = p * r + 1.3981999507e-3f;
  p = p * r + 8.3334519073e-3f;
  p = p * r + 4.1665795894e-2f;
  p = p * r + 1.6666665459e-1f;
  p = p * r + 5.0000001201e-1f;
  p = p * r * r + r + 1.0f;
  union { uint32_t u; f32 f; } s = { .u = (uint32_t)((int32_t)n + 127) << 23 };
  return p * s.f;
}

/** GELU with erf by Abramowitz-Stegun 7.1.26, as the WGSL GELU (scalar reference of gelu4). */
#ifdef SCALAR_GELU
static inline f32 gelu(f32 v) {
  f32 u = v * 0.70710678118654752f;
  f32 a = __builtin_fabsf(u);
  f32 t = 1.0f / (1.0f + 0.3275911f * a);
  f32 y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp_(-a * a);
  return 0.5f * v * (1.0f + (u >= 0.0f ? y : -y));
}
#endif

/**
 * exp_ on 4 lanes: the same operations in the same order as the scalar exp_ (lane-wise IEEE mul / add / floor; the
 * clamps as min / max, equal to the scalar compares for every non-NaN input), so each lane is bit-identical to exp_.
 */
static inline v128_t exp4(v128_t x) {
  x = wasm_f32x4_min(x, wasm_f32x4_splat(88.72f));
  x = wasm_f32x4_max(x, wasm_f32x4_splat(-87.33f));
  v128_t n = wasm_f32x4_floor(wasm_f32x4_add(wasm_f32x4_mul(x, wasm_f32x4_splat(1.44269504088896341f)), wasm_f32x4_splat(0.5f)));
  v128_t r = wasm_f32x4_add(wasm_f32x4_sub(x, wasm_f32x4_mul(n, wasm_f32x4_splat(0.693359375f))), wasm_f32x4_mul(n, wasm_f32x4_splat(2.12194440e-4f)));
  v128_t p = wasm_f32x4_splat(1.9875691500e-4f);
  p = wasm_f32x4_add(wasm_f32x4_mul(p, r), wasm_f32x4_splat(1.3981999507e-3f));
  p = wasm_f32x4_add(wasm_f32x4_mul(p, r), wasm_f32x4_splat(8.3334519073e-3f));
  p = wasm_f32x4_add(wasm_f32x4_mul(p, r), wasm_f32x4_splat(4.1665795894e-2f));
  p = wasm_f32x4_add(wasm_f32x4_mul(p, r), wasm_f32x4_splat(1.6666665459e-1f));
  p = wasm_f32x4_add(wasm_f32x4_mul(p, r), wasm_f32x4_splat(5.0000001201e-1f));
  p = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_mul(p, r), r), r), wasm_f32x4_splat(1.0f));
  v128_t e = wasm_i32x4_shl(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(n), wasm_i32x4_splat(127)), 23);
  return wasm_f32x4_mul(p, e);
}

/** gelu on 4 lanes, bit-identical per lane to the scalar gelu (same operations, same order). */
static inline v128_t gelu4(v128_t v) {
  v128_t u = wasm_f32x4_mul(v, wasm_f32x4_splat(0.70710678118654752f));
  v128_t a = wasm_f32x4_abs(u);
  v128_t t = wasm_f32x4_div(wasm_f32x4_splat(1.0f), wasm_f32x4_add(wasm_f32x4_splat(1.0f), wasm_f32x4_mul(wasm_f32x4_splat(0.3275911f), a)));
  v128_t q = wasm_f32x4_sub(wasm_f32x4_mul(wasm_f32x4_splat(1.061405429f), t), wasm_f32x4_splat(1.453152027f));
  q = wasm_f32x4_add(wasm_f32x4_mul(q, t), wasm_f32x4_splat(1.421413741f));
  q = wasm_f32x4_sub(wasm_f32x4_mul(q, t), wasm_f32x4_splat(0.284496736f));
  q = wasm_f32x4_add(wasm_f32x4_mul(q, t), wasm_f32x4_splat(0.254829592f));
  v128_t y = wasm_f32x4_sub(wasm_f32x4_splat(1.0f), wasm_f32x4_mul(wasm_f32x4_mul(q, t), exp4(wasm_f32x4_mul(wasm_f32x4_neg(a), a))));
  v128_t sy = wasm_v128_bitselect(y, wasm_f32x4_neg(y), wasm_f32x4_ge(u, wasm_f32x4_splat(0.0f)));
  return wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_splat(0.5f), v), wasm_f32x4_add(wasm_f32x4_splat(1.0f), sy));
}

static inline v128_t act4(v128_t v, int act) {
  if (act == 2) return wasm_f32x4_max(v, wasm_f32x4_splat(0.0f));
#ifdef SCALAR_GELU
  if (act == 1) {
    f32 a[4];
    wasm_v128_store(a, v);
    return wasm_f32x4_make(gelu(a[0]), gelu(a[1]), gelu(a[2]), gelu(a[3]));
  }
#else
  if (act == 1) return gelu4(v);
#endif
  return v;
}

static inline void finish(f32 *y, v128_t acc, const f32 *scale, const f32 *bias, int act, int res) {
  v128_t v = wasm_f32x4_add(wasm_f32x4_mul(acc, wasm_v128_load(scale)), wasm_v128_load(bias));
  v = act4(v, act);
  if (res) v = wasm_f32x4_add(v, wasm_v128_load(y));
  wasm_v128_store(y, v);
}

static inline v128_t w4_f32(const f32 *w) { return wasm_v128_load(w); }
static inline v128_t w4_i8(const int8_t *w) {
  v128_t b = wasm_v128_load32_zero(w);
  return wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(wasm_i16x8_extend_low_i8x16(b)));
}

/**
 * y[t, o] (+)= act(scale[o] * sum_i x[t, i] w[i, o] + bias[o]) for o < O4 * 4 (row stride of w and y O4 * 4, scale and bias
 * O4 * 4 floats each); act 1 GELU, 2 ReLU; res 1 adds to y. 4 rows x 16 outputs per inner block.
 */
#define LINEAR(NAME, WT, W4)                                                                                             \
  EXPORT(#NAME) void NAME(const f32 *x, int T, int I, const WT *w, int O4, const f32 *scale, const f32 *bias, f32 *y, int act, int res) { \
    const int O = O4 * 4;                                                                                                \
    for (int t0 = 0; t0 < T; t0 += 4) {                                                                                  \
      const int rows = T - t0 < 4 ? T - t0 : 4;                                                                          \
      const f32 *x0 = x + (t0 + 0) * I;                                                                                  \
      const f32 *x1 = x + (t0 + (rows > 1 ? 1 : 0)) * I;                                                                 \
      const f32 *x2 = x + (t0 + (rows > 2 ? 2 : 0)) * I;                                                                 \
      const f32 *x3 = x + (t0 + (rows > 3 ? 3 : 0)) * I;                                                                 \
      int o = 0;                                                                                                         \
      for (; o + 16 <= O; o += 16) {                                                                                     \
        v128_t a00 = wasm_f32x4_splat(0), a01 = a00, a02 = a00, a03 = a00, a10 = a00, a11 = a00, a12 = a00, a13 = a00;   \
        v128_t a20 = a00, a21 = a00, a22 = a00, a23 = a00, a30 = a00, a31 = a00, a32 = a00, a33 = a00;                   \
        const WT *wp = w + o;                                                                                            \
        for (int i = 0; i < I; i++, wp += O) {                                                                           \
          v128_t w0 = W4(wp), w1 = W4(wp + 4), w2 = W4(wp + 8), w3 = W4(wp + 12);                                        \
          v128_t s0 = wasm_f32x4_splat(x0[i]), s1 = wasm_f32x4_splat(x1[i]), s2 = wasm_f32x4_splat(x2[i]), s3 = wasm_f32x4_splat(x3[i]); \
          a00 = wasm_f32x4_add(a00, wasm_f32x4_mul(s0, w0)); a01 = wasm_f32x4_add(a01, wasm_f32x4_mul(s0, w1));          \
          a02 = wasm_f32x4_add(a02, wasm_f32x4_mul(s0, w2)); a03 = wasm_f32x4_add(a03, wasm_f32x4_mul(s0, w3));          \
          a10 = wasm_f32x4_add(a10, wasm_f32x4_mul(s1, w0)); a11 = wasm_f32x4_add(a11, wasm_f32x4_mul(s1, w1));          \
          a12 = wasm_f32x4_add(a12, wasm_f32x4_mul(s1, w2)); a13 = wasm_f32x4_add(a13, wasm_f32x4_mul(s1, w3));          \
          a20 = wasm_f32x4_add(a20, wasm_f32x4_mul(s2, w0)); a21 = wasm_f32x4_add(a21, wasm_f32x4_mul(s2, w1));          \
          a22 = wasm_f32x4_add(a22, wasm_f32x4_mul(s2, w2)); a23 = wasm_f32x4_add(a23, wasm_f32x4_mul(s2, w3));          \
          a30 = wasm_f32x4_add(a30, wasm_f32x4_mul(s3, w0)); a31 = wasm_f32x4_add(a31, wasm_f32x4_mul(s3, w1));          \
          a32 = wasm_f32x4_add(a32, wasm_f32x4_mul(s3, w2)); a33 = wasm_f32x4_add(a33, wasm_f32x4_mul(s3, w3));          \
        }                                                                                                                \
        f32 *y0 = y + t0 * O + o;                                                                                        \
        finish(y0, a00, scale + o, bias + o, act, res); finish(y0 + 4, a01, scale + o + 4, bias + o + 4, act, res);      \
        finish(y0 + 8, a02, scale + o + 8, bias + o + 8, act, res); finish(y0 + 12, a03, scale + o + 12, bias + o + 12, act, res); \
        if (rows > 1) { f32 *y1 = y0 + O; finish(y1, a10, scale + o, bias + o, act, res); finish(y1 + 4, a11, scale + o + 4, bias + o + 4, act, res); \
          finish(y1 + 8, a12, scale + o + 8, bias + o + 8, act, res); finish(y1 + 12, a13, scale + o + 12, bias + o + 12, act, res); } \
        if (rows > 2) { f32 *y2 = y0 + 2 * O; finish(y2, a20, scale + o, bias + o, act, res); finish(y2 + 4, a21, scale + o + 4, bias + o + 4, act, res); \
          finish(y2 + 8, a22, scale + o + 8, bias + o + 8, act, res); finish(y2 + 12, a23, scale + o + 12, bias + o + 12, act, res); } \
        if (rows > 3) { f32 *y3 = y0 + 3 * O; finish(y3, a30, scale + o, bias + o, act, res); finish(y3 + 4, a31, scale + o + 4, bias + o + 4, act, res); \
          finish(y3 + 8, a32, scale + o + 8, bias + o + 8, act, res); finish(y3 + 12, a33, scale + o + 12, bias + o + 12, act, res); } \
      }                                                                                                                  \
      for (; o < O; o += 4) {                                                                                            \
        v128_t a0 = wasm_f32x4_splat(0), a1 = a0, a2 = a0, a3 = a0;                                                      \
        const WT *wp = w + o;                                                                                            \
        for (int i = 0; i < I; i++, wp += O) {                                                                           \
          v128_t w0 = W4(wp);                                                                                            \
          a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(wasm_f32x4_splat(x0[i]), w0));                                          \
          a1 = wasm_f32x4_add(a1, wasm_f32x4_mul(wasm_f32x4_splat(x1[i]), w0));                                          \
          a2 = wasm_f32x4_add(a2, wasm_f32x4_mul(wasm_f32x4_splat(x2[i]), w0));                                          \
          a3 = wasm_f32x4_add(a3, wasm_f32x4_mul(wasm_f32x4_splat(x3[i]), w0));                                          \
        }                                                                                                                \
        f32 *y0 = y + t0 * O + o;                                                                                        \
        finish(y0, a0, scale + o, bias + o, act, res);                                                                   \
        if (rows > 1) finish(y0 + O, a1, scale + o, bias + o, act, res);                                                 \
        if (rows > 2) finish(y0 + 2 * O, a2, scale + o, bias + o, act, res);                                             \
        if (rows > 3) finish(y0 + 3 * O, a3, scale + o, bias + o, act, res);                                             \
      }                                                                                                                  \
    }                                                                                                                    \
  }

LINEAR(lin_f32, f32, w4_f32)
LINEAR(lin_i8, int8_t, w4_i8)

/**
 * lin_f32 with the float32 weights packed in panels of 16 outputs (CpuKernels.linear 'p16' layout: output o < O16 at
 * w[(o / 16) * I * 16 + i * 16 + o % 16], the last O % 16 outputs in panels of 4 at w[O16 * I + ((o - O16) / 4) * I * 4 +
 * i * 4 + o % 4]); 3 rows x 16 outputs per block, so each panel's 4 vector loads per i are shared by 3 rows. Every output
 * still sums i = 0..I-1 in order (acc + x * w per step, no fused multiply-add), so y is bit-identical to lin_f32.
 */
#define P16_ROWS 3
EXPORT("lin_p16") void lin_p16(const f32 *x, int T, int I, const f32 *w, int O4, const f32 *scale, const f32 *bias, f32 *y, int act, int res) {
  const int O = O4 * 4;
  const int O16 = O & ~15;
  for (int t0 = 0; t0 < T; t0 += P16_ROWS) {
    const int rows = T - t0 < P16_ROWS ? T - t0 : P16_ROWS;
    const f32 *xr[P16_ROWS];
    for (int r = 0; r < P16_ROWS; r++) xr[r] = x + (t0 + (r < rows ? r : 0)) * I;
    for (int o = 0; o < O16; o += 16) {
      v128_t a[P16_ROWS][4];
      _Pragma("clang loop unroll(full)") for (int r = 0; r < P16_ROWS; r++) { a[r][0] = wasm_f32x4_splat(0); a[r][1] = a[r][0]; a[r][2] = a[r][0]; a[r][3] = a[r][0]; }
      const f32 *wp = w + (o / 16) * I * 16;
      for (int i = 0; i < I; i++, wp += 16) {
        v128_t w0 = wasm_v128_load(wp), w1 = wasm_v128_load(wp + 4), w2 = wasm_v128_load(wp + 8), w3 = wasm_v128_load(wp + 12);
        _Pragma("clang loop unroll(full)") for (int r = 0; r < P16_ROWS; r++) {
          v128_t s = wasm_f32x4_splat(xr[r][i]);
          a[r][0] = wasm_f32x4_add(a[r][0], wasm_f32x4_mul(s, w0)); a[r][1] = wasm_f32x4_add(a[r][1], wasm_f32x4_mul(s, w1));
          a[r][2] = wasm_f32x4_add(a[r][2], wasm_f32x4_mul(s, w2)); a[r][3] = wasm_f32x4_add(a[r][3], wasm_f32x4_mul(s, w3));
        }
      }
      for (int r = 0; r < rows; r++) for (int q = 0; q < 4; q++)
        finish(y + (t0 + r) * O + o + 4 * q, a[r][q], scale + o + 4 * q, bias + o + 4 * q, act, res);
    }
    for (int o = O16; o < O; o += 4) {
      const f32 *wp0 = w + O16 * I + (o - O16) * I;
      for (int r = 0; r < rows; r++) {
        v128_t acc = wasm_f32x4_splat(0);
        const f32 *wp = wp0;
        for (int i = 0; i < I; i++, wp += 4) acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_f32x4_splat(xr[r][i]), wasm_v128_load(wp)));
        finish(y + (t0 + r) * O + o, acc, scale + o, bias + o, act, res);
      }
    }
  }
}

/** col[t, k * C + c] = x[t + k - K / 2, c] (0 outside 0..T-1). */
EXPORT("im2col") void im2col(const f32 *x, int T, int C, int K, f32 *col) {
  for (int t = 0; t < T; t++) {
    for (int k = 0; k < K; k++) {
      int tt = t + k - K / 2;
      f32 *d = col + (t * K + k) * C;
      if (tt < 0 || tt >= T) { for (int c = 0; c < C; c++) d[c] = 0.0f; }
      else { const f32 *s = x + tt * C; for (int c = 0; c < C; c++) d[c] = s[c]; }
    }
  }
}

/**
 * y[t] = LayerNorm(conv ? dwconv(x)[t] : x[t]); prm = [dw weight (D * K) | dw bias (D) |] ln weight (D) | ln bias (D).
 * kt: scratch of K * D floats (the dw weight transposed to [k, c]).
 */
EXPORT("dw_norm") void dw_norm(const f32 *x, int T, int D, int K, int conv, const f32 *prm, f32 *y, f32 eps, f32 *kt) {
  const f32 *lw = prm + (conv ? D * K + D : 0);
  const f32 *lb = lw + D;
  if (conv) for (int c = 0; c < D; c++) for (int k = 0; k < K; k++) kt[k * D + c] = prm[c * K + k];
  for (int t = 0; t < T; t++) {
    f32 *yt = y + t * D;
    if (conv) {
      const f32 *b = prm + D * K;
      for (int c = 0; c < D; c++) yt[c] = b[c];
      for (int k = 0; k < K; k++) {
        int tt = t + k - K / 2;
        if (tt < 0 || tt >= T) continue;
        const f32 *xs = x + tt * D;
        const f32 *wk = kt + k * D;
        int c = 0;
        for (; c + 4 <= D; c += 4) wasm_v128_store(yt + c, wasm_f32x4_add(wasm_v128_load(yt + c), wasm_f32x4_mul(wasm_v128_load(wk + c), wasm_v128_load(xs + c))));
        for (; c < D; c++) yt[c] += wk[c] * xs[c];
      }
    } else {
      const f32 *xt = x + t * D;
      for (int c = 0; c < D; c++) yt[c] = xt[c];
    }
    f32 s = 0.0f;
    for (int c = 0; c < D; c++) s += yt[c];
    f32 mean = s / (f32)D;
    f32 q = 0.0f;
    for (int c = 0; c < D; c++) { f32 d = yt[c] - mean; q += d * d; }
    f32 r = 1.0f / __builtin_sqrtf(q / (f32)D + eps);
    for (int c = 0; c < D; c++) yt[c] = (yt[c] - mean) * r * lw[c] + lb[c];
  }
}

/**
 * VITS relative attention on qkv [T, 3 * H * DH] (window W; heads share relk / relv [2W + 1, DH]):
 * s_j = (q_i . (k_j + [|j - i| <= W] relk[j - i + W])) / sqrt(DH); o_i = sum_j p_j (v_j + [|j - i| <= W] relv[j - i + W]).
 * sc: scratch of T floats.
 */
EXPORT("vattn") void vattn(const f32 *qkv, int T, int H, int DH, int W, const f32 *relk, const f32 *relv, f32 *o, f32 *sc) {
  const int D = H * DH;
  const f32 r = 1.0f / __builtin_sqrtf((f32)DH);
  for (int h = 0; h < H; h++) {
    for (int i = 0; i < T; i++) {
      const f32 *q = qkv + i * 3 * D + h * DH;
      f32 m = -3.4e38f;
      for (int j = 0; j < T; j++) {
        const f32 *k = qkv + j * 3 * D + D + h * DH;
        int d = j - i;
        v128_t acc = wasm_f32x4_splat(0.0f);
        f32 s = 0.0f;
        int c = 0;
        if (d >= -W && d <= W) {
          const f32 *rk = relk + (d + W) * DH;
          for (; c + 4 <= DH; c += 4) acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(q + c), wasm_f32x4_add(wasm_v128_load(k + c), wasm_v128_load(rk + c))));
          for (; c < DH; c++) s += q[c] * (k[c] + rk[c]);
        } else {
          for (; c + 4 <= DH; c += 4) acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(q + c), wasm_v128_load(k + c)));
          for (; c < DH; c++) s += q[c] * k[c];
        }
        s += wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) + wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3);
        s *= r;
        sc[j] = s;
        if (s > m) m = s;
      }
      f32 su = 0.0f;
      for (int j = 0; j < T; j++) { f32 e = exp_(sc[j] - m); sc[j] = e; su += e; }
      f32 *out = o + i * D + h * DH;
      for (int c = 0; c < DH; c++) out[c] = 0.0f;
      for (int j = 0; j < T; j++) {
        const f32 *v = qkv + j * 3 * D + 2 * D + h * DH;
        v128_t p = wasm_f32x4_splat(sc[j]);
        int c = 0;
        for (; c + 4 <= DH; c += 4) wasm_v128_store(out + c, wasm_f32x4_add(wasm_v128_load(out + c), wasm_f32x4_mul(p, wasm_v128_load(v + c))));
        for (; c < DH; c++) out[c] += sc[j] * v[c];
      }
      int lo = i - W < 0 ? 0 : i - W;
      int hi = i + W > T - 1 ? T - 1 : i + W;
      for (int j = lo; j <= hi; j++) {
        const f32 *rv = relv + (j + W - i) * DH;
        for (int c = 0; c < DH; c++) out[c] += sc[j] * rv[c];
      }
      for (int c = 0; c < DH; c++) out[c] = out[c] / su;
    }
  }
}

/* ---- NSF decoder signal part (nsf-wgsl.ts EXC_SCAN + EXC_PREFIX + EXC_SOURCE, STFT_FILTER, ISTFT, OLA) ---- */

#define NSF_HOP 512
#define NSF_NFFT 2048
#define NSF_NH 1024
#define NSF_BINS 1025
#define NSF_SR 44100.0f
#define PI2 6.283185307179586f

/** log (Cephes logf), x > 0 and normal. */
static inline f32 log_(f32 v) {
  union { f32 f; uint32_t u; } b = { .f = v };
  int e = (int)(b.u >> 23) - 126;
  b.u = (b.u & 0x7fffff) | 0x3f000000;
  f32 x = b.f;
  if (x < 0.70710678118654752f) { e -= 1; x = x + x - 1.0f; } else x = x - 1.0f;
  f32 z = x * x;
  f32 y = 7.0376836292e-2f;
  y = y * x - 1.1514610310e-1f;
  y = y * x + 1.1676998740e-1f;
  y = y * x - 1.2420140846e-1f;
  y = y * x + 1.4249322787e-1f;
  y = y * x - 1.6668057665e-1f;
  y = y * x + 2.0000714765e-1f;
  y = y * x - 2.4999993993e-1f;
  y = y * x + 3.3333331174e-1f;
  y = y * x * z;
  f32 fe = (f32)e;
  y += -2.12194440e-4f * fe;
  y += -0.5f * z;
  return x + y + 0.693359375f * fe;
}

/** log_ on 4 lanes (x > 0 and normal): the same operations in the same order per lane as log_. */
static inline v128_t log4(v128_t v) {
  v128_t e = wasm_i32x4_sub(wasm_u32x4_shr(v, 23), wasm_i32x4_splat(126));
  v128_t m = wasm_v128_or(wasm_v128_and(v, wasm_i32x4_splat(0x7fffff)), wasm_i32x4_splat(0x3f000000));
  v128_t lt = wasm_f32x4_lt(m, wasm_f32x4_splat(0.70710678118654752f));
  e = wasm_i32x4_add(e, lt);
  v128_t x = wasm_v128_bitselect(wasm_f32x4_sub(wasm_f32x4_add(m, m), wasm_f32x4_splat(1.0f)), wasm_f32x4_sub(m, wasm_f32x4_splat(1.0f)), lt);
  v128_t z = wasm_f32x4_mul(x, x);
  v128_t y = wasm_f32x4_splat(7.0376836292e-2f);
  y = wasm_f32x4_sub(wasm_f32x4_mul(y, x), wasm_f32x4_splat(1.1514610310e-1f));
  y = wasm_f32x4_add(wasm_f32x4_mul(y, x), wasm_f32x4_splat(1.1676998740e-1f));
  y = wasm_f32x4_sub(wasm_f32x4_mul(y, x), wasm_f32x4_splat(1.2420140846e-1f));
  y = wasm_f32x4_add(wasm_f32x4_mul(y, x), wasm_f32x4_splat(1.4249322787e-1f));
  y = wasm_f32x4_sub(wasm_f32x4_mul(y, x), wasm_f32x4_splat(1.6668057665e-1f));
  y = wasm_f32x4_add(wasm_f32x4_mul(y, x), wasm_f32x4_splat(2.0000714765e-1f));
  y = wasm_f32x4_sub(wasm_f32x4_mul(y, x), wasm_f32x4_splat(2.4999993993e-1f));
  y = wasm_f32x4_add(wasm_f32x4_mul(y, x), wasm_f32x4_splat(3.3333331174e-1f));
  y = wasm_f32x4_mul(wasm_f32x4_mul(y, x), z);
  v128_t fe = wasm_f32x4_convert_i32x4(e);
  y = wasm_f32x4_add(y, wasm_f32x4_mul(wasm_f32x4_splat(-2.12194440e-4f), fe));
  y = wasm_f32x4_add(y, wasm_f32x4_mul(wasm_f32x4_splat(-0.5f), z));
  return wasm_f32x4_add(wasm_f32x4_add(x, y), wasm_f32x4_mul(wasm_f32x4_splat(0.693359375f), fe));
}

/** sin and cos (Cephes sinf / cosf polynomials after a 3-part reduction by pi / 2). */
static inline void sincos_(f32 x, f32 *s, f32 *c) {
  f32 j = __builtin_floorf(x * 0.63661977236758134f + 0.5f);
  f32 r = ((x - j * 1.5703125f) - j * 4.837512969970703125e-4f) - j * 7.54978995489188216e-8f;
  f32 z = r * r;
  f32 sr = r + r * z * (-1.6666654611e-1f + z * (8.3321608736e-3f + z * -1.9515295891e-4f));
  f32 cr = 1.0f - 0.5f * z + z * z * (4.166664568298827e-2f + z * (-1.388731625493765e-3f + z * 2.443315711809948e-5f));
  int q = (int)j & 3;
  if (q == 0) { *s = sr; *c = cr; } else if (q == 1) { *s = cr; *c = -sr; } else if (q == 2) { *s = -sr; *c = -cr; } else { *s = -cr; *c = sr; }
}
__attribute__((unused)) static inline f32 sin_(f32 x) { f32 s, c; sincos_(x, &s, &c); return s; }
/**
 * sincos_ on 4 lanes: the same operations in the same order per lane as sincos_ (the quadrant picks by bitselect), so
 * every lane is bit-identical to the scalar sincos_.
 */
static inline void sincos4(v128_t x, v128_t *s, v128_t *c) {
  v128_t j = wasm_f32x4_floor(wasm_f32x4_add(wasm_f32x4_mul(x, wasm_f32x4_splat(0.63661977236758134f)), wasm_f32x4_splat(0.5f)));
  v128_t r = wasm_f32x4_sub(wasm_f32x4_sub(wasm_f32x4_sub(x, wasm_f32x4_mul(j, wasm_f32x4_splat(1.5703125f))), wasm_f32x4_mul(j, wasm_f32x4_splat(4.837512969970703125e-4f))), wasm_f32x4_mul(j, wasm_f32x4_splat(7.54978995489188216e-8f)));
  v128_t z = wasm_f32x4_mul(r, r);
  v128_t sp = wasm_f32x4_add(wasm_f32x4_splat(8.3321608736e-3f), wasm_f32x4_mul(z, wasm_f32x4_splat(-1.9515295891e-4f)));
  sp = wasm_f32x4_add(wasm_f32x4_splat(-1.6666654611e-1f), wasm_f32x4_mul(z, sp));
  v128_t sr = wasm_f32x4_add(r, wasm_f32x4_mul(wasm_f32x4_mul(r, z), sp));
  v128_t cp = wasm_f32x4_add(wasm_f32x4_splat(-1.388731625493765e-3f), wasm_f32x4_mul(z, wasm_f32x4_splat(2.443315711809948e-5f)));
  cp = wasm_f32x4_add(wasm_f32x4_splat(4.166664568298827e-2f), wasm_f32x4_mul(z, cp));
  v128_t cr = wasm_f32x4_add(wasm_f32x4_sub(wasm_f32x4_splat(1.0f), wasm_f32x4_mul(wasm_f32x4_splat(0.5f), z)), wasm_f32x4_mul(wasm_f32x4_mul(z, z), cp));
  v128_t q = wasm_v128_and(wasm_i32x4_trunc_sat_f32x4(j), wasm_i32x4_splat(3));
  // q 0: (sr, cr), 1: (cr, -sr), 2: (-sr, -cr), 3: (-cr, sr)
  v128_t odd = wasm_i32x4_eq(wasm_v128_and(q, wasm_i32x4_splat(1)), wasm_i32x4_splat(1));
  v128_t sneg = wasm_i32x4_eq(wasm_v128_and(q, wasm_i32x4_splat(2)), wasm_i32x4_splat(2));
  v128_t cneg = wasm_i32x4_eq(wasm_v128_and(wasm_i32x4_add(q, wasm_i32x4_splat(1)), wasm_i32x4_splat(2)), wasm_i32x4_splat(2));
  v128_t sv = wasm_v128_bitselect(cr, sr, odd), cv = wasm_v128_bitselect(sr, cr, odd);
  *s = wasm_v128_bitselect(wasm_f32x4_neg(sv), sv, sneg);
  *c = wasm_v128_bitselect(wasm_f32x4_neg(cv), cv, cneg);
}
static inline v128_t sin4(v128_t x) { v128_t s, c; sincos4(x, &s, &c); return s; }
static inline v128_t cos4(v128_t x) { v128_t s, c; sincos4(x, &s, &c); return c; }

__attribute__((unused)) static inline f32 cos_(f32 x) { f32 s, c; sincos_(x, &s, &c); return c; }
static inline f32 fract_(f32 x) { return x - __builtin_floorf(x); }

static inline uint32_t pcg(uint32_t v) {
  uint32_t s = v * 747796405u + 2891336453u;
  uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
__attribute__((unused)) static inline f32 wrap_(f32 a) { return a - PI2 * __builtin_floorf(a / PI2 + 0.5f); }

static f32 exc_sc[NSF_HOP];
static f32 exc_up[2 * NSF_HOP];
#ifdef __wasm__
static inline v128_t rev4(v128_t a) { return wasm_i32x4_shuffle(a, a, 3, 2, 1, 0); }
static inline v128_t zip_lo(v128_t a, v128_t b) { return wasm_i32x4_shuffle(a, b, 0, 4, 1, 5); }
static inline v128_t zip_hi(v128_t a, v128_t b) { return wasm_i32x4_shuffle(a, b, 2, 6, 3, 7); }
#endif

/**
 * Excitation of T frames: ex[2n] = e_h, ex[2n + 1] = e_n (noise 1: PCG + Box-Muller from seed; 0: kept as written).
 * f0 per frame upsampled (log F0 and voicing linear between frame centres), phase = f32 prefix of fract(frame totals) +
 * the frame's own scan (the GPU's 512-wide scan as a sequential sum).
 */
EXPORT("nsf_exc") void nsf_exc(const f32 *f0, int T, int noise, uint32_t seed, f32 *ex) {
  // 4 samples per step; each lane does the scalar steps in the same order (exp4 / sin4 / log4 / cos4 are lane-wise
  // identical to exp_ / sin_ / log_ / cos_), and the phase prefix stays a sequential sum, so ex is bit-identical.
  f32 acc = 0.0f;
  const v128_t lane = wasm_f32x4_make(0.0f, 1.0f, 2.0f, 3.0f);
  for (int f = 0; f < T; f++) {
    int i1 = f + 1 < T ? f + 1 : T - 1;
    f32 a = f0[f], b = f0[i1];
    f32 v0 = a > 0.0f ? 1.0f : 0.0f, v1 = b > 0.0f ? 1.0f : 0.0f;
    f32 l0 = log_(a > 1.0f ? a : 1.0f) * v0, l1 = log_(b > 1.0f ? b : 1.0f) * v1;
    const v128_t V0 = wasm_f32x4_splat(v0), V1 = wasm_f32x4_splat(v1), L0 = wasm_f32x4_splat(l0), L1 = wasm_f32x4_splat(l1);
    for (int j = 0; j < NSF_HOP; j += 4) {
      v128_t w = wasm_f32x4_div(wasm_f32x4_add(wasm_f32x4_splat((f32)j), lane), wasm_f32x4_splat((f32)NSF_HOP));
      v128_t om = wasm_f32x4_sub(wasm_f32x4_splat(1.0f), w);
      v128_t vs = wasm_f32x4_add(wasm_f32x4_mul(V0, om), wasm_f32x4_mul(V1, w));
      v128_t ls = wasm_f32x4_div(wasm_f32x4_add(wasm_f32x4_mul(L0, om), wasm_f32x4_mul(L1, w)), wasm_f32x4_max(vs, wasm_f32x4_splat(1e-3f)));
      v128_t fs = wasm_v128_and(exp4(ls), wasm_f32x4_gt(vs, wasm_f32x4_splat(0.5f)));
      wasm_v128_store(exc_up + j, fs);
      wasm_v128_store(exc_up + NSF_HOP + j, vs);
      wasm_v128_store(exc_sc + j, wasm_f32x4_div(fs, wasm_f32x4_splat(NSF_SR)));
    }
    f32 run = 0.0f;
    for (int j = 0; j < NSF_HOP; j++) { run += exc_sc[j]; exc_sc[j] = run; }
    const v128_t A = wasm_f32x4_splat(acc);
    for (int j = 0; j < NSF_HOP; j += 4) {
      v128_t fs = wasm_v128_load(exc_up + j), vs = wasm_v128_load(exc_up + NSF_HOP + j);
      v128_t c = wasm_f32x4_add(A, wasm_v128_load(exc_sc + j));
      v128_t x = wasm_f32x4_mul(wasm_f32x4_splat(PI2), wasm_f32x4_sub(c, wasm_f32x4_floor(wasm_f32x4_add(c, wasm_f32x4_splat(0.5f)))));
      v128_t K = wasm_f32x4_floor(wasm_f32x4_div(wasm_f32x4_splat(0.45f * NSF_SR), wasm_f32x4_max(fs, wasm_f32x4_splat(1.0f))));
      K = wasm_f32x4_max(wasm_f32x4_min(K, wasm_f32x4_splat(320.0f)), wasm_f32x4_splat(1.0f));
      v128_t s = sin4(wasm_f32x4_mul(x, wasm_f32x4_splat(0.5f)));
      v128_t ka = wasm_f32x4_mul(wasm_f32x4_add(K, wasm_f32x4_splat(0.5f)), x);
      v128_t wr = wasm_f32x4_sub(ka, wasm_f32x4_mul(wasm_f32x4_splat(PI2), wasm_f32x4_floor(wasm_f32x4_add(wasm_f32x4_div(ka, wasm_f32x4_splat(PI2)), wasm_f32x4_splat(0.5f)))));
      v128_t dd = wasm_f32x4_sub(wasm_f32x4_div(sin4(wr), wasm_f32x4_mul(wasm_f32x4_splat(2.0f), s)), wasm_f32x4_splat(0.5f));
      v128_t d = wasm_v128_bitselect(dd, K, wasm_f32x4_ge(wasm_f32x4_abs(s), wasm_f32x4_splat(1e-4f)));
      v128_t on = wasm_v128_and(wasm_f32x4_splat(1.0f), wasm_f32x4_gt(fs, wasm_f32x4_splat(0.0f)));
      v128_t vc = wasm_f32x4_min(wasm_f32x4_max(vs, wasm_f32x4_splat(0.0f)), wasm_f32x4_splat(1.0f));
      v128_t eh = wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_splat(0.1f), d), on), vc);
      uint32_t n0 = (uint32_t)(f * NSF_HOP + j);
      if (noise == 1) {
        f32 u1[4], u2[4];
        for (int l = 0; l < 4; l++) {
          uint32_t n = n0 + (uint32_t)l;
          uint32_t h1 = pcg(n * 2u + seed * 0x9e3779b9u);
          uint32_t h2 = pcg(n * 2u + 1u + seed * 0x9e3779b9u);
          u1[l] = ((f32)(h1 >> 8u) + 1.0f) / 16777217.0f;
          u2[l] = (f32)(h2 >> 8u) / 16777216.0f;
        }
        v128_t en = wasm_f32x4_mul(wasm_f32x4_sqrt(wasm_f32x4_mul(wasm_f32x4_splat(-2.0f), log4(wasm_v128_load(u1)))), cos4(wasm_f32x4_mul(wasm_f32x4_splat(PI2), wasm_v128_load(u2))));
        wasm_v128_store(ex + 2 * n0, zip_lo(eh, en));
        wasm_v128_store(ex + 2 * n0 + 4, zip_hi(eh, en));
      } else {
        f32 e4[4];
        wasm_v128_store(e4, eh);
        for (int l = 0; l < 4; l++) ex[2 * (n0 + (uint32_t)l)] = e4[l];
      }
    }
    acc = fract_(acc + fract_(exc_sc[NSF_HOP - 1]));
  }
}

/** FFT work buffer, real and imaginary parts split (so 4 butterflies are one SIMD step). */
static f32 fre[NSF_NH], fim[NSF_NH];
/** twiddles per stage: stage hs at [hs, 2 hs): cos / sin of 2 pi pos / (2 hs) = tab[pos * NH / hs] (gathered once) */
static f32 twr[NSF_NH], twi[NSF_NH];
static const f32 *tw_of;

static inline uint32_t brev10(uint32_t m) {
  m = ((m >> 1) & 0x55555555u) | ((m & 0x55555555u) << 1);
  m = ((m >> 2) & 0x33333333u) | ((m & 0x33333333u) << 2);
  m = ((m >> 4) & 0x0f0f0f0fu) | ((m & 0x0f0f0f0fu) << 4);
  m = ((m >> 8) & 0x00ff00ffu) | ((m & 0x00ff00ffu) << 8);
  m = (m >> 16) | (m << 16);
  return m >> 22;
}

/**
 * In-place radix-2 FFT of fre / fim (1024 complex, loaded bit-reversed); sgn -1 forward, +1 inverse (unscaled).
 * tab: cos (2048) | sin (2048) of 2 pi k / 2048 (FFT_COMMON's twiddle 2 pi pos / (2 hs) = tab[2 pos NH / (2 hs)]).
 * Stages hs >= 4 run 4 butterflies per step: every lane does the same multiplies and adds in the same order as the
 * scalar butterfly (vr = br wr - bi wi, vi = br wi + bi wr, u +- v), so the result is bit-identical.
 */
static void fft1024(f32 sgn, const f32 *tab) {
  if (tw_of != tab) {
    for (int hs = 1; hs < NSF_NH; hs <<= 1) for (int pos = 0; pos < hs; pos++) { twr[hs + pos] = tab[pos * (NSF_NH / hs)]; twi[hs + pos] = tab[NSF_NFFT + pos * (NSF_NH / hs)]; }
    tw_of = tab;
  }
  for (int hs = 1; hs < 4; hs <<= 1) {
    for (int g = 0; g < NSF_NH; g += 2 * hs) {
      for (int pos = 0; pos < hs; pos++) {
        int i0 = g + pos, i1 = i0 + hs;
        f32 wr = twr[hs + pos], wi = sgn * twi[hs + pos];
        f32 br = fre[i1], bi = fim[i1];
        f32 vr = br * wr - bi * wi, vi = br * wi + bi * wr;
        f32 ur = fre[i0], ui = fim[i0];
        fre[i0] = ur + vr; fim[i0] = ui + vi;
        fre[i1] = ur - vr; fim[i1] = ui - vi;
      }
    }
  }
  const v128_t sg = wasm_f32x4_splat(sgn);
  for (int hs = 4; hs < NSF_NH; hs <<= 1) {
    for (int g = 0; g < NSF_NH; g += 2 * hs) {
      for (int pos = 0; pos < hs; pos += 4) {
        int i0 = g + pos, i1 = i0 + hs;
        v128_t wr = wasm_v128_load(twr + hs + pos), wi = wasm_f32x4_mul(sg, wasm_v128_load(twi + hs + pos));
        v128_t br = wasm_v128_load(fre + i1), bi = wasm_v128_load(fim + i1);
        v128_t vr = wasm_f32x4_sub(wasm_f32x4_mul(br, wr), wasm_f32x4_mul(bi, wi));
        v128_t vi = wasm_f32x4_add(wasm_f32x4_mul(br, wi), wasm_f32x4_mul(bi, wr));
        v128_t ur = wasm_v128_load(fre + i0), ui = wasm_v128_load(fim + i0);
        wasm_v128_store(fre + i0, wasm_f32x4_add(ur, vr)); wasm_v128_store(fim + i0, wasm_f32x4_add(ui, vi));
        wasm_v128_store(fre + i1, wasm_f32x4_sub(ur, vr)); wasm_v128_store(fim + i1, wasm_f32x4_sub(ui, vi));
      }
    }
  }
}

/** One bin k of nsf_stft after the FFT (the real-FFT split, then the filter H or the noise gain G). */
static inline void stft_bin(int k, int ch, const f32 *row, f32 *yr, f32 maxMag, const f32 *cs, const f32 *sn) {
  int ka = k % NSF_NH, kb = (NSF_NH - k) % NSF_NH;
  f32 zr = fre[ka], zi = fim[ka];
  f32 cr = fre[kb], ci = -fim[kb];
  f32 fer = (zr + cr) * 0.5f, fei = (zi + ci) * 0.5f;
  f32 dr = (zr - cr) * 0.5f, di = (zi - ci) * 0.5f;
  f32 for_ = di, foi = -dr;
  f32 wr = cs[k], wi = -sn[k];
  f32 xr = fer + (for_ * wr - foi * wi), xi = fei + (for_ * wi + foi * wr);
  if (ch == 0) {
    f32 mg = exp_(row[k]);
    if (mg > maxMag) mg = maxMag;
    f32 ps, pc;
    sincos_(row[NSF_BINS + k], &ps, &pc);
    f32 hr = mg * pc, hi = mg * ps;
    yr[2 * k] = hr * xr - hi * xi;
    yr[2 * k + 1] = hr * xi + hi * xr;
  } else {
    f32 g = exp_(row[2 * NSF_BINS + k]);
    if (g > maxMag) g = maxMag;
    yr[2 * k] += g * xr;
    yr[2 * k + 1] += g * xi;
  }
}

/**
 * Frames t0..t0+nT-1: Y[t - t0] = H[t] STFT(e_h)[t] + G[t] STFT(e_n)[t] (hann 2048, center reflect padding over N samples);
 * head s rows (t - t0 + headOff) * O: [log |H| (1025) | phase H (1025) | log G (1025)], magnitudes capped at maxMag.
 * Y [nT, 1025] complex interleaved. tab: cos (2048) | sin (2048) of 2 pi k / 2048.
 */
EXPORT("nsf_stft") void nsf_stft(const f32 *ex, int N, int t0, int nT, const f32 *s, int O, int headOff, f32 maxMag, f32 *Y, const f32 *tab) {
  const f32 *cs = tab, *sn = tab + NSF_NFFT;
  for (int w = 0; w < nT; w++) {
    int t = w + t0;
    const f32 *row = s + (w + headOff) * O;
    f32 *yr = Y + w * NSF_BINS * 2;
    for (int ch = 0; ch < 2; ch++) {
      for (int m = 0; m < NSF_NH; m++) {
        uint32_t d = brev10((uint32_t)m);
        for (int e = 0; e < 2; e++) {
          int j = 2 * m + e;
          int i = t * NSF_HOP + j - NSF_NH;
          if (i < 0) i = -i;
          if (i >= N) i = 2 * (N - 1) - i;
          (e ? fim : fre)[d] = ex[2 * i + ch] * (0.5f - 0.5f * cs[j]);
        }
      }
      fft1024(-1.0f, tab);
      for (int k = 0; k < 4; k++) stft_bin(k, ch, row, yr, maxMag, cs, sn);
      // bins 4..1023 four at a time: each lane does stft_bin's operations in the same order (exp4 / sincos4 are
      // lane-wise exp_ / sincos_, min = the scalar cap), so Y is bit-identical
      const v128_t half = wasm_f32x4_splat(0.5f), cap = wasm_f32x4_splat(maxMag);
      for (int k = 4; k < NSF_NH; k += 4) {
        v128_t zr = wasm_v128_load(fre + k), zi = wasm_v128_load(fim + k);
        v128_t cr = rev4(wasm_v128_load(fre + NSF_NH - k - 3)), ci = wasm_f32x4_neg(rev4(wasm_v128_load(fim + NSF_NH - k - 3)));
        v128_t fer = wasm_f32x4_mul(wasm_f32x4_add(zr, cr), half), fei = wasm_f32x4_mul(wasm_f32x4_add(zi, ci), half);
        v128_t dr = wasm_f32x4_mul(wasm_f32x4_sub(zr, cr), half), di = wasm_f32x4_mul(wasm_f32x4_sub(zi, ci), half);
        v128_t for_ = di, foi = wasm_f32x4_neg(dr);
        v128_t wr = wasm_v128_load(cs + k), wi = wasm_f32x4_neg(wasm_v128_load(sn + k));
        v128_t xr = wasm_f32x4_add(fer, wasm_f32x4_sub(wasm_f32x4_mul(for_, wr), wasm_f32x4_mul(foi, wi)));
        v128_t xi = wasm_f32x4_add(fei, wasm_f32x4_add(wasm_f32x4_mul(for_, wi), wasm_f32x4_mul(foi, wr)));
        if (ch == 0) {
          v128_t mg = wasm_f32x4_min(exp4(wasm_v128_load(row + k)), cap);
          v128_t ps, pc;
          sincos4(wasm_v128_load(row + NSF_BINS + k), &ps, &pc);
          v128_t hr = wasm_f32x4_mul(mg, pc), hi = wasm_f32x4_mul(mg, ps);
          v128_t yre = wasm_f32x4_sub(wasm_f32x4_mul(hr, xr), wasm_f32x4_mul(hi, xi));
          v128_t yim = wasm_f32x4_add(wasm_f32x4_mul(hr, xi), wasm_f32x4_mul(hi, xr));
          wasm_v128_store(yr + 2 * k, zip_lo(yre, yim));
          wasm_v128_store(yr + 2 * k + 4, zip_hi(yre, yim));
        } else {
          v128_t g = wasm_f32x4_min(exp4(wasm_v128_load(row + 2 * NSF_BINS + k)), cap);
          v128_t gr = wasm_f32x4_mul(g, xr), gi = wasm_f32x4_mul(g, xi);
          wasm_v128_store(yr + 2 * k, wasm_f32x4_add(wasm_v128_load(yr + 2 * k), zip_lo(gr, gi)));
          wasm_v128_store(yr + 2 * k + 4, wasm_f32x4_add(wasm_v128_load(yr + 2 * k + 4), zip_hi(gr, gi)));
        }
      }
      stft_bin(NSF_NH, ch, row, yr, maxMag, cs, sn);
    }
  }
}

/** Frames t0..t0+nT-1: frames[t - fbase] = hann * irfft_2048(Y[t - t0]) (ISTFT; frames [., 2048]). */
EXPORT("nsf_istft") void nsf_istft(const f32 *Y, int t0, int nT, f32 *frames, int fbase, const f32 *tab) {
  const f32 *cs = tab, *sn = tab + NSF_NFFT;
  for (int w = 0; w < nT; w++) {
    int t = w + t0;
    const f32 *yr = Y + w * NSF_BINS * 2;
    for (int k = 0; k < NSF_NH; k++) {
      f32 xr = yr[2 * k], xi = yr[2 * k + 1];
      f32 cr = yr[2 * (NSF_NH - k)], ci = yr[2 * (NSF_NH - k) + 1];
      if (k == 0) { xi = 0.0f; ci = 0.0f; }
      ci = -ci;
      f32 er = (xr + cr) * 0.5f, ei = (xi + ci) * 0.5f;
      f32 dr = (xr - cr) * 0.5f, di = (xi - ci) * 0.5f;
      f32 wr = cs[k], wi = sn[k];
      f32 or_ = dr * wr - di * wi, oi = dr * wi + di * wr;
      uint32_t d = brev10((uint32_t)k);
      fre[d] = er - oi;
      fim[d] = ei + or_;
    }
    fft1024(1.0f, tab);
    f32 *fr = frames + (t - fbase) * NSF_NFFT;
    for (int m = 0; m < NSF_NH; m++) {
      fr[2 * m] = fre[m] / (f32)NSF_NH * (0.5f - 0.5f * cs[2 * m]);
      fr[2 * m + 1] = fim[m] / (f32)NSF_NH * (0.5f - 0.5f * cs[2 * m + 1]);
    }
  }
}

/**
 * y[o - o0] for o = o0..o0+count-1: sum_t frames[t, m - 512 t] / sum_t hann(m - 512 t)^2 (t < T), m = o + 1024 (OLA);
 * frames holds the frames from fbase on (the caller keeps every frame these samples need).
 */
EXPORT("nsf_ola") void nsf_ola(const f32 *frames, int fbase, int T, int o0, int count, f32 *y, const f32 *tab) {
  const f32 *cs = tab;
  for (int i = 0; i < count; i++) {
    int o = i + o0;
    int n = o + NSF_NFFT / 2;
    int last = n / NSF_HOP;
    if (last > T - 1) last = T - 1;
    f32 acc = 0.0f, env = 0.0f;
    for (int t = n >= NSF_NFFT ? (n - NSF_NFFT) / NSF_HOP + 1 : 0; t <= last; t++) {
      int k = n - t * NSF_HOP;
      f32 w = 0.5f - 0.5f * cs[k];
      acc += frames[(t - fbase) * NSF_NFFT + k];
      env += w * w;
    }
    y[i] = env > 1e-11f ? acc / env : 0.0f;
  }
}

/** y[f] = [h[idx[f]] (D) | extra[f] (4)] (the WGSL FRAME_IN). */
EXPORT("frame_in") void frame_in(const uint32_t *idx, int F, int D, const f32 *h, const f32 *extra, f32 *y) {
  for (int f = 0; f < F; f++) {
    const f32 *s = h + idx[f] * D;
    f32 *d = y + f * (D + 4);
    for (int c = 0; c < D; c++) d[c] = s[c];
    for (int c = 0; c < 4; c++) d[D + c] = extra[f * 4 + c];
  }
}
