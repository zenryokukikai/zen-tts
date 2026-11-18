// The subset of <wasm_simd128.h> used by native/tts-cpu.c, on AArch64 NEON, for a native arm64 build of the same kernels
// (the browser uses the wasm build). Every operation is the same IEEE-754 single operation lane-wise (no FMA contraction:
// build with -ffp-contract=off; AArch64 has vector divide, sqrt and round-to-minus-infinity, and FMIN / FMAX order -0 < +0
// like wasm f32x4.min / max), so the native output is bit-identical to the wasm output.
#pragma once
#include <arm_neon.h>
#include <stdint.h>
typedef float32x4_t v128_t;
#define AF_(x) vreinterpretq_f32_u32(x)
#define AFI_(x) vreinterpretq_f32_s32(x)
#define AU_(x) vreinterpretq_u32_f32(x)
#define AS_(x) vreinterpretq_s32_f32(x)
static inline v128_t wasm_f32x4_splat(float a) { return vdupq_n_f32(a); }
static inline v128_t wasm_i32x4_splat(int32_t a) { return AFI_(vdupq_n_s32(a)); }
static inline v128_t wasm_f32x4_add(v128_t a, v128_t b) { return vaddq_f32(a, b); }
static inline v128_t wasm_f32x4_sub(v128_t a, v128_t b) { return vsubq_f32(a, b); }
static inline v128_t wasm_f32x4_mul(v128_t a, v128_t b) { return vmulq_f32(a, b); }
static inline v128_t wasm_f32x4_div(v128_t a, v128_t b) { return vdivq_f32(a, b); }
static inline v128_t wasm_f32x4_min(v128_t a, v128_t b) { return vminq_f32(a, b); }
static inline v128_t wasm_f32x4_max(v128_t a, v128_t b) { return vmaxq_f32(a, b); }
static inline v128_t wasm_f32x4_abs(v128_t a) { return vabsq_f32(a); }
static inline v128_t wasm_f32x4_neg(v128_t a) { return vnegq_f32(a); }
static inline v128_t wasm_f32x4_floor(v128_t a) { return vrndmq_f32(a); }
static inline v128_t wasm_f32x4_sqrt(v128_t a) { return vsqrtq_f32(a); }
static inline v128_t wasm_f32x4_ge(v128_t a, v128_t b) { return AF_(vcgeq_f32(a, b)); }
static inline v128_t wasm_f32x4_gt(v128_t a, v128_t b) { return AF_(vcgtq_f32(a, b)); }
static inline v128_t wasm_f32x4_lt(v128_t a, v128_t b) { return AF_(vcltq_f32(a, b)); }
static inline v128_t wasm_v128_bitselect(v128_t a, v128_t b, v128_t m) { return vbslq_f32(AU_(m), a, b); }
static inline v128_t wasm_v128_and(v128_t a, v128_t b) { return AF_(vandq_u32(AU_(a), AU_(b))); }
static inline v128_t wasm_v128_or(v128_t a, v128_t b) { return AF_(vorrq_u32(AU_(a), AU_(b))); }
static inline v128_t wasm_v128_load(const void *p) { return vld1q_f32((const float *)p); }
static inline void wasm_v128_store(void *p, v128_t a) { vst1q_f32((float *)p, a); }
static inline v128_t wasm_v128_load32_splat(const void *p) { return vld1q_dup_f32((const float *)p); }
static inline v128_t wasm_v128_load32_zero(const void *p) { uint32_t u; __builtin_memcpy(&u, p, 4); return AF_(vsetq_lane_u32(u, vdupq_n_u32(0), 0)); }
#define wasm_f32x4_extract_lane(a, l) vgetq_lane_f32((a), (l))
static inline v128_t wasm_f32x4_make(float a, float b, float c, float d) { const float t[4] = {a, b, c, d}; return vld1q_f32(t); }
static inline v128_t wasm_f32x4_convert_i32x4(v128_t a) { return vcvtq_f32_s32(AS_(a)); }
static inline v128_t wasm_i16x8_extend_low_i8x16(v128_t a) { return AFI_(vreinterpretq_s32_s16(vmovl_s8(vget_low_s8(vreinterpretq_s8_f32(a))))); }
static inline v128_t wasm_i32x4_extend_low_i16x8(v128_t a) { return AFI_(vmovl_s16(vget_low_s16(vreinterpretq_s16_f32(a)))); }
static inline v128_t wasm_i32x4_add(v128_t a, v128_t b) { return AFI_(vaddq_s32(AS_(a), AS_(b))); }
static inline v128_t wasm_i32x4_sub(v128_t a, v128_t b) { return AFI_(vsubq_s32(AS_(a), AS_(b))); }
static inline v128_t wasm_i32x4_eq(v128_t a, v128_t b) { return AF_(vceqq_s32(AS_(a), AS_(b))); }
static inline v128_t wasm_i32x4_shl(v128_t a, int n) { return AFI_(vshlq_s32(AS_(a), vdupq_n_s32(n))); }
static inline v128_t wasm_u32x4_shr(v128_t a, int n) { return AF_(vshlq_u32(AU_(a), vdupq_n_s32(-n))); }
// wasm i32x4.trunc_sat_f32x4: truncation toward zero, saturating (FCVTZS does the same; NaN -> 0 in both)
static inline v128_t wasm_i32x4_trunc_sat_f32x4(v128_t a) { return AFI_(vcvtq_s32_f32(a)); }
static inline v128_t zip_lo(v128_t a, v128_t b) { return vzip1q_f32(a, b); }
static inline v128_t zip_hi(v128_t a, v128_t b) { return vzip2q_f32(a, b); }
static inline v128_t rev4(v128_t a) { v128_t r = vrev64q_f32(a); return vextq_f32(r, r, 2); }
