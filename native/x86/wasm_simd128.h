// The subset of <wasm_simd128.h> used by native/tts-cpu.c, on x86 SSE4.1, for a native build of the same kernels
// (bench and comparison only; the shipped build is wasm). Every operation is the same IEEE-754 single operation lane-wise
// (no FMA contraction: build with -ffp-contract=off), so the native output is bit-identical to the wasm output.
#pragma once
#include <immintrin.h>
#include <stdint.h>
typedef __m128 v128_t;
#define V(x) ((__m128)(x))
#define I(x) ((__m128i)(x))
static inline v128_t wasm_f32x4_splat(float a) { return _mm_set1_ps(a); }
static inline v128_t wasm_i32x4_splat(int32_t a) { return V(_mm_set1_epi32(a)); }
static inline v128_t wasm_f32x4_add(v128_t a, v128_t b) { return _mm_add_ps(a, b); }
static inline v128_t wasm_f32x4_sub(v128_t a, v128_t b) { return _mm_sub_ps(a, b); }
static inline v128_t wasm_f32x4_mul(v128_t a, v128_t b) { return _mm_mul_ps(a, b); }
static inline v128_t wasm_f32x4_div(v128_t a, v128_t b) { return _mm_div_ps(a, b); }
// wasm f32x4.min / max order -0 < +0 (x86 min/max return the second operand on equal inputs): equal lanes take the AND
// (max) or OR (min) of the two bit patterns, which is the same value for equal non-zero inputs and +0 / -0 for mixed
// zeros. NaN is never passed by the kernels.
static inline v128_t wasm_f32x4_min(v128_t a, v128_t b) { __m128 e = _mm_cmpeq_ps(a, b); return _mm_blendv_ps(_mm_min_ps(a, b), _mm_or_ps(a, b), e); }
static inline v128_t wasm_f32x4_max(v128_t a, v128_t b) { __m128 e = _mm_cmpeq_ps(a, b); return _mm_blendv_ps(_mm_max_ps(a, b), _mm_and_ps(a, b), e); }
static inline v128_t wasm_f32x4_abs(v128_t a) { return _mm_andnot_ps(_mm_set1_ps(-0.0f), a); }
static inline v128_t wasm_f32x4_neg(v128_t a) { return _mm_xor_ps(_mm_set1_ps(-0.0f), a); }
static inline v128_t wasm_f32x4_floor(v128_t a) { return _mm_round_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC); }
static inline v128_t wasm_f32x4_ge(v128_t a, v128_t b) { return _mm_cmpge_ps(a, b); }
static inline v128_t wasm_v128_bitselect(v128_t a, v128_t b, v128_t m) { return _mm_or_ps(_mm_and_ps(m, a), _mm_andnot_ps(m, b)); }
static inline v128_t wasm_v128_load(const void *p) { return _mm_loadu_ps((const float *)p); }
static inline void wasm_v128_store(void *p, v128_t a) { _mm_storeu_ps((float *)p, a); }
static inline v128_t wasm_v128_load32_splat(const void *p) { return _mm_set1_ps(*(const float *)p); }
static inline v128_t wasm_v128_load32_zero(const void *p) { return V(_mm_cvtsi32_si128(*(const int32_t *)p)); }
static inline float wasm_f32x4_extract_lane_(v128_t a, int l) { float t[4]; _mm_storeu_ps(t, a); return t[l]; }
#define wasm_f32x4_extract_lane(a, l) wasm_f32x4_extract_lane_(a, l)
static inline v128_t wasm_f32x4_make(float a, float b, float c, float d) { return _mm_setr_ps(a, b, c, d); }
static inline v128_t wasm_f32x4_convert_i32x4(v128_t a) { return _mm_cvtepi32_ps(I(a)); }
static inline v128_t wasm_i16x8_extend_low_i8x16(v128_t a) { return V(_mm_cvtepi8_epi16(I(a))); }
static inline v128_t wasm_i32x4_extend_low_i16x8(v128_t a) { return V(_mm_cvtepi16_epi32(I(a))); }
static inline v128_t wasm_i32x4_add(v128_t a, v128_t b) { return V(_mm_add_epi32(I(a), I(b))); }
static inline v128_t wasm_i32x4_shl(v128_t a, int n) { return V(_mm_slli_epi32(I(a), n)); }
// trunc_sat: n is an integer-valued float within +-128 here, so cvtt equals the saturating truncation
static inline v128_t wasm_i32x4_trunc_sat_f32x4(v128_t a) { return V(_mm_cvttps_epi32(a)); }
static inline v128_t wasm_f32x4_lt(v128_t a, v128_t b) { return _mm_cmplt_ps(a, b); }
static inline v128_t wasm_f32x4_gt(v128_t a, v128_t b) { return _mm_cmpgt_ps(a, b); }
static inline v128_t wasm_f32x4_sqrt(v128_t a) { return _mm_sqrt_ps(a); }
static inline v128_t wasm_v128_and(v128_t a, v128_t b) { return _mm_and_ps(a, b); }
static inline v128_t wasm_v128_or(v128_t a, v128_t b) { return _mm_or_ps(a, b); }
static inline v128_t wasm_i32x4_sub(v128_t a, v128_t b) { return V(_mm_sub_epi32(I(a), I(b))); }
static inline v128_t wasm_i32x4_eq(v128_t a, v128_t b) { return V(_mm_cmpeq_epi32(I(a), I(b))); }
static inline v128_t wasm_u32x4_shr(v128_t a, int n) { return V(_mm_srli_epi32(I(a), n)); }
static inline v128_t zip_lo(v128_t a, v128_t b) { return _mm_unpacklo_ps(a, b); }
static inline v128_t zip_hi(v128_t a, v128_t b) { return _mm_unpackhi_ps(a, b); }
static inline v128_t rev4(v128_t a) { return _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3)); }
