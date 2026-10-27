// Shared WGSL building blocks (linear layers with fp16 / int8 weights dequantized in the shader, im2col, depthwise conv +
// LayerNorm) and uniform packing.

export type TtsWeightFormat = 'fp16' | 'int8';

export const STORAGE = 0x80;
export const UNIFORM = 0x40;
export const COPY_DST = 0x08;
export const COPY_SRC = 0x04;
export const MAP_READ = 0x01;
export const WG = 64;
/** Rows of x one linear workgroup multiplies with the same 4 weight columns. */
export const ROWS = 8;

export const GELU = `
fn erf1(v: f32) -> f32 {
  // Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7)
  let a = abs(v);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return select(-y, y, v >= 0.0);
}
fn gelu(v: f32) -> f32 { return 0.5 * v * (1.0 + erf1(v * 0.70710678118654752)); }`;

/** Weight read of the linear shader: 4 consecutive outputs o4 of input row i, before the column scale. */
export const WEIGHT_READ: Record<TtsWeightFormat, string> = {
  fp16: `fn wrow(i: u32, o4: u32) -> vec4<f32> {
  let k = (i * p.O4 + o4) * 2u;
  return vec4<f32>(unpack2x16float(w[k]), unpack2x16float(w[k + 1u]));
}`,
  int8: `fn wrow(i: u32, o4: u32) -> vec4<f32> {
  let u = bitcast<i32>(w[i * p.O4 + o4]);
  return vec4<f32>(f32((u << 24u) >> 24u), f32((u << 16u) >> 24u), f32((u << 8u) >> 24u), f32(u >> 24u));
}`,
};

/**
 * y[t, o] (+)= act(scale[o] * sum_i x[t, i] w[i, o] + bias[o]); sb = [scale (O4 vec4) | bias (O4 vec4)]; act 1 GELU, 2 ReLU.
 * One workgroup = 4 outputs x ROWS rows; its 64 threads split i and add their sums in workgroup memory.
 */
export const linear = (f: TtsWeightFormat): string => `${GELU}
struct P { T: u32, I: u32, O4: u32, act: u32, res: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> sb: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<vec4<f32>>;
var<workgroup> part: array<array<vec4<f32>, ${ROWS}>, ${WG}>;
${WEIGHT_READ[f]}
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let n = wg.x + wg.y * 65535u;
  let o4 = n % p.O4;
  let t0 = (n / p.O4) * ${ROWS}u;
  let valid = t0 < p.T;
  let rows = select(0u, min(${ROWS}u, p.T - t0), valid);
  var acc: array<vec4<f32>, ${ROWS}>;
  for (var i = li; i < p.I; i += ${WG}u) {
    let wv = wrow(i, o4);
    for (var r = 0u; r < rows; r++) { acc[r] += x[(t0 + r) * p.I + i] * wv; }
  }
  part[li] = acc;
  workgroupBarrier();
  for (var k = ${WG / 2}u; k > 0u; k >>= 1u) {
    if (li < k) { for (var r = 0u; r < ${ROWS}u; r++) { part[li][r] += part[li + k][r]; } }
    workgroupBarrier();
  }
  if (li >= rows) { return; }
  var v = part[0][li] * sb[o4] + sb[p.O4 + o4];
  if (p.act == 1u) { v = vec4<f32>(gelu(v.x), gelu(v.y), gelu(v.z), gelu(v.w)); }
  if (p.act == 2u) { v = max(v, vec4<f32>(0.0)); }
  let k = (t0 + li) * p.O4 + o4;
  if (p.res == 1u) { y[k] += v; } else { y[k] = v; }
}`;

/** Embed conv as a matrix product: col[t, k * C + c] = x[t + k - K/2, c] (0 outside 0..T-1). */
export const IM2COL = `
struct P { T: u32, C: u32, K: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> col: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = g.x + g.y * 65535u * ${WG}u;
  let W = p.K * p.C;
  if (n >= p.T * W) { return; }
  let t = i32(n / W);
  let k = (n % W) / p.C;
  let c = n % p.C;
  let tt = t + i32(k) - i32(p.K / 2u);
  col[n] = select(0.0, x[u32(max(tt, 0)) * p.C + c], tt >= 0 && tt < i32(p.T));
}`;

/**
 * One workgroup per frame t: y[t] = LayerNorm(conv ? dwconv(x)[t] : x[t]).
 * prm = [dw weight (D * K) | dw bias (D) |] ln weight (D) | ln bias (D) (the conv part only when conv == 1).
 */
export const DW_NORM = `
struct P { T: u32, D: u32, K: u32, conv: u32, eps: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> prm: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, ${WG}>;
fn total(v: f32, li: u32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var k = ${WG / 2}u; k > 0u; k >>= 1u) {
    if (li < k) { red[li] += red[li + k]; }
    workgroupBarrier();
  }
  let s = red[0];
  workgroupBarrier();
  return s;
}
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wg.x;
  let D = p.D;
  let ln = select(0u, D * p.K + D, p.conv == 1u);
  var s = 0.0;
  for (var c = li; c < D; c += ${WG}u) {
    var v = 0.0;
    if (p.conv == 1u) {
      v = prm[D * p.K + c];
      for (var k = 0u; k < p.K; k++) {
        let tt = i32(t) + i32(k) - i32(p.K / 2u);
        if (tt >= 0 && tt < i32(p.T)) { v += prm[c * p.K + k] * x[u32(tt) * D + c]; }
      }
    } else { v = x[t * D + c]; }
    y[t * D + c] = v;
    s += v;
  }
  let mean = total(s, li) / f32(D);
  var q = 0.0;
  for (var c = li; c < D; c += ${WG}u) { let d = y[t * D + c] - mean; q += d * d; }
  let r = inverseSqrt(total(q, li) / f32(D) + p.eps);
  for (var c = li; c < D; c += ${WG}u) { y[t * D + c] = (y[t * D + c] - mean) * r * prm[ln + c] + prm[ln + D + c]; }
}`;

/** float -> half bits (round to nearest even). */
function toHalf(v: number): number {
  const f = new Float32Array([v]);
  const x = new Uint32Array(f.buffer)[0]!;
  const sign = (x >>> 16) & 0x8000;
  const e = ((x >>> 23) & 0xff) - 127 + 15;
  let m = x & 0x7fffff;
  if (e <= 0) {
    if (e < -10) return sign;
    m |= 0x800000;
    const s = 14 - e;
    return sign | ((m + (1 << (s - 1))) >>> s);
  }
  if (e >= 31) return sign | 0x7c00;
  const h = sign | (e << 10) | (m >>> 13);
  return (m & 0x1000) ? h + 1 : h;
}

/** A linear weight [I, O] (O padded to a multiple of 4) in the shader's format, and its per-column scale. */
export function packLinear(w: Float32Array, I: number, O: number, f: TtsWeightFormat): { data: Uint32Array; scale: Float32Array; O4: number } {
  const O4 = Math.ceil(O / 4);
  const scale = new Float32Array(O4 * 4).fill(1);
  if (f === 'fp16') {
    const h = new Uint16Array(I * O4 * 4);
    for (let i = 0; i < I; i++) for (let o = 0; o < O; o++) h[i * O4 * 4 + o] = toHalf(w[i * O + o]!);
    return { data: new Uint32Array(h.buffer), scale, O4 };
  }
  for (let o = 0; o < O; o++) {
    let m = 0;
    for (let i = 0; i < I; i++) m = Math.max(m, Math.abs(w[i * O + o]!));
    scale[o] = m > 0 ? m / 127 : 1;
  }
  const q = new Int8Array(I * O4 * 4);
  for (let i = 0; i < I; i++) for (let o = 0; o < O; o++) q[i * O4 * 4 + o] = Math.round(w[i * O + o]! / scale[o]!);
  return { data: new Uint32Array(q.buffer), scale, O4 };
}

/** u32 fields (and f32 ones where given as {f: value}) of a uniform struct, padded to 16 bytes. */
export function uniformData(fields: (number | { f: number })[]): ArrayBuffer {
  const n = Math.ceil(fields.length / 4) * 4;
  const b = new ArrayBuffer(n * 4);
  const u = new Uint32Array(b);
  const f = new Float32Array(b);
  fields.forEach((v, i) => { if (typeof v === 'number') u[i] = v; else f[i] = v.f; });
  return b;
}
