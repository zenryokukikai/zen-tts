// Attention implementation referencing VITS (MIT License). See THIRD_PARTY_NOTICES.md.
import { WG } from './tts-wgsl.js';

/** Max phones of one chunk (the attention keeps a row of scores in workgroup memory). */
const MAX_PHONES = 512;

const SCORE_ROW = `
var<workgroup> sc: array<f32, ${MAX_PHONES}>;
var<workgroup> red: array<f32, ${WG}>;
fn maxAll(v: f32, li: u32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var k = ${WG / 2}u; k > 0u; k >>= 1u) { if (li < k) { red[li] = max(red[li], red[li + k]); } workgroupBarrier(); }
  let s = red[0];
  workgroupBarrier();
  return s;
}
fn sumAll(v: f32, li: u32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var k = ${WG / 2}u; k > 0u; k >>= 1u) { if (li < k) { red[li] += red[li + k]; } workgroupBarrier(); }
  let s = red[0];
  workgroupBarrier();
  return s;
}`;

/**
 * VITS relative attention (window W, heads share relk/relv [2W + 1, DH]), one workgroup per (head h, query i):
 * s = (q_i.k_j + [|j - i| <= W] q_i.relk[j - i + W]) / sqrt(DH); o = sum_j p_j (v_j + [|j - i| <= W] relv[j - i + W]).
 */
export const VITS_ATTN = `
struct P { T: u32, H: u32, DH: u32, W: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> relk: array<f32>;
@group(0) @binding(3) var<storage, read> relv: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
${SCORE_ROW}
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let h = wg.x / p.T;
  let i = wg.x % p.T;
  let D = p.H * p.DH;
  let qo = i * 3u * D + h * p.DH;
  let r = inverseSqrt(f32(p.DH));
  var m = -3.4e38;
  for (var j = li; j < p.T; j += ${WG}u) {
    let ko = j * 3u * D + D + h * p.DH;
    let d = i32(j) - i32(i);
    let near = abs(d) <= i32(p.W);
    let ro = u32(clamp(d + i32(p.W), 0, 2 * i32(p.W))) * p.DH;
    var s = 0.0;
    for (var c = 0u; c < p.DH; c++) { s += qkv[qo + c] * (qkv[ko + c] + select(0.0, relk[ro + c], near)); }
    s *= r;
    sc[j] = s;
    m = max(m, s);
  }
  m = maxAll(m, li);
  var su = 0.0;
  for (var j = li; j < p.T; j += ${WG}u) { let e = exp(sc[j] - m); sc[j] = e; su += e; }
  su = sumAll(su, li);
  for (var c = li; c < p.DH; c += ${WG}u) {
    var acc = 0.0;
    for (var j = 0u; j < p.T; j++) { acc += sc[j] * qkv[j * 3u * D + 2u * D + h * p.DH + c]; }
    let lo = u32(max(i32(i) - i32(p.W), 0));
    let hi = min(i + p.W, p.T - 1u);
    for (var j = lo; j <= hi; j++) { acc += sc[j] * relv[(j + p.W - i) * p.DH + c]; }
    o[i * D + h * p.DH + c] = acc / su;
  }
}`;

