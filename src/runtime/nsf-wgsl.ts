import { modelFetch } from './model-fetch.js';
// NSF (neural source-filter) decoder in hand-written WGSL: z [T, 192] -> F0 (f0z) -> 44.1 kHz audio.
// Linear weights fp16 (dequantized in the shader), everything else float32. GPU buffers are made once for up to maxT frames.
//
// Math (T frames, N = T * 512 samples):
//   f0z:   x = LN(conv1d_k7(z)); 4x ConvNeXt (192, 576); [lf, vl] = Linear(x); f0 = vl > 0 ? exp(lf * S + M) : 0
//   trunk: x = LN(conv1d_k7([z | (ln max(f0, 1) - M) / S * v | v])); 8x ConvNeXt (384, 1152); s = head(LN(x)) [T, 3075]
//          (M, S = the voice's log-F0 mean / scale from nsf.json: nsfF0Norm; 5.43 / 0.25 for 「全力 マリン」)
//          H = min(exp(s[:1025]), 100) e^{i s[1025:2050]}, G = min(exp(s[2050:]), 100)
//   excitation (per sample): f0 upsampled (log F0 and voicing linear between frame centres), cyc = cumsum(f0s / SR)
//          (per-frame f32 scans + a fract prefix over frames),
//          e_h = 0.1 (sin((K + 1/2) x) / (2 sin(x / 2)) - 1/2) [f0s > 0] clamp(vs, 0, 1), x = 2 pi (cyc - round(cyc)),
//          K = clamp(floor(0.45 SR / f0s), 1, 320); e_n = N(0, 1) (PCG + Box-Muller, or given)
//   filter: Y = H STFT(e_h) + G STFT(e_n) (hann 2048, hop 512, center reflect padding, first T frames)
//   output: iSTFT(Y) as tts-wgsl.ts (overlap-add / sum hann^2, centre trim, length N)
import { DW_NORM, IM2COL, ROWS, STORAGE, UNIFORM, COPY_DST, COPY_SRC, MAP_READ, WG, linear, packLinear, uniformData } from './tts-wgsl.js';
import { DEFAULT_NSF_F0_NORM, F0Z, NSF, TRUNK, applyOutputLevel, type Net, nsfExportToWeights, nsfF0Norm, nsfOutputGain, type NsfExportMeta, type NsfF0Norm, type NsfWeights } from './nsf-weights.js';

export { NSF, nsfExportToWeights, nsfF0Norm, nsfOutputGain, type NsfExportMeta, type NsfF0Norm, type NsfWeights };

/** A WGSL f32 literal of a JS number (always with a decimal point or exponent). */
const f32Lit = (x: number): string => { const t = String(x); return /[.e]/.test(t) ? t : `${t}.0`; };

/* eslint-disable @typescript-eslint/no-explicit-any */
type GpuDevice = any;
type GpuBuffer = any;
type GpuBindGroup = any;
type GpuPipeline = any;

const FFT_WG = 256;
const NH = NSF.nFft / 2;
const LOG2_NH = Math.log2(NH);

/** f0z out [T, 4] (lf, vl, -, -) -> f0 [T] Hz (0 = unvoiced). */
const F0_OUT = (n: NsfF0Norm): string => `
struct P { T: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> o: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> f0: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x;
  if (t >= p.T) { return; }
  let v = o[t];
  f0[t] = select(0.0, exp(v.x * ${f32Lit(n.logF0Std)} + ${f32Lit(n.logF0Mean)}), v.y > 0.0);
}`;

/** [z | normalised log F0 | voiced] [T, 194]. */
const TRUNK_IN = (n: NsfF0Norm): string => `
struct P { T: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> z: array<f32>;
@group(0) @binding(2) var<storage, read> f0: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
const C = ${NSF.zDim + 2}u;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = g.x + g.y * 65535u * ${WG}u;
  if (n >= p.T * C) { return; }
  let t = n / C;
  let c = n % C;
  let f = f0[t];
  let v = select(0.0, 1.0, f > 0.0);
  var r = v;
  if (c < ${NSF.zDim}u) { r = z[t * ${NSF.zDim}u + c]; }
  else if (c == ${NSF.zDim}u) { r = (log(max(f, 1.0)) - ${f32Lit(n.logF0Mean)}) / ${f32Lit(n.logF0Std)} * v; }
  x[n] = r;
}`;

/** One workgroup per frame f (512 samples): sample F0 / voicing (up[n] = (f0s, vs)), the frame's inclusive scan of f0s / SR (loc) and its total. */
const EXC_SCAN = `
struct P { T: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> f0: array<f32>;
@group(0) @binding(2) var<storage, read_write> up: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> loc: array<f32>;
@group(0) @binding(4) var<storage, read_write> tot: array<f32>;
const HOP = ${NSF.hop}u;
var<workgroup> sc: array<f32, ${NSF.hop}>;
@compute @workgroup_size(${NSF.hop / 2}) fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let f = wg.x;
  let i1 = min(f + 1u, p.T - 1u);
  let a = f0[f];
  let b = f0[i1];
  let v0 = select(0.0, 1.0, a > 0.0);
  let v1 = select(0.0, 1.0, b > 0.0);
  let l0 = log(max(a, 1.0)) * v0;
  let l1 = log(max(b, 1.0)) * v1;
  for (var k = 0u; k < 2u; k++) {
    let j = li + k * (HOP / 2u);
    let w = f32(j) / f32(HOP);
    let vs = v0 * (1.0 - w) + v1 * w;
    let ls = (l0 * (1.0 - w) + l1 * w) / max(vs, 1e-3);
    let fs = select(0.0, exp(ls), vs > 0.5);
    up[f * HOP + j] = vec2<f32>(fs, vs);
    sc[j] = fs / ${NSF.sampleRate}.0;
  }
  workgroupBarrier();
  for (var off = 1u; off < HOP; off <<= 1u) {
    let j0 = li;
    let j1 = li + HOP / 2u;
    let s0 = sc[j0] + select(0.0, sc[max(j0, off) - off], j0 >= off);
    let s1 = sc[j1] + select(0.0, sc[max(j1, off) - off], j1 >= off);
    workgroupBarrier();
    sc[j0] = s0;
    sc[j1] = s1;
    workgroupBarrier();
  }
  loc[f * HOP + li] = sc[li];
  loc[f * HOP + li + HOP / 2u] = sc[li + HOP / 2u];
  if (li == 0u) { tot[f] = sc[HOP - 1u]; }
}`;

/** One thread: off[f] = fract(sum of the earlier frames' totals) (the phase in cycles before frame f). */
const EXC_PREFIX = `
struct P { T: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> tot: array<f32>;
@group(0) @binding(2) var<storage, read_write> off: array<f32>;
@compute @workgroup_size(1) fn main() {
  var acc = 0.0;
  for (var f = 0u; f < p.T; f++) { off[f] = acc; acc = fract(acc + fract(tot[f])); }
}`;

/** Per sample: ex[n] = (e_h, e_n). noise 1: e_n from PCG + Box-Muller (seed); 0: ex[n].y is kept (written by the caller). */
const EXC_SOURCE = `
struct P { N: u32, noise: u32, seed: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> up: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> loc: array<f32>;
@group(0) @binding(3) var<storage, read> off: array<f32>;
@group(0) @binding(4) var<storage, read_write> ex: array<vec2<f32>>;
const PI2 = 6.283185307179586;
fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn wrap(a: f32) -> f32 { return a - PI2 * floor(a / PI2 + 0.5); }
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = g.x + g.y * 65535u * ${WG}u;
  if (n >= p.N) { return; }
  let u = up[n];
  let c = off[n / ${NSF.hop}u] + loc[n];
  let x = PI2 * (c - floor(c + 0.5));
  let K = clamp(floor(0.45 * ${NSF.sampleRate}.0 / max(u.x, 1.0)), 1.0, 320.0);
  let s = sin(x * 0.5);
  var d = K;
  if (abs(s) >= 1e-4) { d = sin(wrap((K + 0.5) * x)) / (2.0 * s) - 0.5; }
  let eh = 0.1 * d * select(0.0, 1.0, u.x > 0.0) * clamp(u.y, 0.0, 1.0);
  var en = ex[n].y;
  if (p.noise == 1u) {
    let h1 = pcg(n * 2u + p.seed * 0x9e3779b9u);
    let h2 = pcg(n * 2u + 1u + p.seed * 0x9e3779b9u);
    let u1 = (f32(h1 >> 8u) + 1.0) / 16777217.0;
    let u2 = f32(h2 >> 8u) / 16777216.0;
    en = sqrt(-2.0 * log(u1)) * cos(PI2 * u2);
  }
  ex[n] = vec2<f32>(eh, en);
}`;

const FFT_COMMON = `
const NH = ${NH}u;
const PI2 = 6.283185307179586;
var<workgroup> a: array<vec2<f32>, NH>;
fn cmul(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> { return vec2<f32>(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x); }
/** radix-2 in-place FFT of a (loaded bit-reversed); sgn -1 forward, +1 inverse (unscaled). */
fn fft(li: u32, sgn: f32) {
  for (var hs = 1u; hs < NH; hs <<= 1u) {
    for (var j = li; j < NH / 2u; j += ${FFT_WG}u) {
      let pos = j % hs;
      let i0 = (j / hs) * hs * 2u + pos;
      let ang = sgn * PI2 * f32(pos) / f32(hs * 2u);
      let v = cmul(a[i0 + hs], vec2<f32>(cos(ang), sin(ang)));
      let u = a[i0];
      a[i0] = u + v;
      a[i0 + hs] = u - v;
    }
    workgroupBarrier();
  }
}
fn brev(m: u32) -> u32 { return reverseBits(m) >> ${32 - LOG2_NH}u; }`;

/**
 * One workgroup per frame t: Y[t] = H[t] STFT(e_h)[t] + G[t] STFT(e_n)[t] (hann 2048, center reflect padding), each real
 * 2048-point FFT as a 1024-point complex FFT of z[m] = x[2m] + i x[2m + 1].
 */
const STFT_FILTER = `
struct P { T: u32, N: u32, O: u32, maxMag: f32, offset: u32, headOffset: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> ex: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> s: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<vec2<f32>>;
${FFT_COMMON}
fn xs(t: u32, j: u32, c: u32) -> f32 {
  var i = i32(t * ${NSF.hop}u + j) - ${NH};
  if (i < 0) { i = -i; }
  if (i >= i32(p.N)) { i = 2 * (i32(p.N) - 1) - i; }
  let w = 0.5 - 0.5 * cos(PI2 * f32(j) / ${NSF.nFft}.0);
  return ex[u32(i)][c] * w;
}
fn bin(k: u32) -> vec2<f32> {
  let zk = a[k % NH];
  let q = a[(NH - k) % NH];
  let zc = vec2<f32>(q.x, -q.y);
  let fe = (zk + zc) * 0.5;
  let d = (zk - zc) * 0.5;
  let fo = vec2<f32>(d.y, -d.x);
  let ang = -PI2 * f32(k) / ${NSF.nFft}.0;
  return fe + cmul(fo, vec2<f32>(cos(ang), sin(ang)));
}
@compute @workgroup_size(${FFT_WG}) fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wg.x + p.offset;
  let r = (wg.x + p.headOffset) * p.O;
  let yr = t * ${NSF.bins}u;
  for (var c = 0u; c < 2u; c++) {
    for (var m = li; m < NH; m += ${FFT_WG}u) { a[brev(m)] = vec2<f32>(xs(t, 2u * m, c), xs(t, 2u * m + 1u, c)); }
    workgroupBarrier();
    fft(li, -1.0);
    for (var k = li; k <= NH; k += ${FFT_WG}u) {
      let X = bin(k);
      if (c == 0u) {
        let m = min(exp(s[r + k]), p.maxMag);
        let ph = s[r + ${NSF.bins}u + k];
        Y[yr + k] = cmul(m * vec2<f32>(cos(ph), sin(ph)), X);
      } else {
        Y[yr + k] += min(exp(s[r + ${2 * NSF.bins}u + k]), p.maxMag) * X;
      }
    }
    workgroupBarrier();
  }
}`;

/** One workgroup per frame t: frames[t] = hann * irfft_2048(Y[t]) (as tts-wgsl.ts ISTFT_FRAME). */
const ISTFT = `
struct P { T: u32, offset: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Y: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> frames: array<f32>;
${FFT_COMMON}
@compute @workgroup_size(${FFT_WG}) fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = min(wg.x, p.T - 1u) + p.offset;
  let yr = t * ${NSF.bins}u;
  for (var k = li; k < NH; k += ${FFT_WG}u) {
    var x = Y[yr + k];
    var xc = Y[yr + NH - k];
    if (k == 0u) { x.y = 0.0; xc.y = 0.0; }
    xc = vec2<f32>(xc.x, -xc.y);
    let e = (x + xc) * 0.5;
    let ang = PI2 * f32(k) / f32(2u * NH);
    let o = cmul((x - xc) * 0.5, vec2<f32>(cos(ang), sin(ang)));
    a[brev(k)] = e + vec2<f32>(-o.y, o.x);
  }
  workgroupBarrier();
  fft(li, 1.0);
  let N = 2u * NH;
  for (var m = li; m < NH; m += ${FFT_WG}u) {
    let v = a[m] / f32(NH);
    let n0 = 2u * m;
    let w0 = 0.5 - 0.5 * cos(PI2 * f32(n0) / f32(N));
    let w1 = 0.5 - 0.5 * cos(PI2 * f32(n0 + 1u) / f32(N));
    frames[t * N + n0] = v.x * w0;
    frames[t * N + n0 + 1u] = v.y * w1;
  }
}`;

/** y[n] = sum_t frames[t, m - t * HOP] / sum_t hann(m - t * HOP)^2, m = n + F / 2 (as tts-wgsl.ts OVERLAP_ADD). */
const OLA = `
struct P { T: u32, N: u32, offset: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> frames: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
const F = ${NSF.nFft}u;
const HOP = ${NSF.hop}u;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * 65535u * ${WG}u;
  if (i >= p.N) { return; }
  let o = i + p.offset;
  let n = o + F / 2u;
  let last = min(n / HOP, p.T - 1u);
  var acc = 0.0;
  var env = 0.0;
  for (var t = select(0u, (n - F) / HOP + 1u, n >= F); t <= last; t++) {
    let i = n - t * HOP;
    let w = 0.5 - 0.5 * cos(6.283185307179586 * f32(i) / f32(F));
    acc += frames[t * F + i];
    env += w * w;
  }
  y[o] = select(0.0, acc / env, env > 1e-11);
}`;

/** Deterministic random weights of the NSF shape (for the speed bench: weights do not change the time). */
export function randomNsfWeights(seed = 1): NsfWeights {
  let st = seed >>> 0 || 1;
  const rand = (): number => { st ^= st << 13; st >>>= 0; st ^= st >>> 17; st ^= st << 5; st >>>= 0; return st / 4294967296; };
  const fill = (n: number, std: number): Float32Array => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = (rand() * 2 - 1) * std * 1.732; return a; };
  const norm = (d: number): Float32Array => { const a = new Float32Array(2 * d); a.fill(1, 0, d); return a; };
  const w: NsfWeights = {};
  for (const n of [TRUNK, F0Z]) {
    const p = n.prefix;
    w[`${p}embed.w`] = fill(NSF.kernel * n.C * n.D, 1 / Math.sqrt(NSF.kernel * n.C));
    w[`${p}embed.b`] = fill(n.D, 0.02);
    w[`${p}embed.norm`] = norm(n.D);
    for (let l = 0; l < n.layers; l++) {
      const dw = new Float32Array(n.D * NSF.kernel + 3 * n.D);
      dw.set(fill(n.D * NSF.kernel, 1 / Math.sqrt(NSF.kernel)));
      dw.fill(1, n.D * NSF.kernel + n.D, n.D * NSF.kernel + 2 * n.D);
      w[`${p}${l}.dw`] = dw;
      w[`${p}${l}.pw1.w`] = fill(n.D * n.F, 1 / Math.sqrt(n.D));
      w[`${p}${l}.pw1.b`] = fill(n.F, 0.02);
      w[`${p}${l}.pw2.w`] = fill(n.F * n.D, 1 / n.layers / Math.sqrt(n.F));
      w[`${p}${l}.pw2.b`] = fill(n.D, 0.002);
    }
  }
  w['final.norm'] = norm(NSF.dim);
  w['head.w'] = fill(NSF.dim * 3 * NSF.bins, 0.01);
  const hb = new Float32Array(3 * NSF.bins);
  hb.fill(-1, 0, NSF.bins);
  hb.fill(-6, 2 * NSF.bins);
  w['head.b'] = hb;
  w['f0z.out.w'] = fill(NSF.f0zDim * 2, 0.05);
  w['f0z.out.b'] = new Float32Array([0, 1]);
  return w;
}

/** Fetches <base>.json and <base>.bin and converts them. */
export async function loadNsfWeights(base: string): Promise<{ weights: NsfWeights; meta: NsfExportMeta; binBytes: number }> {
  const [mr, br] = await Promise.all([modelFetch(`${base}.json`), modelFetch(`${base}.bin`)]);
  if (!mr.ok || !br.ok) throw new Error(`nsf weights ${base}: ${mr.status} / ${br.status}`);
  const meta = (await mr.json()) as NsfExportMeta;
  const bin = await br.arrayBuffer();
  return { weights: nsfExportToWeights(meta, bin), meta, binBytes: bin.byteLength };
}

/** The parts of a run, in order (the page's per-part times). */
export const NSF_PARTS = ['f0z', 'embed', 'blocks', 'head', 'upsample', 'source', 'filter', 'istft'] as const;
export type NsfPart = typeof NSF_PARTS[number];
export const NSF_PART_LABEL: Record<NsfPart, string> = {
  f0z: 'F0推定(f0z)',
  embed: '入力+embed',
  blocks: 'ConvNeXt 8段',
  head: '出力層',
  upsample: 'F0の上採樣+位相',
  source: 'パルス+雑音',
  filter: 'STFT×2+フィルタ',
  istft: 'iSTFT+重ね合わせ',
};

interface Step { part: NsfPart; pipeline: GpuPipeline; bind: GpuBindGroup; uniform: GpuBuffer; params: () => ArrayBuffer; groups: () => number }

export interface NsfRunOptions {
  /** frame F0 [T] Hz: used instead of f0z (f0z is skipped). */
  f0?: Float32Array;
  /** noise e_n [T * 512]: used instead of the shader's PCG noise. */
  noise?: Float32Array;
}

export class WgslNsfDecoder {
  private readonly pipelines: Record<string, GpuPipeline> = {};
  private readonly buffers: GpuBuffer[] = [];
  private steps: Step[] = [];
  private T = 0;
  private noiseMode = 1;
  private stream: { total: number; start: number; end: number; from: number; sampleStart: number; sampleCount: number } | null = null;
  private f0Cache!: GpuBuffer;
  private zIn!: GpuBuffer;
  private f0!: GpuBuffer;
  private ex!: GpuBuffer;
  private audio!: GpuBuffer;
  private read!: GpuBuffer;
  private f0Read!: GpuBuffer;
  bytes = 0;
  weightBytes = 0;

  private constructor(readonly device: GpuDevice, readonly maxT: number, readonly seed: number, readonly outputGain = 1) {}

  static async create(device: GpuDevice, weights: NsfWeights, maxT: number, seed = 1234, f0Norm: NsfF0Norm = DEFAULT_NSF_F0_NORM, outputGain = 1): Promise<WgslNsfDecoder> {
    const m = new WgslNsfDecoder(device, maxT, seed, outputGain);
    const codes = { linear: linear('fp16'), im2col: IM2COL, dwNorm: DW_NORM, f0Out: F0_OUT(f0Norm), trunkIn: TRUNK_IN(f0Norm), excScan: EXC_SCAN, excPrefix: EXC_PREFIX, excSource: EXC_SOURCE, stft: STFT_FILTER, istft: ISTFT, ola: OLA };
    for (const [name, code] of Object.entries(codes)) {
      m.pipelines[name] = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
    }
    m.build(weights);
    return m;
  }

  static samples(T: number): number { return T * NSF.hop; }

  /** The decoder's output at its voice's level (nsf.json outputGain; 1 = as the weights give it). */
  private level(audio: Float32Array): Float32Array {
    return applyOutputLevel(audio, this.outputGain);
  }

  private buffer(size: number, usage: number): GpuBuffer {
    const s = Math.ceil(size / 16) * 16;
    this.bytes += s;
    const b = this.device.createBuffer({ size: s, usage });
    this.buffers.push(b);
    return b;
  }

  private upload(data: Float32Array | Uint32Array): GpuBuffer {
    const b = this.buffer(data.byteLength, STORAGE | COPY_DST);
    this.weightBytes += data.byteLength;
    this.device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
    return b;
  }

  private step(part: NsfPart, name: string, buffers: GpuBuffer[], params: () => ArrayBuffer, groups: () => number): Step {
    const pipeline = this.pipelines[name];
    const uniform = this.buffer(32, UNIFORM | COPY_DST);
    const bind = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [uniform, ...buffers].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    return { part, pipeline, bind, uniform, params, groups };
  }

  private linearStep(part: NsfPart, x: GpuBuffer, w: Float32Array, bias: Float32Array, I: number, O: number, y: GpuBuffer, act: number, res: number): Step {
    const pk = packLinear(w, I, O, 'fp16');
    const sb = new Float32Array(pk.O4 * 8);
    sb.set(pk.scale);
    sb.set(bias, pk.O4 * 4);
    const wb = this.upload(pk.data);
    const sbb = this.upload(sb);
    return this.step(part, 'linear', [x, wb, sbb, y], () => uniformData([this.T, I, pk.O4, act, res]), () => pk.O4 * Math.ceil(this.T / ROWS));
  }

  /** embed conv + LN + ConvNeXt blocks of one net: input src [T, C] -> x [T, D] (h, a scratch). */
  private convnext(W: NsfWeights, net: Net, src: GpuBuffer, col: GpuBuffer, x: GpuBuffer, h: GpuBuffer, a: GpuBuffer, embedPart: NsfPart, blockPart: NsfPart): Step[] {
    const need = (k: string): Float32Array => { const v = W[net.prefix + k]; if (!v) throw new Error(`nsf weight ${net.prefix}${k} missing`); return v; };
    const T = (): number => this.T;
    const K = NSF.kernel;
    const KC = K * net.C;
    const norm = (part: NsfPart, s: GpuBuffer, prm: Float32Array, d: GpuBuffer, conv: number): Step => {
      const pb = this.upload(prm);
      return this.step(part, 'dwNorm', [s, pb, d], () => uniformData([T(), net.D, K, conv, { f: NSF.lnEps }]), T);
    };
    const s: Step[] = [];
    s.push(this.step(embedPart, 'im2col', [src, col], () => uniformData([T(), net.C, K]), () => Math.ceil(T() * KC / WG)));
    s.push(this.linearStep(embedPart, col, need('embed.w'), need('embed.b'), KC, net.D, h, 0, 0));
    s.push(norm(embedPart, h, need('embed.norm'), x, 0));
    for (let l = 0; l < net.layers; l++) {
      s.push(norm(blockPart, x, need(`${l}.dw`), h, 1));
      s.push(this.linearStep(blockPart, h, need(`${l}.pw1.w`), need(`${l}.pw1.b`), net.D, net.F, a, 1, 0));
      s.push(this.linearStep(blockPart, a, need(`${l}.pw2.w`), need(`${l}.pw2.b`), net.F, net.D, x, 0, 1));
    }
    return s;
  }

  private build(W: NsfWeights): void {
    const maxT = this.maxT;
    const N = WgslNsfDecoder.samples(maxT);
    const D = NSF.dim;
    const O = 3 * NSF.bins;
    const O4 = Math.ceil(O / 4);
    const T = (): number => this.T;
    const n = (): number => WgslNsfDecoder.samples(this.T);
    this.zIn = this.buffer(maxT * NSF.zDim * 4, STORAGE | COPY_DST);
    this.f0 = this.buffer(maxT * 4, STORAGE | COPY_DST | COPY_SRC);
    const tin = this.buffer(maxT * (NSF.zDim + 2) * 4, STORAGE);
    const col = this.buffer(maxT * NSF.kernel * (NSF.zDim + 2) * 4, STORAGE);
    const x = this.buffer(maxT * D * 4, STORAGE);
    const h = this.buffer(maxT * D * 4, STORAGE);
    const a = this.buffer(maxT * Math.max(NSF.ffDim, O4 * 4) * 4, STORAGE);
    const up = this.buffer(N * 8, STORAGE);
    const loc = this.buffer(N * 4, STORAGE);
    const tot = this.buffer(maxT * 4, STORAGE);
    const off = this.buffer(maxT * 4, STORAGE);
    this.ex = this.buffer(N * 8, STORAGE | COPY_DST);
    const Y = this.buffer(maxT * NSF.bins * 8, STORAGE);
    const frames = this.buffer(maxT * NSF.nFft * 4, STORAGE);
    this.audio = this.buffer(N * 4, STORAGE | COPY_SRC);
    this.read = this.buffer(N * 4, MAP_READ | COPY_DST);
    this.f0Read = this.buffer(maxT * 4, MAP_READ | COPY_DST);
    this.f0Cache = this.buffer(maxT * 4, STORAGE | COPY_SRC | COPY_DST);
    const s: Step[] = [];
    // f0z: z -> f0
    s.push(...this.convnext(W, F0Z, this.zIn, col, x, h, a, 'f0z', 'f0z'));
    s.push(this.linearStep('f0z', x, W['f0z.out.w']!, W['f0z.out.b']!, NSF.f0zDim, 2, a, 0, 0));
    s.push(this.step('f0z', 'f0Out', [a, this.f0], () => uniformData([T()]), () => Math.ceil(T() / WG)));
    // trunk
    s.push(this.step('embed', 'trunkIn', [this.zIn, this.f0, tin], () => uniformData([T()]), () => Math.ceil(T() * (NSF.zDim + 2) / WG)));
    s.push(...this.convnext(W, TRUNK, tin, col, x, h, a, 'embed', 'blocks'));
    const fn = this.upload(W['final.norm']!);
    s.push(this.step('head', 'dwNorm', [x, fn, h], () => uniformData([T(), D, NSF.kernel, 0, { f: NSF.lnEps }]), T));
    s.push(this.linearStep('head', h, W['head.w']!, W['head.b']!, D, O, a, 0, 0));
    // excitation
    s.push(this.step('upsample', 'excScan', [this.f0, up, loc, tot], () => uniformData([T()]), T));
    s.push(this.step('upsample', 'excPrefix', [tot, off], () => uniformData([T()]), () => 1));
    s.push(this.step('source', 'excSource', [up, loc, off, this.ex], () => uniformData([n(), this.noiseMode, this.seed]), () => Math.ceil(n() / WG)));
    // filter and output
    s.push(this.step('filter', 'stft', [this.ex, a, Y], () => uniformData([this.stream ? this.stream.end - this.stream.start : T(), this.stream ? this.stream.total * NSF.hop : n(), O4 * 4, { f: NSF.maxMag }, this.stream?.start ?? 0, this.stream ? this.stream.start - this.stream.from : 0]), () => this.stream ? this.stream.end - this.stream.start : T()));
    s.push(this.step('istft', 'istft', [Y, frames], () => uniformData([this.stream ? this.stream.end - this.stream.start : T(), this.stream?.start ?? 0]), () => this.stream ? this.stream.end - this.stream.start : T()));
    s.push(this.step('istft', 'ola', [frames, this.audio], () => uniformData([this.stream?.end ?? T(), this.stream?.sampleCount ?? n(), this.stream?.sampleStart ?? 0]), () => Math.ceil((this.stream?.sampleCount ?? n()) / WG)));
    this.steps = s;
  }

  /** z: written to the decoder's input here; null: the caller copies it in (runFromBuffer). */
  private prepare(z: Float32Array | null, T: number, opt: NsfRunOptions): Step[] {
    if (T < 3 || T > this.maxT) throw new Error(`nsf decoder T ${T} (3..${this.maxT})`);
    this.T = T;
    if (z) this.device.queue.writeBuffer(this.zIn, 0, z.buffer, z.byteOffset, T * NSF.zDim * 4);
    if (opt.f0) this.device.queue.writeBuffer(this.f0, 0, opt.f0.buffer, opt.f0.byteOffset, T * 4);
    this.noiseMode = opt.noise ? 0 : 1;
    if (opt.noise) {
      const e = new Float32Array(2 * T * NSF.hop);
      for (let i = 0; i < T * NSF.hop; i++) e[2 * i + 1] = opt.noise[i]!;
      this.device.queue.writeBuffer(this.ex, 0, e.buffer, 0, e.byteLength);
    }
    const steps = opt.f0 ? this.steps.filter((st) => st.part !== 'f0z') : this.steps;
    for (const st of steps) this.device.queue.writeBuffer(st.uniform, 0, st.params());
    return steps;
  }

  private encode(enc: GpuDevice, steps: Step[]): void {
    const pass = enc.beginComputePass();
    for (const st of steps) {
      const groups = st.groups();
      pass.setPipeline(st.pipeline);
      pass.setBindGroup(0, st.bind);
      pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
    }
    pass.end();
  }

  /** z [T, zDim] -> audio [T * 512] and the frame F0 [T] used (one submit). */
  async run(z: Float32Array, T: number, opt: NsfRunOptions = {}): Promise<{ audio: Float32Array; f0: Float32Array }> {
    return this.runEncoded(this.prepare(z, T, opt), T, null);
  }

  /** The same on z [T, zDim] already in a GPU buffer (src rows from..from+T-1: the own TTS generator's output). */
  async runFromBuffer(src: GpuBuffer, from: number, T: number, opt: NsfRunOptions = {}): Promise<{ audio: Float32Array; f0: Float32Array }> {
    const steps = this.prepare(null, T, opt);
    return this.runEncoded(steps, T, (enc) => enc.copyBufferToBuffer(src, from * NSF.zDim * 4, this.zIn, 0, T * NSF.zDim * 4));
  }

  private async runEncoded(steps: Step[], T: number, before: ((enc: GpuDevice) => void) | null): Promise<{ audio: Float32Array; f0: Float32Array }> {
    const enc = this.device.createCommandEncoder();
    before?.(enc);
    this.encode(enc, steps);
    const bytes = WgslNsfDecoder.samples(T) * 4;
    enc.copyBufferToBuffer(this.audio, 0, this.read, 0, bytes);
    enc.copyBufferToBuffer(this.f0, 0, this.f0Read, 0, T * 4);
    this.device.queue.submit([enc.finish()]);
    await Promise.all([this.read.mapAsync(1, 0, bytes), this.f0Read.mapAsync(1, 0, T * 4)]);
    const audio = this.level(new Float32Array(this.read.getMappedRange(0, bytes).slice(0)));
    const f0 = new Float32Array(this.f0Read.getMappedRange(0, T * 4).slice(0));
    this.read.unmap();
    this.f0Read.unmap();
    return { audio, f0 };
  }

  /**
   * Exact lookahead streaming of a non-causal model (not independent padded-window synthesis).
   * f0/excitation are global: oscillator prefix and PCG sample indices never reset. Trunk radius is
   * (embed + 8 blocks) * 3 = 27 frames. Global inverse-FFT frames retain the OLA tail; two future
   * filter frames are needed before a sample is final. The caller must keep src alive until done.
   */
  async *streamFromBuffer(src: GpuBuffer, T: number, blockFrames = 128, opt: NsfRunOptions = {}): AsyncGenerator<NsfPcmChunk> {
    if (!Number.isInteger(blockFrames) || blockFrames < 4 || blockFrames > this.maxT) throw new Error('nsf streaming block size');
    if (this.stream) throw new Error('nsf streaming already active');
    const t0 = performance.now();
    const steps = this.prepare(null, T, opt);
    const pre = this.device.createCommandEncoder();
    pre.copyBufferToBuffer(src, 0, this.zIn, 0, T * NSF.zDim * 4);
    this.encode(pre, steps.filter((s) => s.part === 'f0z' || s.part === 'upsample' || s.part === 'source'));
    pre.copyBufferToBuffer(this.f0, 0, this.f0Cache, 0, T * 4);
    this.device.queue.submit([pre.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const prepareMs = performance.now() - t0;
    let emitted = 0;
    try {
      for (let start = 0; start < T; start += blockFrames) {
        const end = Math.min(T, start + blockFrames);
        const from = Math.max(0, start - 27);
        const to = Math.min(T, end + 27);
        // Outputs before end*hop - nfft/2 cannot depend on a future filter frame.
        const sampleEnd = end === T ? T * NSF.hop : Math.max(0, end * NSF.hop - NSF.nFft / 2);
        this.T = to - from;
        this.stream = { total: T, start, end, from, sampleStart: emitted, sampleCount: sampleEnd - emitted };
        const work = steps.filter((s) => s.part !== 'f0z' && s.part !== 'upsample' && s.part !== 'source');
        for (const st of work) this.device.queue.writeBuffer(st.uniform, 0, st.params());
        const at = performance.now();
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(src, from * NSF.zDim * 4, this.zIn, 0, this.T * NSF.zDim * 4);
        enc.copyBufferToBuffer(this.f0Cache, from * 4, this.f0, 0, this.T * 4);
        this.encode(enc, work);
        const bytes = (sampleEnd - emitted) * 4;
        enc.copyBufferToBuffer(this.audio, emitted * 4, this.read, 0, bytes);
        this.device.queue.submit([enc.finish()]);
        await this.read.mapAsync(1, 0, bytes);
        const audio = this.level(new Float32Array(this.read.getMappedRange(0, bytes).slice(0)));
        this.read.unmap();
        yield { audio, final: end === T, sampleStart: emitted, blockMs: performance.now() - at, prepareMs, firstMs: performance.now() - t0, frames: to - from };
        emitted = sampleEnd;
        if (end < T) await decoderRenderTurn();
      }
    } finally {
      this.stream = null;
      this.T = T;
    }
  }

  /** The same run, one submit per part, each waited for: ms per part (the sum is a little over run()'s time). */
  async profile(z: Float32Array, T: number, opt: NsfRunOptions = {}): Promise<Record<NsfPart, number>> {
    const steps = this.prepare(z, T, opt);
    const out = {} as Record<NsfPart, number>;
    await this.device.queue.onSubmittedWorkDone();
    for (const part of NSF_PARTS) {
      const ps = steps.filter((st) => st.part === part);
      if (!ps.length) { out[part] = 0; continue; }
      const t0 = performance.now();
      const enc = this.device.createCommandEncoder();
      this.encode(enc, ps);
      this.device.queue.submit([enc.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      out[part] = performance.now() - t0;
    }
    return out;
  }

  destroy(): void {
    for (const b of this.buffers) b.destroy();
    this.buffers.length = 0;
  }
}

export interface NsfPcmChunk { audio: Float32Array; final: boolean; sampleStart: number; blockMs: number; prepareMs: number; firstMs: number; frames: number }

/** Yield through a complete rendering opportunity, including when called from an rAF callback. */
export function decoderRenderTurn(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));
}
