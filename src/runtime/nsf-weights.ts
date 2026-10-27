// The NSF decoder's shape and the conversion of nsf.{json,bin} into float32 weights. DOM-free.
import { HALF } from './half.js';

export const NSF = {
  zDim: 192,
  dim: 384,
  ffDim: 1152,
  layers: 8,
  kernel: 7,
  nFft: 2048,
  bins: 1025,
  hop: 512,
  sampleRate: 44100,
  lnEps: 1e-6,
  maxMag: 100,
  f0zDim: 192,
  f0zFf: 576,
  f0zLayers: 4,
} as const;

/**
 * float32 weights, linear ones [in, out]. Keys `<p>embed.w` [7 * C, D] (row k * C + c), `<p>embed.b`, `<p>embed.norm`
 * [w | b], per block `<p><l>.dw` [dw w (D * 7) | dw b | ln w | ln b], `<p><l>.pw1.w|b`, `<p><l>.pw2.w|b` (gamma folded in),
 * then `final.norm`, `head.w` [384, 3075], `head.b` (p = '') and `f0z.out.w` [192, 2], `f0z.out.b` (p = 'f0z.').
 */
export type NsfWeights = Record<string, Float32Array>;

export interface Net { prefix: string; C: number; D: number; F: number; layers: number }
export const TRUNK: Net = { prefix: '', C: NSF.zDim + 2, D: NSF.dim, F: NSF.ffDim, layers: NSF.layers };
export const F0Z: Net = { prefix: 'f0z.', C: NSF.zDim, D: NSF.f0zDim, F: NSF.f0zFf, layers: NSF.f0zLayers };

/** nsf.json: tensor table of nsf.bin (tensors under nsf. and f0z.), and the voice's log-F0 normalisation. */
export interface NsfExportMeta {
  dtype: 'fp16' | 'fp32';
  tensors: { name: string; shape: number[]; offset: number; bytes: number }[];
  /** mean of ln(F0 Hz) the decoder uses (trunk input and f0z output); omitted = DEFAULT_NSF_F0_NORM */
  logF0Mean?: number;
  /** its scale; omitted = DEFAULT_NSF_F0_NORM */
  logF0Std?: number;
  /**
   * Linear gain on the decoder output (omitted = 1). It normalises a voice's output level to the others' without
   * changing the weights; gain > 1 also enables the peak limiter (applyOutputLevel). hashiro: 2.8.
   */
  outputGain?: number;
}

/** The decoder's F0 input / f0z output normalisation: x = (ln F0 - logF0Mean) / logF0Std. */
export interface NsfF0Norm { logF0Mean: number; logF0Std: number }
/** The value of the 「全力 マリン」 decoder (~229 Hz), used when nsf.json does not say. */
export const DEFAULT_NSF_F0_NORM: NsfF0Norm = { logF0Mean: 5.43, logF0Std: 0.25 };

/** Plain multiplier on the decoder's output; the value when nsf.json does not say (the weights as they are). */
export const DEFAULT_NSF_OUTPUT_GAIN = 1;

/** Peak limiter of a raised voice: samples up to the knee pass unchanged, above it they bend smoothly towards the ceiling. */
export const NSF_LIMIT_KNEE = 0.8;
export const NSF_LIMIT_CEILING = 0.95;

/**
 * The decoder's output at its voice's level, in place. gain 1 returns the samples untouched. A raised voice (gain > 1)
 * also goes through a stateless soft limiter (|y| < NSF_LIMIT_CEILING), so a gain chosen for the quiet sentences does
 * not clip the loud ones; being per sample, it gives the same samples in one run and in streamed blocks.
 */
export function applyOutputLevel(audio: Float32Array, gain: number): Float32Array {
  if (gain === 1) return audio;
  const k = NSF_LIMIT_KNEE, span = NSF_LIMIT_CEILING - NSF_LIMIT_KNEE, limit = gain > 1;
  for (let i = 0; i < audio.length; i++) {
    let y = audio[i]! * gain;
    if (limit) {
      const a = Math.abs(y);
      if (a > k) y = Math.sign(y) * (k + span * Math.tanh((a - k) / span));
    }
    audio[i] = Math.fround(y);
  }
  return audio;
}

/** The output level of an nsf.json (its outputGain, else 1). */
export function nsfOutputGain(meta: Pick<NsfExportMeta, 'outputGain'>): number {
  const g = meta.outputGain ?? DEFAULT_NSF_OUTPUT_GAIN;
  if (!Number.isFinite(g) || !(g > 0)) throw new Error(`nsf.json: bad outputGain ${g}`);
  return g;
}

/** The normalisation of an nsf.json (its logF0Mean / logF0Std, else the default). */
export function nsfF0Norm(meta: Pick<NsfExportMeta, 'logF0Mean' | 'logF0Std'>): NsfF0Norm {
  const n = { logF0Mean: meta.logF0Mean ?? DEFAULT_NSF_F0_NORM.logF0Mean, logF0Std: meta.logF0Std ?? DEFAULT_NSF_F0_NORM.logF0Std };
  if (!Number.isFinite(n.logF0Mean) || !(n.logF0Std > 0)) throw new Error(`nsf.json: bad logF0Mean / logF0Std ${n.logF0Mean} / ${n.logF0Std}`);
  return n;
}

/** nsf.json tensors -> NsfWeights. */
export function nsfExportToWeights(meta: NsfExportMeta, bin: ArrayBuffer): NsfWeights {
  const byName = new Map(meta.tensors.map((t) => [t.name, t]));
  const used = new Set<string>();
  const get = (name: string, size: number): Float32Array => {
    const t = byName.get(name);
    if (!t) throw new Error(`nsf weights: ${name} missing`);
    const n = t.shape.reduce((a, b) => a * b, 1);
    if (n !== size) throw new Error(`nsf weights: ${name} has ${n} values (want ${size})`);
    used.add(name);
    const out = new Float32Array(n);
    if (meta.dtype === 'fp16') { const h = new Uint16Array(bin, t.offset, n); for (let i = 0; i < n; i++) out[i] = HALF[h[i]!]!; } else out.set(new Float32Array(bin, t.offset, n));
    return out;
  };
  const transpose = (w: Float32Array, O: number, I: number, colScale?: Float32Array): Float32Array => {
    const r = new Float32Array(I * O);
    for (let o = 0; o < O; o++) { const s = colScale ? colScale[o]! : 1; for (let i = 0; i < I; i++) r[i * O + o] = w[o * I + i]! * s; }
    return r;
  };
  const cat = (...a: Float32Array[]): Float32Array => { const r = new Float32Array(a.reduce((n, x) => n + x.length, 0)); let k = 0; for (const x of a) { r.set(x, k); k += x.length; } return r; };
  const K = NSF.kernel;
  const w: NsfWeights = {};
  for (const [net, src] of [[TRUNK, 'nsf.'], [F0Z, 'f0z.']] as const) {
    const p = net.prefix;
    const { C, D, F } = net;
    const ew = get(`${src}embed.weight`, D * C * K);
    const embed = new Float32Array(K * C * D);
    for (let o = 0; o < D; o++) for (let c = 0; c < C; c++) for (let k = 0; k < K; k++) embed[(k * C + c) * D + o] = ew[(o * C + c) * K + k]!;
    w[`${p}embed.w`] = embed;
    w[`${p}embed.b`] = get(`${src}embed.bias`, D);
    w[`${p}embed.norm`] = cat(get(`${src}norm.weight`, D), get(`${src}norm.bias`, D));
    for (let l = 0; l < net.layers; l++) {
      const b = `${src}blocks.${l}.`;
      const gamma = get(`${b}gamma`, D);
      w[`${p}${l}.dw`] = cat(get(`${b}dwconv.weight`, D * K), get(`${b}dwconv.bias`, D), get(`${b}norm.weight`, D), get(`${b}norm.bias`, D));
      w[`${p}${l}.pw1.w`] = transpose(get(`${b}pw1.weight`, F * D), F, D);
      w[`${p}${l}.pw1.b`] = get(`${b}pw1.bias`, F);
      w[`${p}${l}.pw2.w`] = transpose(get(`${b}pw2.weight`, D * F), D, F, gamma);
      const pb = get(`${b}pw2.bias`, D);
      for (let o = 0; o < D; o++) pb[o] = pb[o]! * gamma[o]!;
      w[`${p}${l}.pw2.b`] = pb;
    }
  }
  w['final.norm'] = cat(get('nsf.final_norm.weight', NSF.dim), get('nsf.final_norm.bias', NSF.dim));
  w['head.w'] = transpose(get('nsf.out.weight', 3 * NSF.bins * NSF.dim), 3 * NSF.bins, NSF.dim);
  w['head.b'] = get('nsf.out.bias', 3 * NSF.bins);
  w['f0z.out.w'] = transpose(get('f0z.out.weight', 2 * NSF.f0zDim), 2, NSF.f0zDim);
  w['f0z.out.b'] = get('f0z.out.bias', 2);
  const unused = meta.tensors.map((t) => t.name).filter((n) => !used.has(n));
  if (unused.length) throw new Error(`nsf weights: unused tensors ${unused.join(', ')}`);
  return w;
}

