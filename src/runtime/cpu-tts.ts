// The TTS (tts-own-device.ts WgslOwnTts) on the CPU with the tts-cpu.wasm kernels, then the NSF
// decoder (cpu-nsf.ts). The same steps and the same stored weights (int8 / fp16 / f32 as in own.bin); the CPU parts
// of WgslOwnTts (G2P embedding, argmax, predictor input, prosody, dictionary) are the same shared functions. DOM-free.
import { emotionNames, type EmotionBlend } from './emotion.js';
import { normalizeOwnText } from './ja-text-normalize.js';
import {
  applyOwnDict, dictLabels, dictProsody, frameRows, predictorInput, prosodyFromHead, slotsToPhones, templateParams, voiceDictSettings,
  type DictWord, type OwnDictEntry, type OwnMeta, type OwnProsody,
} from './tts-own-prosody.js';
import type { OwnSpeakOptions, OwnSpeakResult } from './tts-own-device.js';
import { CpuKernels, type CpuLinear } from './cpu-kernels.js';
import { CpuNsfDecoder } from './cpu-nsf.js';
import type { NsfF0Norm, NsfWeights } from './nsf-weights.js';

const MAX_CHARS = 256;
const MAX_PHONES = 512;
const MAX_FRAMES = 2048;

interface Layer { qkv: CpuLinear; o: CpuLinear; ff1: CpuLinear; ff2: CpuLinear; relk: number; relv: number; ln1: number; ln2: number }

export class CpuOwnTts {
  private readonly cid = new Map<string, number>();
  private readonly symbolId = new Map<string, number>();
  private readonly punct: Set<string>;
  private readonly g2pLn: number;
  private readonly g2pLayers: Layer[];
  private readonly g2pHead: CpuLinear;
  private readonly predLn: number;
  private readonly predLayers: Layer[];
  private readonly predHead: CpuLinear;
  private readonly genInp: CpuLinear;
  private readonly genLayers: { dw: number; pw1: CpuLinear; pw2: CpuLinear }[];
  private readonly genNorm: number;
  private readonly genOut: CpuLinear;
  readonly decoder: CpuNsfDecoder;
  /** the predictor's hidden [P, Dp] of the last predict() (scratch-free: kept for the generator) */
  private hidden: Float32Array = new Float32Array(0);

  private constructor(readonly k: CpuKernels, readonly meta: OwnMeta, private readonly bin: ArrayBuffer, nsf: NsfWeights, f0Norm?: NsfF0Norm, outputGain?: number) {
    meta.vocab.forEach((ch, i) => this.cid.set(ch, i + 2));
    meta.symbols.forEach((s, i) => this.symbolId.set(s, i));
    this.punct = new Set(meta.punctuation);
    const S = meta.shape;
    const layer = (n: string, D: number, ff: number, kin: number): Layer => ({
      qkv: this.lin(`${n}.qkv`, D, 3 * D), o: this.lin(`${n}.o`, D, D), ff1: this.lin(`${n}.ff1`, kin * D, ff), ff2: this.lin(`${n}.ff2`, kin * ff, D),
      relk: k.put(this.f32(`${n}.relk`)), relv: k.put(this.f32(`${n}.relv`)), ln1: k.put(this.f32(`${n}.ln1`)), ln2: k.put(this.f32(`${n}.ln2`)),
    });
    this.g2pLn = k.put(this.f32('g2p.ln'));
    this.g2pLayers = Array.from({ length: S.g2p.layers }, (_, l) => layer(`g2p.l${l}`, S.g2p.dim, S.g2p.ff, 1));
    this.g2pHead = this.lin('g2p.head', S.g2p.dim, S.g2p.slots * (S.g2p.phoneClasses + S.g2p.marks));
    this.predLn = k.put(this.f32('pred.ln'));
    this.predLayers = Array.from({ length: S.pred.layers }, (_, l) => layer(`pred.l${l}`, S.pred.dim, S.pred.ff, S.pred.kernel));
    this.predHead = this.lin('pred.head', S.pred.dim, 4);
    this.genInp = this.lin('gen.inp', S.pred.dim + S.gen.extra, S.gen.dim);
    this.genLayers = Array.from({ length: S.gen.layers }, (_, l) => ({
      dw: k.put(this.f32(`gen.l${l}.dw`)), pw1: this.lin(`gen.l${l}.pw1`, S.gen.dim, S.gen.ff), pw2: this.lin(`gen.l${l}.pw2`, S.gen.ff, S.gen.dim),
    }));
    this.genNorm = k.put(this.f32('gen.norm'));
    this.genOut = this.lin('gen.out', S.gen.dim, S.gen.z);
    this.decoder = new CpuNsfDecoder(k, nsf, undefined, f0Norm, outputGain);
  }

  /**
   * f0Norm: the decoder's log-F0 normalisation (nsfF0Norm of its nsf.json; omitted = 「全力 マリン」's).
   * outputGain: the decoder's output level (nsfOutputGain of its nsf.json; omitted = 1).
   */
  static async create(wasm: ArrayBuffer | Uint8Array, meta: OwnMeta, bin: ArrayBuffer, nsf: NsfWeights, f0Norm?: NsfF0Norm, outputGain?: number): Promise<CpuOwnTts> {
    return new CpuOwnTts(await CpuKernels.create(wasm), meta, bin, nsf, f0Norm, outputGain);
  }

  private tensor(name: string): OwnMeta['tensors'][number] {
    const t = this.meta.tensors.find((x) => x.name === name);
    if (!t) throw new Error(`own tts tensor ${name} missing`);
    return t;
  }

  private f32(name: string): Float32Array {
    const t = this.tensor(name);
    if (t.dtype !== 'f32') throw new Error(`own tts tensor ${name}: ${t.dtype}`);
    return new Float32Array(this.bin, t.offset, t.bytes / 4);
  }

  /** As WgslOwnTts.lin: int8 `.q` [I, O4 * 4] with `.s`, or fp16 `.w` [I, O]; `.b`. */
  private lin(name: string, I: number, O: number): CpuLinear {
    const bias = this.f32(`${name}.b`);
    const q = this.meta.tensors.find((t) => t.name === `${name}.q`);
    if (q) {
      if (q.shape[0] !== I || q.shape[1] !== Math.ceil(O / 4) * 4) throw new Error(`own tts ${name}.q shape ${q.shape}`);
      return this.k.linear('i8', new Int8Array(this.bin, q.offset, q.bytes), I, O, bias, this.f32(`${name}.s`));
    }
    const t = this.tensor(`${name}.w`);
    if (t.dtype !== 'f16' || t.shape[0] !== I || t.shape[1] !== O) throw new Error(`own tts ${name}.w shape ${t.shape}`);
    return this.k.linear('f16', new Uint16Array(this.bin, t.offset, I * O), I, O, bias);
  }

  /** LN -> layers (attention, FFN as k-conv when kin > 1); x0 [T, D] in -> x [T, D] (scratch). */
  private encoder(x0: number, T: number, D: number, H: number, ff: number, kin: number, ln: number, layers: Layer[]): number {
    const k = this.k;
    const eps = this.meta.shape.lnEps;
    const x = k.floats(T * D);
    const x2 = k.floats(T * D);
    const qkv = k.floats(T * 3 * D);
    const att = k.floats(T * D);
    const f = k.floats(T * ff);
    const col = kin > 1 ? k.floats(T * kin * Math.max(D, ff)) : 0;
    k.dwNorm(x0, T, D, 0, false, ln, x, eps);
    for (const L of layers) {
      k.lin(L.qkv, x, T, qkv);
      k.vattn(qkv, T, H, D / H, this.meta.shape.window, L.relk, L.relv, att);
      k.lin(L.o, att, T, x, 0, 1);
      k.dwNorm(x, T, D, 0, false, L.ln1, x2, eps);
      if (kin > 1) {
        k.im2col(x2, T, D, kin, col);
        k.lin(L.ff1, col, T, f, 2);
        k.im2col(f, T, ff, kin, col);
        k.lin(L.ff2, col, T, x2, 0, 1);
      } else {
        k.lin(L.ff1, x2, T, f, 1);
        k.lin(L.ff2, f, T, x2, 0, 1);
      }
      k.dwNorm(x2, T, D, 0, false, L.ln2, x, eps);
    }
    return x;
  }

  g2p(ids: readonly number[]): number[][] {
    const g = this.meta.shape.g2p;
    const N = ids.length;
    if (N < 1 || N > MAX_CHARS) throw new Error(`own tts: ${N} chars (max ${MAX_CHARS})`);
    const D = g.dim;
    const q = this.tensor('g2p.emb.q');
    const emb = new Int8Array(this.bin, q.offset, q.bytes);
    const sc = this.f32('g2p.emb.s');
    const O = g.slots * (g.phoneClasses + g.marks);
    const k = this.k;
    const logits = k.scratch(() => {
      const xin = k.floats(N * D);
      const xv = k.f32(xin, N * D);
      ids.forEach((id, i) => { for (let c = 0; c < D; c++) xv[i * D + c] = id === this.meta.charPad ? 0 : emb[id * D + c]! * sc[id]!; });
      const x = this.encoder(xin, N, D, g.heads, g.ff, 1, this.g2pLn, this.g2pLayers);
      const O4 = this.g2pHead.O4 * 4;
      const y = k.floats(N * O4);
      k.lin(this.g2pHead, x, N, y);
      const raw = k.f32(y, N * O4);
      const out = new Float32Array(N * O);
      for (let i = 0; i < N; i++) out.set(raw.subarray(i * O4, i * O4 + O), i * O);
      return out;
    });
    const nP = g.slots * g.phoneClasses;
    const argmax = (o: number, n: number): number => { let b = 0; for (let j = 1; j < n; j++) if (logits[o + j]! > logits[o + b]!) b = j; return b; };
    return ids.map((_, i) => {
      const r: number[] = [];
      for (let s = 0; s < g.slots; s++) r.push(argmax(i * O + s * g.phoneClasses, g.phoneClasses));
      for (let s = 0; s < g.slots; s++) r.push(argmax(i * O + nP + s * g.marks, g.marks));
      return r;
    });
  }

  predict(input: ReturnType<typeof predictorInput>, blend: EmotionBlend, blendAt?: (x: number) => EmotionBlend): Float32Array {
    const p = this.meta.shape.pred;
    const Dp = p.dim;
    const P = input.phones.length;
    if (P > MAX_PHONES) throw new Error(`own tts: ${P} phones (max ${MAX_PHONES})`);
    const emb = this.f32('pred.emb');
    const acc = this.f32('pred.accent');
    const sty = this.f32('pred.style');
    const style = new Float32Array(Dp);
    emotionNames.forEach((n, i) => { const w = blend[n]; if (w) for (let c = 0; c < Dp; c++) style[c] = style[c]! + w * sty[i * Dp + c]!; });
    const k = this.k;
    return k.scratch(() => {
      const xin = k.floats(P * Dp);
      const xv = k.f32(xin, P * Dp);
      for (let i = 0; i < P; i++) {
        const id = this.symbolId.get(input.phones[i]!);
        if (id === undefined) throw new Error(`own tts: phone ${input.phones[i]} not in the symbols`);
        const a = [input.acc.tone[i]!, input.acc.nucleus[i]!, input.acc.boundary[i]!, input.acc.pos[i]!];
        if (blendAt) {
          const b = blendAt(P > 1 ? i / (P - 1) : 0);
          style.fill(0);
          emotionNames.forEach((n, j) => { const w = b[n]; if (w) for (let c = 0; c < Dp; c++) style[c] = style[c]! + w * sty[j * Dp + c]!; });
        }
        for (let c = 0; c < Dp; c++) xv[i * Dp + c] = emb[id * Dp + c]! + a[0]! * acc[c]! + a[1]! * acc[Dp + c]! + a[2]! * acc[2 * Dp + c]! + a[3]! * acc[3 * Dp + c]! + style[c]!;
      }
      const x = this.encoder(xin, P, Dp, p.heads, p.ff, p.kernel, this.predLn, this.predLayers);
      this.hidden = k.f32(x, P * Dp).slice();
      const y = k.floats(P * 4);
      k.lin(this.predHead, x, P, y);
      return k.f32(y, P * 4).slice();
    });
  }

  /** z [F, 192] of frame rows (the hidden of the last predict()). */
  generate(pr: Pick<OwnProsody, 'dur' | 'f0' | 'energy' | 'voiced'>, blend: EmotionBlend): Float32Array {
    const S = this.meta.shape;
    const n = S.gen;
    const Dp = S.pred.dim;
    const { idx, extra } = frameRows(pr);
    const F = idx.length;
    if (F > MAX_FRAMES) throw new Error(`own tts: ${F} frames (max ${MAX_FRAMES})`);
    const k = this.k;
    return k.scratch(() => {
      const ip = k.alloc(F * 4);
      k.u32(ip, F).set(idx);
      const hp = k.floats(this.hidden.length, this.hidden);
      const ep = k.floats(F * 4, extra);
      const gin = k.floats(F * (Dp + n.extra));
      k.frameIn(ip, F, Dp, hp, ep, gin);
      const bias = k.f32(this.genInp.bias, this.genInp.O4 * 4).slice();
      const gs = this.f32('gen.style');
      emotionNames.forEach((e, i) => { const w = blend[e]; if (w) for (let c = 0; c < n.dim; c++) bias[c] = bias[c]! + w * gs[i * n.dim + c]!; });
      const bp = k.floats(bias.length, bias);
      const gx = k.floats(F * n.dim);
      const gh = k.floats(F * n.dim);
      const ga = k.floats(F * n.ff);
      k.lin(this.genInp, gin, F, gx, 0, 0, bp);
      for (const L of this.genLayers) {
        k.dwNorm(gx, F, n.dim, n.kernel, true, L.dw, gh, S.lnEps);
        k.lin(L.pw1, gh, F, ga, 1);
        k.lin(L.pw2, ga, F, gx, 0, 1);
      }
      k.dwNorm(gx, F, n.dim, 0, false, this.genNorm, gh, S.lnEps);
      const z = k.floats(F * this.genOut.O4 * 4);
      k.lin(this.genOut, gh, F, z);
      const O4 = this.genOut.O4 * 4;
      const zv = k.f32(z, F * O4);
      const out = new Float32Array(F * n.z);
      for (let f = 0; f < F; f++) out.set(zv.subarray(f * O4, f * O4 + n.z), f * n.z);
      return out;
    });
  }

  /** G2P labels of text (normalized, after dictionary L1), as WgslOwnTts.analyze. */
  analyze(text: string, dict: readonly OwnDictEntry[] = []): { norm: string; g2pText: string; unknown: string[]; g2p: { phones: string[]; marks: number[]; spans: [number, number][] }; phones: string[]; marks: number[]; words: DictWord[]; aligned: boolean } {
    const norm = normalizeOwnText(text);
    const d = applyOwnDict(norm, dict);
    const ids: number[] = [];
    const unknown: string[] = [];
    for (const ch of d.text) { const id = this.cid.get(ch); if (id === undefined) unknown.push(ch); ids.push(id ?? this.meta.charUnk); }
    const g = slotsToPhones(this.g2p(ids), this.meta, this.meta.shape.g2p.slots);
    return { norm, g2pText: d.text, unknown, g2p: g, ...dictLabels(g, d.matches, this.meta.kana) };
  }

  /** One sentence to audio (44.1 kHz): the same steps and options as WgslOwnTts.speak (no stream). */
  speak(text: string, o: Omit<OwnSpeakOptions, 'stream'>): OwnSpeakResult {
    const t0 = performance.now();
    const settings = voiceDictSettings(o.settings, this.meta);
    const a = this.analyze(text, o.dict ?? []);
    const t1 = performance.now();
    const phones = o.labels?.phones ?? a.phones;
    const marks = o.labels?.marks ?? a.marks;
    const words = o.labels ? a.words.filter((w) => w.b <= phones.length) : a.words;
    const strength = o.strength ?? 1;
    let base: OwnProsody | null = null;
    if (words.some((w) => w.entry.accent !== undefined) && a.aligned && !o.labels) {
      const inB = predictorInput(a.g2p.phones, a.g2p.marks, this.punct);
      base = prosodyFromHead(inB, this.predict(inB, o.blend), this.meta, o.blend, strength);
    }
    const inp = predictorInput(phones, marks, this.punct);
    const pr = prosodyFromHead(inp, this.predict(inp, o.blend, o.blendAt), this.meta, o.blend, strength);
    if (o.blendAt) {
      const P = pr.phones.length;
      pr.phones.forEach((_, i) => {
        const { B, H, D } = templateParams(this.meta, o.blendAt!(P > 1 ? i / (P - 1) : 0));
        pr.template[i] = B + H * pr.acc.tone[i]! - D * pr.acc.pos[i]!;
        pr.f0[i] = pr.template[i]! + strength * pr.dev[i]!;
      });
    }
    let { f0, dur } = { f0: pr.f0, dur: pr.dur };
    if (words.length) ({ f0, dur } = dictProsody(base ?? pr, pr, words, this.meta, o.blend, settings));
    if (o.f0Add) f0 = f0.map((v, i) => v + (i > 0 && i <= o.f0Add!.length ? o.f0Add![i - 1]! : 0));
    if (o.durScale) dur = dur.map((v, i) => (i > 0 && i <= o.durScale!.length ? Math.max(1, Math.round(v * o.durScale![i - 1]!)) : v));
    const t2 = performance.now();
    const rows = { dur, f0, energy: pr.energy, voiced: pr.voiced };
    let z: Float32Array;
    if (!o.blendAt) z = this.generate(rows, o.blend);
    else {
      // z per frame mixed between generator runs at CURVE_ANCHORS blends along the curve (as WgslOwnTts.synth)
      const zs: Float32Array[] = [];
      for (let k = 0; k < CURVE_ANCHORS; k++) zs.push(this.generate(rows, o.blendAt(k / (CURVE_ANCHORS - 1))));
      const Dz = this.meta.shape.gen.z;
      const F = zs[0]!.length / Dz;
      z = new Float32Array(F * Dz);
      for (let t = 0; t < F; t++) {
        const u = (F > 1 ? t / (F - 1) : 0) * (CURVE_ANCHORS - 1);
        const k = Math.min(CURVE_ANCHORS - 2, Math.floor(u));
        const h = u - k;
        const za = zs[k]!, zb = zs[k + 1]!;
        for (let c = 0; c < Dz; c++) z[t * Dz + c] = (1 - h) * za[t * Dz + c]! + h * zb[t * Dz + c]!;
      }
    }
    const F = z.length / this.meta.shape.gen.z;
    const t3 = performance.now();
    // the NSF decoder needs at least 3 frames: shorter z is padded with its last frame and the audio cut back
    const T = Math.max(3, F);
    let zz = z;
    if (T > F) { zz = new Float32Array(T * NSF_Z); zz.set(z); for (let f = F; f < T; f++) zz.set(z.subarray((F - 1) * NSF_Z, F * NSF_Z), f * NSF_Z); }
    const dec = this.decoder.run(zz, T);
    const audio = T > F ? dec.audio.slice(0, F * 512) : dec.audio;
    const t4 = performance.now();
    let nan = 0;
    for (const v of audio) if (!Number.isFinite(v)) nan++;
    const r = (v: number): number => +v.toFixed(1);
    return {
      norm: a.norm, unknown: a.unknown, g2p: { phones: a.g2p.phones, marks: a.g2p.marks }, phones, marks, words, prosody: pr, f0, dur, audio,
      timing: { g2pMs: r(t1 - t0), predMs: r(t2 - t1), genMs: r(t3 - t2), decoderMs: r(t4 - t3), totalMs: r(t4 - t0), frames: F, nan },
    };
  }
}
const NSF_Z = 192;
/** Generator runs along a blendAt curve (the same number as the WebGPU path). */
const CURVE_ANCHORS = 9;
