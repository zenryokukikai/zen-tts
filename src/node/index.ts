// zen-tts/node: ZEN-TTS on Node.js (>= 26.12.1), CPU only.
//
//   import { ZenTtsNode } from 'zen-tts/node';
//   const tts = await ZenTtsNode.load({ modelsPath: './models' });
//   const r = await tts.speak('こんにちは。', { emotion: { happy: 70, neutral: 30 } });   // r.pcm: Float32Array, 44.1 kHz
//   for await (const c of tts.speakStream(longText)) { ... c.pcm ... }                 // one sentence at a time
//
// The same steps and weights as the browser's CPU path (src/runtime/cpu-tts.ts: text frontend, G2P, prosody, emotion
// blend / curve, dictionary). The kernels run in the native build of native/tts-cpu.c when it is there (linux-x64:
// AVX-512 > AVX2 > SSE4.1, picked at load), else in cpu/tts-cpu.wasm. The kernels give the same audio on every build
// (same sums in the same order, no FMA), so the output equals the browser's CPU output.
import { CpuOwnTts } from '../runtime/cpu-tts.js';
import { nsfExportToWeights, nsfF0Norm, nsfOutputGain, type NsfExportMeta } from '../runtime/nsf-weights.js';
import type { NativeKernels } from '../runtime/cpu-kernels.js';
import type { OwnMeta } from '../runtime/tts-own-prosody.js';
import type { OwnSpeakOptions, OwnSpeakResult } from '../runtime/tts-own-device.js';
import { withBuiltinDict } from '../runtime/builtin-dict.js';
import { curveFn, curvePoints, engineOptionsOf, splitSentences, type SpeakOptions } from '../runtime/speak-options.js';
import { nodeApis, type NodeApis } from './node-api.js';

export { emotionNames, neutralBlend, type EmotionBlend, type EmotionName } from '../runtime/emotion.js';
export { emotionBlend, splitSentences, type EmotionAmounts, type SpeakOptions } from '../runtime/speak-options.js';
export { TTS_DEMO_CURVES as EMOTION_CURVES, type TtsDemoCurve as EmotionCurve } from '../runtime/emotion-curve.js';
export type { OwnDictEntry as DictEntry, OwnDictSettings as DictSettings } from '../runtime/tts-own-prosody.js';
export { DEFAULT_OWN_DICT_SETTINGS as DEFAULT_DICT_SETTINGS } from '../runtime/tts-own-prosody.js';

export const SAMPLE_RATE = 44100;
export { VERSION, VOICE, VOICES, voiceDirs, type VoiceId } from '../version.js';
import { voiceDirs, type VoiceId } from '../version.js';

/** The kernel build in use: a native x86-64 build, or the WebAssembly SIMD build. */
export type TtsKernel = 'avx512' | 'avx2' | 'sse41' | 'wasm';
/** 'auto' (default): the best native build this CPU runs, else wasm; 'native': a native build or an error; 'wasm'. */
export type NodeTtsBackend = 'auto' | 'native' | 'wasm';

export interface NodeLoadOptions {
  /** Directory with tts/own.{json,bin} and decoder/nsf.{json,bin} (default: ./models of the working directory). */
  modelsPath?: string;
  /** The voice (default 'marine'): 'marine' reads tts/ and decoder/, another id voices/<id>/tts/ and voices/<id>/decoder/
   *  under modelsPath (voiceDirs). */
  voice?: VoiceId;
  /** default 'auto' */
  backend?: NodeTtsBackend;
  /** Force one native build: 'avx512' | 'avx2' | 'sse41' (also ZEN_TTS_KERNEL). */
  kernel?: Exclude<TtsKernel, 'wasm'>;
  /** Directory with the native builds zen_tts_<kernel>.node (default: native/x86/build of the package; also ZEN_TTS_NATIVE_DIR). */
  nativeDir?: string;
}

/** speak()'s result: the audio and what the model read. */
export interface NodeSpeakResult extends OwnSpeakResult {
  /** The audio (mono float32 in [-1, 1]); the same array as `audio`. */
  pcm: Float32Array;
  sampleRate: number;
}

/** One sentence of speakStream(). */
export interface NodeSpeechChunk {
  index: number;
  text: string;
  pcm: Float32Array;
  sampleRate: number;
  timing: OwnSpeakResult['timing'];
}

const BUILDS: Exclude<TtsKernel, 'wasm'>[] = ['avx512', 'avx2', 'sse41'];

/** CPU features a native build needs (from /proc/cpuinfo on Linux; empty elsewhere). */
function cpuFlags(n: NodeApis): Set<string> {
  if (n.process.platform !== 'linux') return new Set();
  try {
    const t = new TextDecoder().decode(n.fs.readFileSync('/proc/cpuinfo'));
    const line = t.split('\n').find((l) => l.startsWith('flags'));
    return new Set((line?.split(':')[1] ?? '').trim().split(/\s+/));
  } catch { return new Set(); }
}
const NEEDS: Record<Exclude<TtsKernel, 'wasm'>, string[]> = { avx512: ['avx512f'], avx2: ['avx2'], sse41: ['sse4_1'] };

/** The package root (two levels above dist/node/). */
function packageRoot(n: NodeApis): string {
  return n.path.resolve(n.path.dirname(n.url.fileURLToPath(import.meta.url)), '..', '..');
}

function loadNative(n: NodeApis, o: NodeLoadOptions): { addon: NativeKernels; kernel: Exclude<TtsKernel, 'wasm'> } | { error: string } {
  if (n.process.platform !== 'linux' || n.process.arch !== 'x64') return { error: `no native build for ${n.process.platform}-${n.process.arch}` };
  const dir = o.nativeDir ?? n.process.env['ZEN_TTS_NATIVE_DIR'] ?? n.path.join(packageRoot(n), 'native', 'x86', 'build');
  const forced = (o.kernel ?? n.process.env['ZEN_TTS_KERNEL']) as Exclude<TtsKernel, 'wasm'> | undefined;
  if (forced && !BUILDS.includes(forced)) return { error: `kernel must be ${BUILDS.join(', ')}` };
  const flags = cpuFlags(n);
  const req = n.module.createRequire(import.meta.url);
  const tried: string[] = [];
  for (const k of forced ? [forced] : BUILDS) {
    if (!NEEDS[k].every((f) => flags.has(f))) { tried.push(`${k}: CPU lacks ${NEEDS[k].join('+')}`); continue; }
    const p = n.path.join(dir, `zen_tts_${k}.node`);
    if (!n.fs.existsSync(p)) { tried.push(`${k}: ${p} not built`); continue; }
    try { return { addon: req(p) as NativeKernels, kernel: k }; } catch (e) { tried.push(`${k}: ${e instanceof Error ? e.message : String(e)}`); }
  }
  return { error: `no native build usable (${tried.join('; ')}); build with \`pnpm run build:native\`` };
}

const buf = (b: Uint8Array): ArrayBuffer => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;

export class ZenTtsNode {
  private constructor(private readonly tts: CpuOwnTts, readonly kernel: TtsKernel, readonly voice: VoiceId) {}

  static async load(o: NodeLoadOptions = {}): Promise<ZenTtsNode> {
    const n = await nodeApis();
    const models = n.path.resolve(o.modelsPath ?? 'models');
    const dirs = voiceDirs(o.voice);
    const read = (p: string): Uint8Array => {
      const f = n.path.join(models, p);
      if (!n.fs.existsSync(f)) throw new Error(`zen-tts/node: ${f} is missing (modelsPath: the directory with ${dirs.tts}/ and ${dirs.decoder}/)`);
      return n.fs.readFileSync(f);
    };
    const meta = JSON.parse(new TextDecoder().decode(read(`${dirs.tts}/own.json`))) as OwnMeta;
    const nsfMeta = JSON.parse(new TextDecoder().decode(read(`${dirs.decoder}/nsf.json`))) as NsfExportMeta;
    const wasm = n.fs.readFileSync(n.path.join(packageRoot(n), 'cpu', 'tts-cpu.wasm'));
    const tts = await CpuOwnTts.create(wasm, meta, buf(read(`${dirs.tts}/own.bin`)), nsfExportToWeights(nsfMeta, buf(read(`${dirs.decoder}/nsf.bin`))), nsfF0Norm(nsfMeta), nsfOutputGain(nsfMeta));
    const want = o.backend ?? 'auto';
    let kernel: TtsKernel = 'wasm';
    if (want !== 'wasm') {
      const r = loadNative(n, o);
      if ('addon' in r) { tts.k.useNative(r.addon); kernel = r.kernel; } else if (want === 'native') throw new Error(`zen-tts/node: ${r.error}`);
    }
    return new ZenTtsNode(tts, kernel, o.voice ?? 'marine');
  }

  /** One sentence (up to about 11 s of speech) to 44.1 kHz mono audio. Synchronous work on this thread. */
  async speak(text: string, o: SpeakOptions = {}): Promise<NodeSpeakResult> {
    return this.speakWith(text, engineOptionsOf(o));
  }

  /** speak() with the engine's options as they are (blend / blendAt, labels, f0Add, durScale). */
  async speakWith(text: string, o: OwnSpeakOptions): Promise<NodeSpeakResult> {
    if (o.stream) throw new Error('stream is not supported here (use speakStream)');
    const { blendAt, stream: _s, ...rest } = o;
    // a blendAt curve is sampled at the same points as on the browser's CPU path, so the audio matches it
    const opts = blendAt ? { ...rest, blendAt: curveFn(curvePoints(blendAt)) } : rest;
    const r = this.tts.speak(text, { ...opts, dict: withBuiltinDict(o.dict) });
    return { ...r, pcm: r.audio, sampleRate: SAMPLE_RATE };
  }

  /**
   * A text of any length, one sentence at a time (splitSentences): yields each sentence's audio when it is made, so it
   * can be played, streamed or lip-synced while the next one is synthesised.
   */
  async *speakStream(text: string, o: SpeakOptions = {}): AsyncGenerator<NodeSpeechChunk> {
    const opts = engineOptionsOf(o);
    let index = 0;
    for (const sentence of splitSentences(text)) {
      const r = await this.speakWith(sentence, opts);
      yield { index: index++, text: sentence, pcm: r.pcm, sampleRate: SAMPLE_RATE, timing: r.timing };
      // let other work run between sentences (synthesis itself is synchronous)
      await new Promise<void>((res) => setTimeout(res, 0));
    }
  }

  /** Readings and accent marks the model would use (after the dictionary). */
  analyze(text: string, dict: Parameters<CpuOwnTts['analyze']>[1] = []): ReturnType<CpuOwnTts['analyze']> {
    return this.tts.analyze(text, withBuiltinDict(dict));
  }

  /** Bytes of the kernels' memory (weights + work). */
  get memoryBytes(): number { return this.tts.k.bytes; }

  destroy(): void { /* the memory is freed with the object */ }
}

/** Mono float audio -> 16-bit PCM WAV bytes. */
export function wavBytes(pcm: Float32Array, sampleRate = SAMPLE_RATE): Uint8Array {
  const n = pcm.length;
  const out = new Uint8Array(44 + n * 2);
  const v = new DataView(out.buffer);
  const s = (o: number, t: string): void => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
  s(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); s(8, 'WAVE'); s(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); s(36, 'data'); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, pcm[i]!)) * 0x7fff, true);
  return out;
}
