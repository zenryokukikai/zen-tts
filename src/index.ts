// ZEN-TTS: the public API.
//
//   const tts = await ZenTts.load({ modelsUrl: './models' });
//   const r = await tts.speak('こんにちは。', { emotion: { happy: 70, neutral: 30 } });
//   // r.audio: Float32Array at r.sampleRate (44.1 kHz)
//
// Everything runs in the browser: on the page's WebGPU device, or on the CPU (WebAssembly SIMD in a Worker) with the
// same weights and the same steps. backend 'auto' (default) uses WebGPU when the browser has it, else the CPU.
import { WgslOwnTts, type OwnSpeakOptions, type OwnSpeakResult } from './runtime/tts-own-device.js';
import { CpuTts } from './runtime/cpu-client.js';
import { emotionNames, neutralBlend, type EmotionBlend, type EmotionName } from './runtime/emotion.js';
import { engineOptionsOf, splitSentences, type SpeakOptions } from './runtime/speak-options.js';
import type { OwnDictEntry, OwnDictSettings, OwnMeta } from './runtime/tts-own-prosody.js';
import { withBuiltinDict } from './runtime/builtin-dict.js';

export { emotionNames, neutralBlend, type EmotionBlend, type EmotionName };
export { TTS_DEMO_CURVES as EMOTION_CURVES, TTS_DEMO_CURVE_LABEL as EMOTION_CURVE_LABEL, type TtsDemoCurve as EmotionCurve } from './runtime/emotion-curve.js';
export type { OwnDictEntry as DictEntry, OwnDictSettings as DictSettings, OwnSpeakResult as SpeakResult, OwnSpeakOptions as EngineSpeakOptions };
export { DEFAULT_OWN_DICT_SETTINGS as DEFAULT_DICT_SETTINGS, pitchRangeOf } from './runtime/tts-own-prosody.js';

/** The weight set names under modelsUrl (of the default voice 'marine'; see voiceDirs). */
export const MODEL_SET = 'tts';
export const DECODER_SET = 'decoder';
export const SAMPLE_RATE = 44100;
export { VERSION, VOICE, VOICES, voiceDirs, type VoiceId } from './version.js';
import { voiceDirs, type VoiceId } from './version.js';

export { emotionBlend, splitSentences, type EmotionAmounts, type SpeakOptions } from './runtime/speak-options.js';

/** Where the voice is made: 'webgpu', 'cpu' (WebAssembly SIMD), or 'auto' (WebGPU if available, else CPU). */
export type Backend = 'auto' | 'webgpu' | 'cpu';

export interface LoadOptions {
  /** URL of the directory that holds tts/ and decoder/ (default './models'). */
  modelsUrl?: string;
  /** The voice (default 'marine'): 'marine' loads tts/ and decoder/, another id voices/<id>/tts/ and voices/<id>/decoder/
   *  under modelsUrl (voiceDirs). The text frontend and G2P are the same for every voice. */
  voice?: VoiceId;
  /** default 'auto' */
  backend?: Backend;
  /** URL of tts-cpu.wasm for the CPU backend (default './cpu/tts-cpu.wasm'). */
  cpuWasmUrl?: string;
  /** An existing GPUDevice to share (WebGPU backend); otherwise one is requested. */
  device?: unknown;
}

/** The engine under ZenTts: WgslOwnTts (WebGPU) or CpuTts (CPU worker); the same options and results. */
interface Engine {
  readonly meta: OwnMeta;
  speak(text: string, o: OwnSpeakOptions): Promise<OwnSpeakResult>;
  analyze(text: string, dict?: readonly OwnDictEntry[]): Promise<Awaited<ReturnType<WgslOwnTts['analyze']>>>;
  destroy(): void;
}

async function webgpuDevice(): Promise<unknown | undefined> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const gpu = (globalThis.navigator as any)?.gpu;
  if (!gpu) return undefined;
  const adapter = await gpu.requestAdapter();
  return adapter ? adapter.requestDevice() : undefined;
}

export class ZenTts {
  private constructor(readonly engine: Engine, readonly backend: 'webgpu' | 'cpu', readonly voice: VoiceId) {}

  static async load(o: LoadOptions = {}): Promise<ZenTts> {
    const want = o.backend ?? 'auto';
    const modelsUrl = o.modelsUrl ?? './models';
    const dirs = voiceDirs(o.voice);
    if (want !== 'cpu') {
      const device = o.device ?? await webgpuDevice();
      if (device) return new ZenTts(await WgslOwnTts.load(device, dirs.tts, dirs.decoder, false, modelsUrl), 'webgpu', o.voice ?? 'marine');
      if (want === 'webgpu') throw new Error('WebGPU is not available in this browser');
    }
    return new ZenTts(await CpuTts.load(o.cpuWasmUrl ?? './cpu/tts-cpu.wasm', modelsUrl, dirs.tts, dirs.decoder), 'cpu', o.voice ?? 'marine');
  }

  /** The model's tables (kana, punctuation, ...) for showing the readings. */
  get meta(): OwnMeta { return this.engine.meta; }

  /** One sentence (up to about 11 s of speech) to 44.1 kHz mono audio. */
  async speak(text: string, o: SpeakOptions = {}): Promise<OwnSpeakResult & { sampleRate: number }> {
    return this.speakWith(text, engineOptionsOf(o));
  }

  /**
   * A text of any length, one sentence at a time (splitSentences): yields each sentence's audio as soon as it is made,
   * for playing (or lip-syncing) while the next is synthesised. Same options as speak().
   */
  async *speakStream(text: string, o: SpeakOptions = {}): AsyncGenerator<SpeechChunk> {
    const opts = engineOptionsOf(o);
    let index = 0;
    for (const sentence of splitSentences(text)) {
      const r = await this.speakWith(sentence, opts);
      yield { index: index++, text: sentence, pcm: r.audio, sampleRate: SAMPLE_RATE, timing: r.timing };
    }
  }

  /** speak() with the engine's options as they are: a blend (or blendAt curve), and edits from an intonation UI
   *  (labels: accent marks, f0Add: semitones per phone, durScale: length per phone). */
  async speakWith(text: string, o: OwnSpeakOptions): Promise<OwnSpeakResult & { sampleRate: number }> {
    if (o.stream) throw new Error('stream is not supported here');
    const r = await this.engine.speak(text, { ...o, dict: withBuiltinDict(o.dict) });
    return { ...r, sampleRate: SAMPLE_RATE };
  }

  /** Readings and accent marks the model would use (after the dictionary), for checking / editing. */
  analyze(text: string, dict: readonly OwnDictEntry[] = []): ReturnType<Engine['analyze']> {
    return this.engine.analyze(text, withBuiltinDict(dict));
  }

  destroy(): void { this.engine.destroy(); }
}

/** One sentence of speakStream(): its text and audio (mono float32, 44.1 kHz). */
export interface SpeechChunk {
  index: number;
  text: string;
  pcm: Float32Array;
  sampleRate: number;
  timing: OwnSpeakResult['timing'];
}

/** Mono float audio -> a 16-bit PCM WAV Blob. */
export function wavBlob(audio: Float32Array, sampleRate = SAMPLE_RATE): Blob {
  const n = audio.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const s = (o: number, t: string): void => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
  s(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); s(8, 'WAVE'); s(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); s(36, 'data'); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, audio[i]!)) * 0x7fff, true);
  return new Blob([buf], { type: 'audio/wav' });
}
