// The page side of the CPU TTS (cpu-worker.ts): fetches cpu/tts-cpu.wasm and the model files, hands them to the worker
// and asks it per sentence. Same speak / analyze options and results as WgslOwnTts (tts-own-device.ts).
import type { EmotionBlend } from './emotion.js';
import { modelFetch } from './model-fetch.js';
import { curvePoints } from './speak-options.js';
import type { OwnSpeakOptions, OwnSpeakResult } from './tts-own-device.js';
import type { DictWord, OwnDictEntry, OwnMeta } from './tts-own-prosody.js';


export type CpuSpeakOptions = Omit<OwnSpeakOptions, 'stream' | 'blendAt'> & { curve?: EmotionBlend[] };
export type CpuAnalysis = { norm: string; g2pText: string; unknown: string[]; g2p: { phones: string[]; marks: number[]; spans: [number, number][] }; phones: string[]; marks: number[]; words: DictWord[]; aligned: boolean };
export type CpuRequest =
  | { type: 'load'; id: number; wasm: ArrayBuffer; meta: unknown; bin: ArrayBuffer; nsfMeta: unknown; nsfBin: ArrayBuffer }
  | { type: 'speak'; id: number; text: string; options: CpuSpeakOptions }
  | { type: 'analyze'; id: number; text: string; dict: OwnDictEntry[] };
export type CpuReply =
  | { type: 'ready'; id: number }
  | { type: 'spoken'; id: number; result: OwnSpeakResult }
  | { type: 'analyzed'; id: number; result: CpuAnalysis }
  | { type: 'error'; id: number; message: string };

export class CpuTts {
  private seq = 0;
  private readonly waiting = new Map<number, { resolve: (r: CpuReply) => void; reject: (e: Error) => void }>();

  private constructor(private readonly worker: Worker, readonly meta: OwnMeta) {
    worker.onmessage = (e: MessageEvent<CpuReply>) => {
      const w = this.waiting.get(e.data.id);
      if (!w) return;
      this.waiting.delete(e.data.id);
      if (e.data.type === 'error') w.reject(new Error(e.data.message)); else w.resolve(e.data);
    };
    worker.onerror = (e) => { for (const w of this.waiting.values()) w.reject(new Error(`cpu tts worker: ${e.message}`)); this.waiting.clear(); };
  }

  private ask(m: CpuRequest, transfer: Transferable[] = []): Promise<CpuReply> {
    return new Promise((resolve, reject) => { this.waiting.set(m.id, { resolve, reject }); this.worker.postMessage(m, transfer); });
  }

  /** wasmUrl: cpu/tts-cpu.wasm; modelsUrl: the directory with <set>/own.* and <decoderSet>/nsf.*. */
  static async load(wasmUrl: string, modelsUrl: string, set: string, decoderSet: string): Promise<CpuTts> {
    const base = modelsUrl.replace(/\/$/, '');
    const got = async (r: Promise<Response>, what: string): Promise<Response> => { const x = await r; if (!x.ok) throw new Error(`cpu tts ${what}: ${x.status}`); return x; };
    const [wasm, meta, bin, nsfMeta, nsfBin] = await Promise.all([
      got(fetch(wasmUrl), 'tts-cpu.wasm').then((r) => r.arrayBuffer()),
      got(modelFetch(`${base}/${set}/own.json`), 'own.json').then((r) => r.json() as Promise<OwnMeta>),
      got(modelFetch(`${base}/${set}/own.bin`), 'own.bin').then((r) => r.arrayBuffer()),
      got(modelFetch(`${base}/${decoderSet}/nsf.json`), 'nsf.json').then((r) => r.json() as Promise<unknown>),
      got(modelFetch(`${base}/${decoderSet}/nsf.bin`), 'nsf.bin').then((r) => r.arrayBuffer()),
    ]);
    const c = new CpuTts(new Worker(new URL('./cpu-worker.js', import.meta.url), { type: 'module' }), meta);
    try {
      await c.ask({ type: 'load', id: ++c.seq, wasm, meta, bin, nsfMeta, nsfBin }, [wasm, bin, nsfBin]);
    } catch (err) {
      c.destroy();
      throw err;
    }
    return c;
  }

  async speak(text: string, o: Omit<OwnSpeakOptions, 'stream'>): Promise<OwnSpeakResult> {
    const { blendAt, ...rest } = o;
    const options: CpuSpeakOptions = blendAt ? { ...rest, curve: curvePoints(blendAt) } : rest;
    const r = await this.ask({ type: 'speak', id: ++this.seq, text, options });
    if (r.type !== 'spoken') throw new Error('cpu tts: unexpected reply');
    return r.result;
  }

  async analyze(text: string, dict: readonly OwnDictEntry[] = []): Promise<CpuAnalysis> {
    const r = await this.ask({ type: 'analyze', id: ++this.seq, text, dict: [...dict] });
    if (r.type !== 'analyzed') throw new Error('cpu tts: unexpected reply');
    return r.result;
  }

  destroy(): void { this.worker.terminate(); }
}
