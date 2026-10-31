// The demo's TTS (shared by 「話す」 and the intonation panel), the sentence split, and the dictionary kept in this
// browser only (localStorage). Nothing is sent anywhere: the model files (the voice's .ztm from the CDN, the G2P) are the only downloads.
import { ZenTts, ztmUrl, type Backend, type DictEntry, type EngineSpeakOptions, type SpeakResult, type VoiceId } from '../index.js';
import { BROWSER_DICT_STORAGE_KEY, BrowserDictError, dictFile, parseDictFile } from './dictionary.js';

/** iOS Safari (16.4+): the page's voice as media playback. Left at 'auto', a page that only uses Web Audio plays in the
 *  ambient category, and the voice can be missing from a screen recording. Set before any AudioContext is made. */
(() => {
  const session = (navigator as Navigator & { audioSession?: { type: string } }).audioSession;
  if (session) { try { session.type = 'playback'; } catch { /* not settable */ } }
})();

let engine: Promise<ZenTts> | undefined;
let engineKey: string | undefined;
let loaded: ZenTts | undefined;
export const engineLoaded = (): ZenTts | undefined => loaded;

/** The page's voice (the 声 radio: marine / hashiro / kai / kikka). */
export function pageVoiceId(): VoiceId {
  const v = document.querySelector<HTMLInputElement>('input[name=voice]:checked')?.value;
  return v === 'hashiro' || v === 'kai' || v === 'kikka' ? v : 'marine';
}

/** The TTS on the chosen backend with the page's voice (loaded once per backend and voice; a failed load is tried again
 *  on the next call). */
export function loadEngine(backend: Backend): Promise<ZenTts> {
  // ?model=<url of a .ztm>: that voice instead of the 声 radio's (docs/ztm-format.md)
  const model = new URLSearchParams(location.search).get('model') ?? undefined;
  const voice = pageVoiceId();
  const key = `${backend}/${model ?? voice}`;
  if (engine && engineKey !== key) {
    const old = engine;
    engine = undefined;
    loaded = undefined;
    void queue.then(() => old).then((t) => t.destroy(), () => undefined);
  }
  engineKey = key;
  // the voice's .ztm from the release's CDN (ztmUrl), unless ?model= names another; the G2P from ../models/g2p
  engine ??= ZenTts.load({ modelsUrl: '../models', cpuWasmUrl: '../cpu/tts-cpu.wasm', backend, model: model ?? ztmUrl(voice) }).then((t) => { loaded = t; return t; });
  engine.catch(() => { engine = undefined; });
  return engine;
}

let queue: Promise<unknown> = Promise.resolve();
/** One utterance after another (the engine's buffers are shared). The dictionary is this browser's; over: one entry put
 *  over it for this run only (a 読み being tried in the intonation panel). */
export function speakQueued(backend: Backend, text: string, o: Omit<EngineSpeakOptions, 'dict' | 'settings'>, over?: DictEntry): Promise<SpeakResult & { sampleRate: number }> {
  const r = queue.then(async () => {
    const t = await loadEngine(backend);
    const dict = browserDict();
    return t.speakWith(text, { ...o, dict: over ? [...dict.filter((x) => x.surface !== over.surface), over] : dict });
  });
  queue = r.catch(() => undefined);
  return r;
}

/** The text cut into sentences at 。！？ (and line breaks), never at 、; closing brackets stay with their sentence. */
export function splitSentences(text: string): string[] {
  const out: string[] = [];
  let buf = text;
  let m: RegExpMatchArray | null;
  while ((m = buf.match(/^([\s\S]*?[。！？!?…\n]+[」』）)]*)/)) !== null) {
    const piece = m[1]!;
    buf = buf.slice(piece.length);
    const t = piece.trim();
    if (/[^\s、。！？!?…」』）)]/.test(t)) out.push(t);
    else if (out.length > 0) out[out.length - 1] += t;
  }
  const rest = buf.trim();
  if (/[^\s、。！？!?…」』）)]/.test(rest)) out.push(rest);
  return out;
}

// ---- the dictionary of this browser ----

let cache: DictEntry[] | undefined;
/** Set when the stored value could not be read (it is left as it is until the user saves). */
export let browserDictProblem: string | undefined;

export function browserDict(): DictEntry[] {
  if (cache) return cache;
  let raw: string | null = null;
  try { raw = localStorage.getItem(BROWSER_DICT_STORAGE_KEY); } catch { browserDictProblem = 'このブラウザでは辞書を保存できません（設定で保存が止められています）'; }
  try {
    cache = raw ? parseDictFile(raw) : [];
  } catch (err) {
    browserDictProblem = `保存されている辞書を読めませんでした：${err instanceof Error ? err.message : String(err)}`;
    cache = [];
  }
  return cache;
}

/** Saves the entries in this browser (throws BrowserDictError when the browser does not keep it). */
export function saveBrowserDict(entries: DictEntry[]): void {
  try {
    localStorage.setItem(BROWSER_DICT_STORAGE_KEY, JSON.stringify(dictFile(entries)));
  } catch {
    throw new BrowserDictError('このブラウザに辞書を保存できませんでした（容量か設定を確かめてください）');
  }
  cache = entries;
  browserDictProblem = undefined;
}
