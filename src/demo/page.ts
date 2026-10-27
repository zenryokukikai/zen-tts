// The demo page (demo/index.html): free text, the emotion sliders (a mix, or 「文の途中で変化」 from a start mix to an end
// mix with a curve shape over a range of the text), 抑揚の強さ, and where the voice is made (WebGPU or CPU). The
// dictionary is this browser's (engine.ts), shared with the intonation panel (tune.ts). Results of the last run are on
// window.zenTtsDemo (automated check). Nothing is sent anywhere.
import { readyAudioContext } from './audio.js';
import { emotionNames, SAMPLE_RATE, type Backend, type EmotionBlend, type EmotionName } from '../index.js';
import { isTtsDemoCurve, ttsDemoBlend, ttsDemoCurveBlend, ttsDemoCurveShape, TTS_DEMO_CURVE_LABEL, TTS_DEMO_CURVES, TTS_DEMO_MAX_TEXT, type TtsDemoCurve } from '../runtime/emotion-curve.js';
import { browserDict, engineLoaded, loadEngine, speakQueued, splitSentences } from './engine.js';

const LABEL: Record<EmotionName, string> = { neutral: '平静', happy: '喜び', sad: '悲しみ', angry: '怒り', apologetic: '申し訳なさ' };
const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const textEl = $<HTMLTextAreaElement>('text');
const statusEl = $('status');
const speakBtn = $<HTMLButtonElement>('speak');
const stopBtn = $<HTMLButtonElement>('stop');
const dlBtn = $<HTMLButtonElement>('download');
const strengthEl = $<HTMLInputElement>('strength');
const strengthOut = $('strength-out');
const mixEl = $('mix');
const curveModeEl = $<HTMLInputElement>('curve-mode');
const curveCard = $('curve-card');
const curvesEl = $('curves');
const previewEl = $('curve-preview');
const curveFromEl = $<HTMLInputElement>('curve-from');
const curveToEl = $<HTMLInputElement>('curve-to');
const samplesEl = $('samples');
const backendEl = $('backend');
textEl.maxLength = TTS_DEMO_MAX_TEXT;

interface Slider { range: HTMLInputElement; num: HTMLInputElement }
/** One set of five emotion sliders in the box (id prefix: emo- for the start / only set, emo-end- for the curve's end). */
function sliderSet(box: string, prefix: string, initial: EmotionName): Slider[] {
  return emotionNames.map((k) => {
    const row = document.createElement('div');
    row.className = 'emo';
    const v = k === initial ? 100 : 0;
    row.innerHTML = `<label for="${prefix}${k}">${LABEL[k]}</label><input type="range" id="${prefix}${k}" min="0" max="100" step="1" value="${v}"><span class="num"><input type="number" inputmode="numeric" min="0" max="100" step="1" value="${v}" aria-label="${LABEL[k]}（%）">%</span>`;
    $(box).append(row);
    const range = row.querySelector<HTMLInputElement>('input[type=range]')!;
    const num = row.querySelector<HTMLInputElement>('input[type=number]')!;
    range.addEventListener('input', () => { num.value = range.value; showMix(); });
    num.addEventListener('input', () => { const n = Math.min(100, Math.max(0, Number(num.value) || 0)); range.value = String(n); showMix(); });
    num.addEventListener('change', () => { num.value = range.value; });
    return { range, num };
  });
}
const sliders = sliderSet('emotions', 'emo-', 'neutral');
const endSliders = sliderSet('emotions-end', 'emo-end-', 'angry');
const percents = (): number[] => sliders.map((s) => Number(s.range.value));
const endPercents = (): number[] => endSliders.map((s) => Number(s.range.value));
const pct = (el: HTMLInputElement): number => Math.min(100, Math.max(0, Number(el.value) || 0)) / 100;
function curveSettings(): { start: EmotionBlend; end: EmotionBlend; curve: TtsDemoCurve; from: number; to: number } {
  return { start: ttsDemoBlend(percents()), end: ttsDemoBlend(endPercents()), curve: chosenCurve(), from: pct(curveFromEl), to: pct(curveToEl) };
}

/** The page's voice settings now for the part a..b (0..1 by characters) of the text: the emotion mix (or, with 「文の途中で
 *  変化」, that part of the curve) and 抑揚の強さ. key changes whenever any of them changes. Used by 「話す」 and by the
 *  intonation panel (tune.ts), which listens to the 'tts-demo-voice' event on document. */
export function pageVoice(a = 0, b = 1): { blend: EmotionBlend; strength: number; blendAt?: (x: number) => EmotionBlend; key: string } {
  const strength = Number(strengthEl.value);
  if (!curveModeEl.checked) {
    const blend = ttsDemoBlend(percents());
    return { blend, strength, key: JSON.stringify({ blend, strength, backend: pageBackend() }) };
  }
  const cs = curveSettings();
  const whole = ttsDemoCurveBlend(cs.start, cs.end, cs.curve, cs.from, cs.to);
  const blendAt = (x: number): EmotionBlend => whole(a + x * (b - a));
  return { blend: blendAt(0.5), strength, blendAt, key: JSON.stringify({ cs, strength, a, b, backend: pageBackend() }) };
}
const voiceChanged = (): void => { document.dispatchEvent(new Event('tts-demo-voice')); };

/** Where the voice is made (the page's radio: auto / webgpu / cpu). */
export function pageBackend(): Backend {
  const v = backendEl.querySelector<HTMLInputElement>('input[name=backend]:checked')?.value;
  return v === 'webgpu' || v === 'cpu' ? v : 'auto';
}
const BACKEND_LABEL = { webgpu: 'GPU（WebGPU）', cpu: 'CPU（WebAssembly）' } as const;
/** The backend the loaded engine really runs on, shown next to the radio. */
export function showBackend(): void {
  const t = engineLoaded();
  $('backend-now').textContent = t ? `いまは ${BACKEND_LABEL[t.backend]} で作っています。` : '';
}
document.querySelectorAll<HTMLInputElement>('input[name=voice]').forEach((el) => el.addEventListener('change', () => {
  $('backend-now').textContent = '';
  void loadEngine(pageBackend()).then(showBackend, () => undefined);
  voiceChanged();
}));
backendEl.querySelectorAll<HTMLInputElement>('input[name=backend]').forEach((el) => el.addEventListener('change', () => {
  $('backend-now').textContent = '';
  void loadEngine(pageBackend()).then(showBackend, () => undefined);
  voiceChanged();
}));
if (!('gpu' in navigator)) {
  const g = backendEl.querySelector<HTMLInputElement>('input[value=webgpu]');
  if (g) { g.disabled = true; g.closest('label')!.title = 'このブラウザでは WebGPU が使えません'; }
}
function showMode(): void {
  const curve = curveModeEl.checked;
  curveCard.hidden = !curve;
  $('start-title').textContent = curve ? 'はじめの感情の配合' : '感情の配合';
  showMix();
}
document.querySelectorAll<HTMLInputElement>('input[name=mode]').forEach((el) => el.addEventListener('change', showMode));
[curveFromEl, curveToEl].forEach((el) => el.addEventListener('input', showMix));

const SVG = 'http://www.w3.org/2000/svg';
/** SVG path of a curve shape drawn in a w x h box (0 at the bottom = はじめ, 1 at the top = おわり), over x in 0..1 of the
 * text with the change range a..b. */
function curvePath(curve: TtsDemoCurve, w: number, h: number, pad: number, a: number, b: number): string {
  const n = 96;
  const pts: string[] = [];
  for (let i = 0; i <= n; i++) {
    const x = i / n;
    const u = b <= a ? (x < a ? 0 : 1) : (x - a) / (b - a);
    const y = ttsDemoCurveShape(curve, u);
    pts.push(`${(pad + x * (w - 2 * pad)).toFixed(1)},${(h - pad - y * (h - 2 * pad)).toFixed(1)}`);
  }
  return 'M' + pts.join('L');
}
/** One radio per curve shape, with its thumbnail. */
curvesEl.replaceChildren(...TTS_DEMO_CURVES.map((c) => {
  const label = document.createElement('label');
  label.className = 'curve-opt';
  label.innerHTML = `<input type="radio" name="curve" value="${c}"${c === 'ease' ? ' checked' : ''}><svg viewBox="0 0 60 36" aria-hidden="true"><rect x="0.5" y="0.5" width="59" height="35" rx="5"/><path d="${curvePath(c, 60, 36, 5, 0, 1)}"/></svg><span></span>`;
  label.querySelector('span')!.textContent = TTS_DEMO_CURVE_LABEL[c];
  label.querySelector('input')!.addEventListener('change', showMix);
  return label;
}));
function chosenCurve(): TtsDemoCurve {
  const v = curvesEl.querySelector<HTMLInputElement>('input[name=curve]:checked')?.value;
  return isTtsDemoCurve(v) ? v : 'ease';
}
/** The preview: the chosen shape over the whole text, with the change range shaded. */
function drawPreview(curve: TtsDemoCurve, a: number, b: number): void {
  const w = 320, h = 120, pad = 10;
  const sx = (x: number): number => pad + x * (w - 2 * pad);
  const svg = document.createElementNS(SVG, 'svg');
  svg.setAttribute('viewBox', `0 0 ${w} ${h + 18}`);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', `${TTS_DEMO_CURVE_LABEL[curve]}の変わり方`);
  svg.innerHTML = `<rect class="range" x="${sx(a).toFixed(1)}" y="${pad}" width="${Math.max(0, sx(b) - sx(a)).toFixed(1)}" height="${h - 2 * pad}"/>`
    + `<line class="axis" x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}"/><line class="axis" x1="${pad}" y1="${pad}" x2="${w - pad}" y2="${pad}"/>`
    + `<path class="curve" d="${curvePath(curve, w, h, pad, a, b)}"/>`
    + `<text x="${pad + 2}" y="${h - pad - 4}">はじめ</text><text x="${pad + 2}" y="${pad + 13}">おわり</text>`
    + `<text x="${pad}" y="${h + 14}">文の先頭</text><text x="${w - pad}" y="${h + 14}" text-anchor="end">文の最後</text>`;
  previewEl.replaceChildren(svg);
}
const mixText = (b: EmotionBlend): string => emotionNames.filter((k) => b[k] > 0.0049).map((k) => `${LABEL[k]} ${Math.round(b[k] * 100)}%`).join('・');
function showMix(): void {
  mixEl.textContent = '実際の配合：' + mixText(ttsDemoBlend(percents()));
  $('mix-end').textContent = '実際の配合：' + mixText(ttsDemoBlend(endPercents()));
  const c = curveSettings();
  const a = Math.min(c.from, c.to), b = Math.max(c.from, c.to);
  $('curve-plan').textContent = `文の ${Math.round(a * 100)}%〜${Math.round(b * 100)}% の間を「${TTS_DEMO_CURVE_LABEL[c.curve]}」で変えます。グラフの下が「はじめ」、上が「おわり」の配合です。`;
  drawPreview(c.curve, a, b);
  voiceChanged();
}
showMix();
strengthEl.addEventListener('input', () => { strengthOut.textContent = `${Number(strengthEl.value).toFixed(1)} 倍`; voiceChanged(); });
$('reset').addEventListener('click', () => {
  sliders.forEach((s, i) => { s.range.value = s.num.value = i === 0 ? '100' : '0'; });
  endSliders.forEach((s, i) => { s.range.value = s.num.value = emotionNames[i] === 'angry' ? '100' : '0'; });
  curvesEl.querySelector<HTMLInputElement>('input[value=ease]')!.checked = true; curveFromEl.value = '20'; curveToEl.value = '80';
  strengthEl.value = '1'; strengthOut.textContent = '1.0 倍';
  showMix();
});

function setStatus(t: string, kind: 'info' | 'error' = 'info'): void { statusEl.textContent = t; statusEl.dataset.kind = kind; }

let audioCtx: AudioContext | undefined;
let source: AudioBufferSourceNode | undefined;
let lastWav: Blob | undefined;
let lastWavUrl: string | undefined;
let run = 0;
let resultSeq = 0;
// iOS: after the page was hidden (the WAV preview, another app) its AudioContext can report 'running' and still play
// nothing; the next tap then gets a new context (the old one is closed).
let hiddenSincePlay = false;
document.addEventListener('visibilitychange', () => { if (document.hidden) hiddenSincePlay = true; });

function pcmOf(audio: Float32Array): Int16Array {
  return Int16Array.from(audio, (v) => Math.max(-32768, Math.min(32767, Math.round((Number.isFinite(v) ? v : 0) * 32768))));
}
function wavOf(pcm: Int16Array, sr: number): Blob {
  const h = new DataView(new ArrayBuffer(44));
  const s = (o: number, t: string): void => { for (let i = 0; i < t.length; i++) h.setUint8(o + i, t.charCodeAt(i)); };
  s(0, 'RIFF'); h.setUint32(4, 36 + pcm.byteLength, true); s(8, 'WAVE'); s(12, 'fmt '); h.setUint32(16, 16, true);
  h.setUint16(20, 1, true); h.setUint16(22, 1, true); h.setUint32(24, sr, true); h.setUint32(28, sr * 2, true);
  h.setUint16(32, 2, true); h.setUint16(34, 16, true); s(36, 'data'); h.setUint32(40, pcm.byteLength, true);
  return new Blob([h.buffer, pcm.buffer as ArrayBuffer], { type: 'audio/wav' });
}

function stop(): void {
  run++;
  try { source?.stop(); } catch { /* already ended */ }
  source = undefined;
  speakBtn.disabled = false;
  stopBtn.disabled = true;
}

async function speak(): Promise<void> {
  const text = textEl.value.trim();
  if (!text) { setStatus('話させたい文を入力してください。', 'error'); return; }
  stop();
  const my = run;
  speakBtn.disabled = true;
  stopBtn.disabled = false;
  dlBtn.disabled = true;
  // inside the tap: iOS plays only from a context made / resumed here
  if (hiddenSincePlay && audioCtx) { void audioCtx.close().catch(() => {}); audioCtx = undefined; }
  hiddenSincePlay = false;
  const ready = await readyAudioContext(audioCtx, () => new AudioContext());
  audioCtx = ready.ctx;
  const blend = ttsDemoBlend(percents());
  const strength = Number(strengthEl.value);
  try {
    const backend = pageBackend();
    if (!engineLoaded()) setStatus('声のモデルを読み込んでいます…（約 31MB、初回だけ）');
    await loadEngine(backend);
    showBackend();
    if (my !== run) return;
    setStatus('声を作っています…');
    const r = { chunks: splitSentences(text).map((t) => ({ text: t, speech: t })) };
    if (r.chunks.length === 0) throw new Error('読める文字がありません');
    samplesEl.replaceChildren();
    const t0 = performance.now();
    // the curve runs over the whole text: each sentence chunk gets its part of it (by character position)
    const cs = curveModeEl.checked ? curveSettings() : undefined;
    const whole = cs ? ttsDemoCurveBlend(cs.start, cs.end, cs.curve, cs.from, cs.to) : undefined;
    const lens = r.chunks.map((c) => [...c.text].length);
    const total = lens.reduce((a, n) => a + n, 0);
    let before = 0;
    const spans = lens.map((n) => { const s0 = before / total; before += n; return [s0, before / total] as const; });
    const made = await Promise.all(r.chunks.map((c, index) => {
      const [a, b] = spans[index]!;
      const v = pageVoice(a, b);
      return speakQueued(backend, c.speech, { blend: v.blend, strength: v.strength, ...(v.blendAt ? { blendAt: v.blendAt } : {}) });
    }));
    if (my !== run) return;
    const sr = SAMPLE_RATE;
    const parts = made.map((m) => pcmOf(m.audio));
    const pcm = new Int16Array(parts.reduce((n, p) => n + p.length, 0));
    let o = 0;
    for (const p of parts) { pcm.set(p, o); o += p.length; }
    let peak = 0;
    for (const v of pcm) peak = Math.max(peak, Math.abs(v) / 32768);
    const sec = pcm.length / sr;
    if (pcm.length === 0 || peak < 1e-4) throw new Error('声が無音になりました。もう一度「話す」を押してください');
    lastWav = wavOf(pcm, sr);
    // the URL lives until the next WAV: iOS opens the download in a preview that reads it later
    if (lastWavUrl) URL.revokeObjectURL(lastWavUrl);
    lastWavUrl = URL.createObjectURL(lastWav);
    dlBtn.disabled = false;
    const curveLog = cs && whole ? { ...cs, samples: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1].map((x) => ({ x, blend: whole(x) })) } : undefined;
    if (curveLog) samplesEl.replaceChildren(...curveLog.samples.filter((_, i) => i % 2 === 0).map((q) => {
      const li = document.createElement('li');
      li.className = 'made';
      li.innerHTML = '<span class="ptext"></span><span class="pmix"></span>';
      li.querySelector('.ptext')!.textContent = `文の ${Math.round(q.x * 100)}% の位置`;
      li.querySelector('.pmix')!.textContent = mixText(q.blend);
      return li;
    }));
    const makeMs = made.reduce((n, m) => n + m.timing.totalMs, 0);
    let sq = 0;
    for (const v of pcm) sq += (v / 32768) ** 2;
    const result = { seq: ++resultSeq, backend: engineLoaded()?.backend ?? backend, voice: engineLoaded()?.voice, mode: cs ? 'curve' : 'whole', curve: curveLog, segments: cs ? segmentLog(made.map(() => sr), parts, spans, whole!) : undefined, text, blend, strength, chunks: made.length, browserDictEntries: browserDict().length, dictWords: made.reduce((n, m) => n + m.words.length, 0), phones: made.map((m) => m.phones.join(' ')).join(' | '), sec: +sec.toFixed(2), peak: +peak.toFixed(3), rms: +Math.sqrt(sq / Math.max(1, pcm.length)).toFixed(4), makeMs: Math.round(makeMs), rtf: +(makeMs / 1000 / sec).toFixed(3), wallMs: Math.round(performance.now() - t0), played: false };
    (window as Window & { zenTtsDemo?: unknown }).zenTtsDemo = result;
    const buf = audioCtx.createBuffer(1, pcm.length, sr);
    buf.getChannelData(0).set(Float32Array.from(pcm, (v) => v / 32768));
    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(audioCtx.destination);
    source = src;
    src.onended = () => {
      if (source !== src) return;
      result.played = true;
      source = undefined;
      speakBtn.disabled = false;
      stopBtn.disabled = true;
      setStatus(`再生しました（${sec.toFixed(1)} 秒）。`);
    };
    setStatus(`再生しています…（${sec.toFixed(1)} 秒の声を ${(makeMs / 1000).toFixed(2)} 秒で作りました。${BACKEND_LABEL[result.backend as 'webgpu' | 'cpu']}）`);
    src.start();
  } catch (err) {
    if (my !== run) return;
    const m = err instanceof Error ? err.message : String(err);
    setStatus(/WebGPU|adapter/i.test(m) ? 'この端末・ブラウザでは WebGPU が使えません。「作る場所」で CPU を選んでください。' : `声を作れませんでした：${m}`, 'error');
    speakBtn.disabled = false;
    stopBtn.disabled = true;
  }
}

/** Per sentence chunk: where it sits in the text, the blend the curve gives its middle, and its loudness (headless check). */
function segmentLog(srs: number[], parts: Int16Array[], spans: (readonly [number, number])[], whole: (x: number) => EmotionBlend): { from: number; to: number; sec: number; peak: number; mid: EmotionBlend }[] {
  return parts.map((p, i) => {
    let peak = 0;
    for (const v of p) peak = Math.max(peak, Math.abs(v) / 32768);
    const [a, b] = spans[i]!;
    return { from: +a.toFixed(3), to: +b.toFixed(3), sec: +(p.length / srs[i]!).toFixed(2), peak: +peak.toFixed(3), mid: whole((a + b) / 2) };
  });
}

speakBtn.addEventListener('click', () => { void speak(); });
stopBtn.addEventListener('click', () => { stop(); setStatus('止めました。'); });
dlBtn.addEventListener('click', () => {
  if (!lastWav || !lastWavUrl) return;
  const a = document.createElement('a');
  a.href = lastWavUrl;
  a.download = 'zen-tts.wav';
  a.click();
});
setStatus('文を入力して「話す」を押してください。');
