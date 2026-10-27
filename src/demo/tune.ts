// The demo's intonation panel. Everything runs in this browser (engine.ts: the same TTS as 「話す」, WebGPU or CPU):
//   「読みと高さを見る」: one sentence of the text read with the page's voice settings -> its sounds (kana, 高い / 低い,
//   where the pitch falls, accent phrases), the pitch line and the lengths; that voice is 「直す前」.
//   Edits: 下がる位置 per phrase, 高さ per sound (drag a point; or tap a sound and use − / ＋), 長さ per sound; a 読み for
//   a word (表記 + 読み, tried before it is saved). 「直して読む」 makes 「直した後」. 「元に戻す」 clears the edits.
//   The page's emotion: when the mix, the curve, 抑揚の強さ or the backend change while a sentence is open, it is read
//   again with them and the edits (relative: semitones added, length factors, 下がる位置) are kept on the new prediction.
//   「辞書に登録」: the word's 読み and its intonation into this browser's dictionary (localStorage): its pitch shape
//   (relative to the word's own mean, scaled to the emotion's range when read, so it keeps its form under any emotion
//   and position), its lengths and, when the word's phrase was given one, its 下がる位置. Every later voice on this page
//   uses it. 「読みを試す」 and the list's 「直す」 open the word alone in this panel. The list can be written to a file and
//   read back from one (to move it to another browser). The dictionary is never sent anywhere.
import { readyAudioContext } from './audio.js';
import {
  BrowserDictError, contourSketch, dictFile, emptySoundEdits, findReadingUnits, hasEdits, mergeDictEntries, parseDictFile, phraseAccents, soundPitch, soundUnits,
  speakOptionsOf, validateDictEntry, wordEntryOf, type SoundEdits, type SoundUnit,
} from './dictionary.js';
import { pitchRangeOf, SAMPLE_RATE, type DictEntry as OwnDictEntry, type EngineSpeakOptions as OwnSpeakOptions, type SpeakResult as OwnSpeakResult } from '../index.js';
import { pageBackend, pageVoice, showBackend } from './page.js';
import { browserDict, browserDictProblem, engineLoaded, loadEngine, saveBrowserDict, speakQueued, splitSentences } from './engine.js';

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;
const COL = 44;
const PLOT_H = 170;
const PAD = 16;
const PITCH_STEP = 0.5;
const LENGTH_STEPS = [0.5, 0.67, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 2];

/** What the panel reads: a sentence of the text (span: its part of the text, for the curve) or one word alone. */
interface Target { speech: string; span: [number, number]; word?: string }

interface Analysis extends Target {
  /** a 読み tried before it is saved (applies to this reading only) */
  trial?: OwnDictEntry;
  voice: ReturnType<typeof pageVoice>;
  units: SoundUnit[];
  phones: string[];
  marks: number[];
  accents: number[];
  before: OwnSpeakResult;
  beforePitch: (number | null)[];
  after?: OwnSpeakResult | undefined;
  afterPitch?: (number | null)[] | undefined;
}

let cur: Analysis | undefined;
let edits: SoundEdits = { accent: [], pitch: [], length: [] };
let selected = -1;
let sentences: { text: string; speech: string }[] = [];

function tuneStatus(t: string, kind: 'info' | 'error' = 'info'): void {
  const el = $('tune-status');
  el.textContent = t;
  el.dataset.kind = kind;
}

/** The speak options of the page's voice now (tts-demo.ts pageVoice) for that target. */
function voiceOf(t: Target): { voice: ReturnType<typeof pageVoice>; o: Pick<OwnSpeakOptions, 'blend' | 'strength' | 'blendAt'> } {
  const voice = pageVoice(t.span[0], t.span[1]);
  return { voice, o: { blend: voice.blend, strength: voice.strength, ...(voice.blendAt ? { blendAt: voice.blendAt } : {}) } };
}

// ------------------------------------------------------------ audio

let audioCtx: AudioContext | undefined;
let playing: AudioBufferSourceNode | undefined;
async function play(audio: Float32Array): Promise<void> {
  try { playing?.stop(); } catch { /* ended */ }
  const r = await readyAudioContext(audioCtx, () => new AudioContext());
  audioCtx = r.ctx;
  const buf = r.ctx.createBuffer(1, audio.length, SAMPLE_RATE);
  buf.getChannelData(0).set(audio);
  const s = r.ctx.createBufferSource();
  s.buffer = buf;
  s.connect(r.ctx.destination);
  s.start();
  playing = s;
}

// ------------------------------------------------------------ analysis

/** The sentence chosen in the panel (the text is cut into sentences as 「話す」 does), with its part of the text. */
async function chosenSentence(): Promise<Target | undefined> {
  const text = ($('text') as HTMLTextAreaElement).value.trim();
  if (!text) return undefined;
  sentences = splitSentences(text).map((t) => ({ text: t, speech: t }));
  const sel = $('tune-sentence') as HTMLSelectElement;
  const keep = Number(sel.value) || 0;
  sel.replaceChildren(...sentences.map((s, i) => { const o = document.createElement('option'); o.value = String(i); o.textContent = s.text; return o; }));
  sel.value = String(Math.min(keep, Math.max(0, sentences.length - 1)));
  $('tune-sentence-row').hidden = sentences.length < 2;
  const i = Number(sel.value);
  const s = sentences[i];
  if (!s) return undefined;
  // the same split of the text as 「話す」 (by characters), for that sentence's part of the curve
  const lens = sentences.map((c) => [...c.text].length);
  const total = lens.reduce((a, n) => a + n, 0) || 1;
  const a = lens.slice(0, i).reduce((x, n) => x + n, 0) / total;
  return { speech: s.speech, span: [a, a + lens[i]! / total] };
}

/** keep: edits to put on the new reading (the voice changed); kept when the sounds are the same. */
interface AnalyzeOptions { trial?: OwnDictEntry; target?: Target; keep?: SoundEdits; remake?: boolean; reason?: 'button' | 'try' | 'entry' | 'voice' }

async function analyze(opt: AnalyzeOptions = {}): Promise<void> {
  const btn = $('tune-analyze') as HTMLButtonElement;
  btn.disabled = true;
  const quiet = opt.reason === 'voice';
  if (!quiet) void readyAudioContext(audioCtx, () => new AudioContext()).then((r) => { audioCtx = r.ctx; });
  try {
    if (!engineLoaded()) tuneStatus('声のモデルを読み込んでいます…（約 31MB、初回だけ）');
    const s = opt.target ?? await chosenSentence();
    if (!s) { tuneStatus('上の「話させたい文」に文を入れてください。', 'error'); return; }
    if (s.word) $('tune-sentence-row').hidden = true;
    const e = await loadEngine(pageBackend());
    showBackend();
    tuneStatus(quiet ? '感情の設定が変わったので、今の設定で読み直しています…' : '読みと高さを調べています…');
    const { voice, o } = voiceOf(s);
    const before = await speakQueued(pageBackend(), s.speech, o, opt.trial);
    const punct = new Set(e.meta.punctuation);
    const units = soundUnits(before.phones, before.marks, e.meta.kana, punct);
    if (!units.some((u) => !u.punct)) throw new Error('読める音がありません');
    const same = !!opt.keep && !!cur && cur.phones.length === before.phones.length && cur.phones.every((p, i) => p === before.phones[i]);
    cur = { ...s, ...(opt.trial ? { trial: opt.trial } : {}), voice, units, phones: before.phones, marks: before.marks, accents: phraseAccents(units, before.phones, before.marks), before, beforePitch: soundPitch(units, before.f0) };
    edits = same ? opt.keep! : emptySoundEdits(units);
    if (!same) selected = -1;
    draw();
    $('tune-body').hidden = false;
    for (const id of ['tune-play-before', 'tune-make', 'tune-reset', 'dict-save']) ($(id) as HTMLButtonElement).disabled = false;
    ($('tune-play-after') as HTMLButtonElement).disabled = true;
    const sec = before.audio.length / SAMPLE_RATE;
    const what = s.word ? `「${s.word}」だけを読みました` : '読みました';
    let msg = quiet
      ? (opt.keep && hasEdits(opt.keep) ? (same ? '感情の設定が変わったので、今の設定で読み直しました。直したところはそのままです。' : '感情の設定が変わったので、今の設定で読み直しました。読みが変わったため、直したところは元に戻しました。') : '感情の設定が変わったので、今の設定で読み直しました。')
      : `${what}（${sec.toFixed(1)} 秒）。点を動かすか、音をタップして直し、「直して読む」を押してください。`;
    if (opt.trial) msg += `\n「${opt.trial.surface}」を「${opt.trial.reading}」と読ませています（まだ辞書には入っていません）。`;
    if (s.word && !quiet) msg += '\n直したら「辞書に登録」で、読みと高さ・長さを保存します。';
    tuneStatus(msg);
    document.body.dataset.tuneAnalyzed = String(Date.now());
    if (same && opt.remake && hasEdits(edits)) await makeEdited(false);
    else if (!quiet) void play(before.audio);
  } catch (err) {
    tuneStatus(`読めませんでした：${err instanceof Error ? err.message : String(err)}`, 'error');
  } finally {
    btn.disabled = false;
  }
}

async function makeEdited(sound = true): Promise<void> {
  if (!cur) return;
  const btn = $('tune-make') as HTMLButtonElement;
  btn.disabled = true;
  tuneStatus('直した声を作っています…');
  try {
    const o = speakOptionsOf(cur.units, cur.phones, cur.marks, edits);
    const { voice, o: v } = voiceOf(cur);
    if (voice.key !== cur.voice.key) throw new Error('感情の設定が変わりました。読み直すまで待ってから押してください。');
    const r = await speakQueued(pageBackend(), cur.speech, { ...v, ...o }, cur.trial);
    if (r.phones.length !== cur.phones.length) throw new Error('文が変わりました。もう一度「読みと高さを見る」を押してください。');
    cur.after = r;
    cur.afterPitch = soundPitch(cur.units, r.f0);
    ($('tune-play-after') as HTMLButtonElement).disabled = false;
    draw();
    tuneStatus(`直した声を作りました（${(r.audio.length / SAMPLE_RATE).toFixed(1)} 秒）。「直す前」「直した後」で聞き比べられます。`);
    document.body.dataset.tuneMade = String(Date.now());
    if (sound) await play(r.audio);
  } catch (err) {
    tuneStatus(`作れませんでした：${err instanceof Error ? err.message : String(err)}`, 'error');
  } finally {
    btn.disabled = false;
  }
}

// The page's emotion / curve / strength / backend changed (page.ts sends 'tts-demo-voice'): read the open sentence again with
// them, once the sliders rest, keeping the edits (and the made voice, made again) when the sounds are the same.
let voiceTimer: ReturnType<typeof setTimeout> | undefined;
let reading: Promise<void> = Promise.resolve();
document.addEventListener('tts-demo-voice', () => {
  if (!cur) return;
  clearTimeout(voiceTimer);
  voiceTimer = setTimeout(() => {
    reading = reading.then(async () => {
      if (!cur || pageVoice(cur.span[0], cur.span[1]).key === cur.voice.key) return;
      const target: Target = { speech: cur.speech, span: cur.span, ...(cur.word ? { word: cur.word } : {}) };
      await analyze({ ...(cur.trial ? { trial: cur.trial } : {}), target, keep: edits, remake: !!cur.after, reason: 'voice' });
    });
  }, 400);
});

function resetEdits(): void {
  if (!cur) return;
  edits = emptySoundEdits(cur.units);
  cur.after = undefined;
  cur.afterPitch = undefined;
  selected = -1;
  ($('tune-play-after') as HTMLButtonElement).disabled = true;
  draw();
  tuneStatus('直す前に戻しました。');
}

// ------------------------------------------------------------ drawing

/** The pitch of each sound now: the made voice's (after the edits once made, else before) + edits not made yet. */
function shownPitch(i: number): number | null {
  if (!cur) return null;
  const made = cur.afterPitch ?? cur.beforePitch;
  const v = made[i];
  if (v === null || v === undefined) return null;
  return cur.afterPitch ? v : v + (edits.pitch[i] ?? 0);
}

function sentenceMean(): number {
  const v = cur!.beforePitch.filter((x): x is number => x !== null);
  return v.reduce((a, b) => a + b, 0) / (v.length || 1);
}

function draw(): void {
  if (!cur) return;
  const units = cur.units;
  const w = units.length * COL;
  const mean = sentenceMean();
  const rel = (v: number | null): number | null => (v === null ? null : v - mean);
  const vals = units.flatMap((_, i) => [rel(cur!.beforePitch[i] ?? null), rel(shownPitch(i))]).filter((v): v is number => v !== null);
  const lo = Math.floor(Math.min(-4, ...vals) - 1);
  const hi = Math.ceil(Math.max(4, ...vals) + 1);
  const y = (v: number): number => PAD + ((hi - v) / (hi - lo)) * (PLOT_H - 2 * PAD);
  const x = (i: number): number => i * COL + COL / 2;
  const svg = $('tune-plot') as unknown as SVGSVGElement;
  svg.setAttribute('width', String(w));
  svg.setAttribute('viewBox', `0 0 ${w} ${PLOT_H}`);
  svg.dataset.lo = String(lo);
  svg.dataset.hi = String(hi);
  let h = '';
  for (let v = lo; v <= hi; v++) if (v % 2 === 0) h += `<line x1="0" x2="${w}" y1="${y(v)}" y2="${y(v)}" class="${v === 0 ? 'zero' : 'grid'}"/>`;
  units.forEach((u, i) => { if (i > 0 && u.phrase !== units[i - 1]!.phrase && !u.punct) h += `<line x1="${i * COL}" x2="${i * COL}" y1="0" y2="${PLOT_H}" class="phrase"/>`; });
  const line = (f: (i: number) => number | null, cls: string): string => {
    const pts = units.map((u, i) => { const v = rel(f(i)); return u.punct || v === null ? '' : `${x(i)},${y(v)}`; }).filter(Boolean);
    return `<polyline class="${cls}" points="${pts.join(' ')}"/>`;
  };
  h += line((i) => cur!.beforePitch[i] ?? null, 'before');
  h += line(shownPitch, 'now');
  units.forEach((u, i) => {
    const v = rel(shownPitch(i));
    if (u.punct || v === null) return;
    const edited = edits.pitch[i] !== 0;
    h += `<circle data-u="${i}" cx="${x(i)}" cy="${y(v)}" r="${i === selected ? 12 : 9}" class="pt${edited ? ' edited' : ''}${i === selected ? ' sel' : ''}"/>`;
  });
  svg.innerHTML = h;
  // sounds: kana, 高 / 低 (the made voice's), where the pitch falls (⌝), length
  const made = cur.after ?? cur.before;
  const tone = made.prosody.acc.tone;
  const nuc = made.prosody.acc.nucleus;
  const durs = units.map((u) => { let d = 0; for (let i = u.a; i < u.b; i++) d += made.dur[i + 1]!; return d; });
  const maxD = Math.max(1, ...durs);
  const row = $('tune-sounds');
  row.style.width = `${w}px`;
  row.innerHTML = units.map((u, i) => {
    const brk = i > 0 && u.phrase !== units[i - 1]!.phrase && !u.punct ? ' brk' : '';
    if (u.punct) return `<div class="snd punct${brk}"><b>${u.kana}</b></div>`;
    const high = tone[u.b]! > 0;
    const falls = nuc[u.b]! > 0;
    const len = edits.length[i] ?? 1;
    return `<button type="button" class="snd${brk}${i === selected ? ' sel' : ''}" data-u="${i}" aria-label="${u.kana}（${high ? '高い' : '低い'}）">`
      + `<b>${u.kana}${falls ? '<sup>⌝</sup>' : ''}</b><span class="hl${high ? ' h' : ''}">${high ? '高' : '低'}</span>`
      + `<i style="height:${Math.max(3, (durs[i]! / maxD) * 26)}px" class="${len !== 1 ? 'edited' : ''}"></i><small>${len !== 1 ? `×${len}` : ''}</small></button>`;
  }).join('');
  $('tune-reading').textContent = `読み：${units.map((u, i) => (i > 0 && u.phrase !== units[i - 1]!.phrase && !u.punct ? '｜' : '') + u.kana + (!u.punct && nuc[u.b]! > 0 ? '⌝' : '')).join('')}`;
  drawPhrases();
  drawSelected();
}

function drawPhrases(): void {
  if (!cur) return;
  const box = $('tune-phrases');
  const groups = new Map<number, number[]>();
  cur.units.forEach((u, i) => { if (!u.punct) groups.set(u.phrase, [...(groups.get(u.phrase) ?? []), i]); });
  box.replaceChildren(...[...groups.entries()].map(([p, idx]) => {
    const label = document.createElement('label');
    label.className = 'phr';
    const kana = idx.map((i) => cur!.units[i]!.kana).join('');
    const now = edits.accent[p] ?? cur!.accents[p] ?? 0;
    const opts = Array.from({ length: idx.length + 1 }, (_, k) => `<option value="${k}"${k === now ? ' selected' : ''}>${k === 0 ? '下がらない' : `${idx.slice(0, k).map((i) => cur!.units[i]!.kana).join('')}｜のあと`}</option>`).join('');
    label.innerHTML = `<span>${kana}</span><select data-phrase="${p}" aria-label="${kana} の下がる位置">${opts}</select>`;
    if (edits.accent[p] !== null && edits.accent[p] !== undefined) label.classList.add('edited');
    return label;
  }));
}

function drawSelected(): void {
  const box = $('tune-selected');
  if (!cur || selected < 0) { box.innerHTML = '<span class="hint">上の点を上下に動かすか、音をタップすると、ここで高さと長さを直せます。</span>'; return; }
  const u = cur.units[selected]!;
  const p = edits.pitch[selected] ?? 0;
  const l = edits.length[selected] ?? 1;
  box.innerHTML = `<b class="sk">${u.kana}</b>`
    + `<span class="ctl">高さ <button type="button" data-pitch="-1" aria-label="低く">−</button><output>${p > 0 ? '+' : ''}${p}</output><button type="button" data-pitch="1" aria-label="高く">＋</button></span>`
    + `<span class="ctl">長さ <button type="button" data-length="-1" aria-label="短く">−</button><output>×${l}</output><button type="button" data-length="1" aria-label="長く">＋</button></span>`;
}

function markEdited(): void {
  if (cur) { cur.after = undefined; cur.afterPitch = undefined; }
  ($('tune-play-after') as HTMLButtonElement).disabled = true;
  draw();
}

function setPitch(i: number, v: number): void {
  edits.pitch[i] = Math.max(-12, Math.min(12, Math.round(v / PITCH_STEP) * PITCH_STEP));
  markEdited();
}

function stepLength(i: number, dir: number): void {
  const c = edits.length[i] ?? 1;
  let k = LENGTH_STEPS.findIndex((s) => s >= c - 1e-6);
  if (k < 0) k = LENGTH_STEPS.length - 1;
  edits.length[i] = LENGTH_STEPS[Math.max(0, Math.min(LENGTH_STEPS.length - 1, k + dir))]!;
  markEdited();
}

function wire(): void {
  const svg = $('tune-plot') as unknown as SVGSVGElement;
  let drag: { u: number; y0: number; p0: number; moved: boolean } | undefined;
  svg.addEventListener('pointerdown', (e) => {
    const t = e.target as Element;
    if (!t.matches('circle.pt')) return;
    const u = Number(t.getAttribute('data-u'));
    selected = u;
    drag = { u, y0: e.clientY, p0: edits.pitch[u] ?? 0, moved: false };
    svg.setPointerCapture(e.pointerId);
    draw();
    e.preventDefault();
  });
  svg.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const lo = Number(svg.dataset.lo), hi = Number(svg.dataset.hi);
    const px = svg.getBoundingClientRect().height || PLOT_H;
    const st = (-(e.clientY - drag.y0) / (px * (1 - (2 * PAD) / PLOT_H))) * (hi - lo);
    if (Math.abs(e.clientY - drag.y0) > 3) drag.moved = true;
    if (drag.moved) setPitch(drag.u, drag.p0 + st);
  });
  const end = (): void => { drag = undefined; };
  svg.addEventListener('pointerup', end);
  svg.addEventListener('pointercancel', end);
  $('tune-sounds').addEventListener('click', (e) => {
    const b = (e.target as HTMLElement).closest<HTMLElement>('[data-u]');
    if (!b) return;
    selected = Number(b.dataset.u);
    draw();
  });
  $('tune-selected').addEventListener('click', (e) => {
    const t = (e.target as HTMLElement).closest<HTMLElement>('button');
    if (!t || selected < 0) return;
    if (t.dataset.pitch) setPitch(selected, (edits.pitch[selected] ?? 0) + Number(t.dataset.pitch) * PITCH_STEP);
    if (t.dataset.length) stepLength(selected, Number(t.dataset.length));
  });
  $('tune-phrases').addEventListener('change', (e) => {
    const t = e.target as HTMLSelectElement;
    if (!cur || t.dataset.phrase === undefined) return;
    const p = Number(t.dataset.phrase);
    const v = Number(t.value);
    edits.accent[p] = v === cur.accents[p] ? null : v;
    markEdited();
  });
  $('tune-analyze').addEventListener('click', () => void analyze());
  $('tune-make').addEventListener('click', () => void makeEdited());
  $('tune-reset').addEventListener('click', resetEdits);
  $('tune-play-before').addEventListener('click', () => { if (cur) void play(cur.before.audio); });
  $('tune-play-after').addEventListener('click', () => { if (cur?.after) void play(cur.after.audio); });
  $('dict-try').addEventListener('click', () => void tryReading());
  $('dict-save').addEventListener('click', () => void saveWord());
  $('dict-list').addEventListener('click', (e) => onListClick(e));
  $('dict-export').addEventListener('click', exportDict);
  $('dict-import').addEventListener('change', (e) => void importDict(e));
}

// ------------------------------------------------------------ the dictionary of this browser

function wordFields(): { surface: string; reading: string } {
  return { surface: ($('dict-surface') as HTMLInputElement).value.trim(), reading: ($('dict-reading') as HTMLInputElement).value.trim() };
}

/** 読みを試す: the word alone, read as typed (not saved yet), opened in the panel to set its intonation. */
async function tryReading(): Promise<void> {
  const w = wordFields();
  try {
    const e = validateDictEntry(w, '読みを試す');
    const saved = browserDict().find((x) => x.surface === e.surface);
    // the same reading as the saved entry: read with that entry (its intonation), not a bare reading
    const trial = saved && saved.reading === e.reading ? undefined : e;
    await analyze({ ...(trial ? { trial } : {}), target: { speech: e.surface, span: [0, 1], word: e.surface }, reason: 'try' });
  } catch (err) {
    tuneStatus(err instanceof Error ? err.message : String(err), 'error');
  }
}

/** The list's 「直す」: the entry's word alone in the panel, read with the entry (its reading and intonation). */
async function openEntry(e: OwnDictEntry): Promise<void> {
  ($('dict-surface') as HTMLInputElement).value = e.surface;
  ($('dict-reading') as HTMLInputElement).value = e.reading;
  $('tune-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
  await analyze({ target: { speech: e.surface, span: [0, 1], word: e.surface }, reason: 'entry' });
}

async function saveWord(): Promise<void> {
  if (!cur) return;
  const w = wordFields();
  try {
    validateDictEntry(w, '辞書に登録');
    const e = await loadEngine(pageBackend());
    const idx = findReadingUnits(cur.units, w.reading, e.meta.kana);
    if (!idx) {
      tuneStatus(`読み「${w.reading}」が今の文の読みの中にありません。読みを変えるときは、先に「読みを試す」を押してください。`, 'error');
      return;
    }
    if (hasEdits(edits) && !cur.after) await makeEdited();
    if (hasEdits(edits) && !cur.after) return;
    const made = cur.after ? cur.afterPitch! : cur.beforePitch;
    // the entry the word was read with (its lengths / 下がる位置 are in the made voice)
    const base = cur.trial?.surface === w.surface ? cur.trial : browserDict().find((x) => x.surface === w.surface);
    const entry = validateDictEntry(wordEntryOf(w.surface, w.reading, cur.units, idx, edits, made, pitchRangeOf(cur.voice.blend, e.meta.pitchRange), base));
    // a 下がる位置 set on a phrase that is exactly the word: kept as the word's accent type
    const phrases = [...new Set(idx.map((i) => cur!.units[i]!.phrase))];
    if (phrases.length === 1) {
      const p = phrases[0]!;
      const all = cur.units.flatMap((u, i) => (!u.punct && u.phrase === p ? [i] : []));
      if (edits.accent[p] !== null && edits.accent[p] !== undefined && all.length === idx.length && all.every((i, k) => i === idx[k])) entry.accent = edits.accent[p]!;
    }
    const r = mergeDictEntries(browserDict(), [entry]);
    saveBrowserDict(r.entries);
    drawList();
    const parts = ['読み', entry.contour ? '高さ' : '', entry.length ? '長さ' : '', entry.accent !== undefined ? '下がる位置' : ''].filter(Boolean).join('・');
    tuneStatus(`このブラウザの辞書に${r.replaced ? '上書き' : '登録'}しました：${entry.surface}（${entry.reading}）［${parts}］。これからこのページで読む声に使います。`);
    document.body.dataset.dictSaved = JSON.stringify(entry);
  } catch (err) {
    tuneStatus(err instanceof Error ? err.message : String(err), 'error');
  }
}

/** A small SVG of an entry's pitch shape: one point per sound, the fall marked. */
function sketchSvg(e: OwnDictEntry): string {
  const s = contourSketch(e);
  if (!s) return '';
  const w = Math.max(40, s.values.length * 14), h = 30, pad = 5;
  const lo = Math.min(-0.3, ...s.values), hi = Math.max(0.3, ...s.values);
  const x = (k: number): number => (s.values.length < 2 ? w / 2 : pad + (k / (s.values.length - 1)) * (w - 2 * pad));
  const y = (v: number): number => pad + ((hi - v) / (hi - lo)) * (h - 2 * pad);
  const pts = s.values.map((v, k) => `${x(k).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const dots = s.values.map((v, k) => `<circle cx="${x(k).toFixed(1)}" cy="${y(v).toFixed(1)}" r="2.5"/>`).join('');
  const fall = s.fallAfter >= 0 ? `<line class="fall" x1="${((x(s.fallAfter) + x(s.fallAfter + 1)) / 2).toFixed(1)}" x2="${((x(s.fallAfter) + x(s.fallAfter + 1)) / 2).toFixed(1)}" y1="${pad}" y2="${h - pad}"/>` : '';
  return `<svg class="sketch" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="高さの形">${fall}<polyline points="${pts}"/>${dots}</svg>`;
}

function drawList(): void {
  const list = browserDict();
  $('dict-count').textContent = list.length ? `${list.length} 語` : 'まだありません';
  if (browserDictProblem) $('dict-count').textContent += `（${browserDictProblem}）`;
  const ul = $('dict-list');
  ul.replaceChildren(...list.map((e, i) => {
    const li = document.createElement('li');
    li.dataset.i = String(i);
    li.dataset.intonation = e.contour ? '1' : '0';
    const tags = [e.contour ? '高さ' : '', e.length ? '長さ' : '', e.accent !== undefined ? '下がる位置' : ''].filter(Boolean);
    li.innerHTML = '<span class="w"><b></b><span class="r"></span><small></small></span>'
      + `<span class="tone">${sketchSvg(e)}</span>`
      + '<span class="ops"><button type="button" class="btn small" data-op="edit">直す</button><button type="button" class="btn small danger" data-op="delete">消す</button></span>';
    li.querySelector('b')!.textContent = e.surface;
    li.querySelector('.r')!.textContent = e.reading;
    li.querySelector('small')!.textContent = tags.length ? `${tags.join('・')}あり` : '高さ：予測どおり（「直す」で調整）';
    return li;
  }));
}

function onListClick(ev: Event): void {
  const t = (ev.target as HTMLElement).closest<HTMLElement>('button[data-op]');
  const li = t?.closest<HTMLElement>('li');
  if (!t || !li) return;
  const i = Number(li.dataset.i);
  const list = [...browserDict()];
  const e = list[i];
  if (!e) return;
  try {
    if (t.dataset.op === 'delete') {
      if (!confirm(`「${e.surface}」を辞書から消しますか？`)) return;
      list.splice(i, 1);
      saveBrowserDict(list);
      drawList();
      tuneStatus(`「${e.surface}」を消しました。`);
      return;
    }
    // 直す: the word alone in the panel (reading and intonation edited there, saved with 「辞書に登録」)
    if (t.dataset.op === 'edit') void openEntry(e);
  } catch (err) {
    tuneStatus(err instanceof Error ? err.message : String(err), 'error');
  }
}

function exportDict(): void {
  const blob = new Blob([`${JSON.stringify(dictFile(browserDict()), null, 1)}\n`], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'zen-tts-dictionary.json';
  a.click();
  // iOS opens a preview that reads the URL later
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
  tuneStatus(`辞書（${browserDict().length} 語）をファイルに書き出しました。`);
}

async function importDict(ev: Event): Promise<void> {
  const input = ev.target as HTMLInputElement;
  const f = input.files?.[0];
  input.value = '';
  if (!f) return;
  try {
    if (f.size > 1_000_000) throw new BrowserDictError('ファイルが大きすぎます（1MB まで）');
    const entries = parseDictFile(await f.text());
    const r = mergeDictEntries(browserDict(), entries);
    saveBrowserDict(r.entries);
    drawList();
    tuneStatus(`ファイルから読み込みました：新しく ${r.added} 語、同じ表記を上書き ${r.replaced} 語。`);
    document.body.dataset.dictImported = String(r.entries.length);
  } catch (err) {
    tuneStatus(`読み込めませんでした：${err instanceof Error ? err.message : String(err)}`, 'error');
  }
}

/** For the headless check: the open reading (its voice, the pitch of each sound before / after, the edits). */
(window as Window & { zenTtsTuneState?: () => unknown }).zenTtsTuneState = () => cur && ({
  word: cur.word ?? null, voiceKey: cur.voice.key, blend: cur.voice.blend, curve: !!cur.voice.blendAt, kana: cur.units.map((u) => u.kana),
  beforePitch: cur.beforePitch, afterPitch: cur.afterPitch ?? null, edits: { pitch: edits.pitch, length: edits.length, accent: edits.accent },
  beforeAudioSum: cur.before.audio.reduce((s, v) => s + Math.abs(v), 0),
});

/** For the headless check: how 「直した後」 differs from 「直す前」 (seconds, the largest pitch change, audio difference). */
(window as Window & { zenTtsTune?: () => unknown }).zenTtsTune = () => {
  if (!cur?.after || !cur.afterPitch) return null;
  const a = cur.before.audio, b = cur.after.audio;
  const n = Math.min(a.length, b.length);
  let d = 0;
  for (let i = 0; i < n; i++) d += Math.abs(a[i]! - b[i]!);
  const dp = cur.units.map((_, i) => Math.abs((cur!.afterPitch![i] ?? 0) - (cur!.beforePitch[i] ?? 0)));
  return { beforeSec: +(a.length / SAMPLE_RATE).toFixed(2), afterSec: +(b.length / SAMPLE_RATE).toFixed(2), maxPitchChange: +Math.max(...dp).toFixed(2), meanAbsDiff: +(d / Math.max(1, n)).toFixed(4) };
};

wire();
drawList();
drawSelected();
