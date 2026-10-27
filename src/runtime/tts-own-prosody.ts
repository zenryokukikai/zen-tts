// CPU steps between the GPU networks: G2P slots -> phones / accent marks; accent features; predictor head ->
// durations, F0 = template + strength * deviation, voiced; frame rows; and the reading / intonation dictionary:
// L1 surface -> reading (kana -> phones by the mora table), L2 accent type -> marks, L3 F0 / length per mora.
import { emotionNames, type EmotionBlend } from './emotion.js';
import { normalizeOwnText } from './ja-text-normalize.js';
import { hiraganaToKatakana, kanaToPhones } from './kana-phones.js';

export { kanaToPhones };

/** own.json (the parts the CPU steps use). */
export interface OwnMeta {
  shape: {
    g2p: { dim: number; layers: number; ff: number; heads: number; vocab: number; slots: number; phoneClasses: number; marks: number };
    pred: { dim: number; layers: number; ff: number; heads: number; kernel: number; phones: number };
    gen: { dim: number; ff: number; layers: number; kernel: number; extra: number; z: number };
    window: number; lnEps: number; styles: number; sampleRate: number; hop: number;
  };
  vocab: string[];
  charPad: number;
  charUnk: number;
  symbols: string[];
  phoneClasses: string[];
  marks: string[];
  punctuation: string[];
  template: { B: number[]; H: number[]; D: number[] };
  /** the voice's F0 range per emotion (see DEFAULT_PITCH_RANGE); omitted = DEFAULT_PITCH_RANGE (「全力 マリン」) */
  pitchRange?: number[];
  kana: Record<string, [string, string]>;
  tensors: { name: string; dtype: 'i8' | 'f16' | 'f32'; shape: number[]; offset: number; bytes: number }[];
}

/** Marks of a phone (the accent / phrase symbol after it in the prosody label string). */
export const MARK = { none: 0, rise: 1, fall: 2, phrase: 3, pause: 4 } as const;
const MORA_END = new Set(['a', 'i', 'u', 'e', 'o', 'N', 'a:', 'i:', 'u:', 'e:', 'o:', 'q']);
/**
 * The own dictionary's global settings : accentHeight = H of a word with an accent type, in semitones;
 * pitchRange = the speaker's F0 range (p90 - p10, semitones) per emotion in emotionNames order, what a contour value of 1
 * is worth (omitted: DEFAULT_PITCH_RANGE).
 */
export interface OwnDictSettings { accentHeight: number; pitchRange?: number[] }
export const DEFAULT_OWN_DICT_SETTINGS: OwnDictSettings = { accentHeight: 4.5 };
/** The voice's F0 range (p90 - p10, semitones) per emotion, in emotionNames order. */
export const DEFAULT_PITCH_RANGE: readonly number[] = [7.52, 8.38, 5.77, 7.48, 6.36];

/** The dictionary settings for a voice: the given ones (or the defaults), with the voice's pitchRange when they have none. */
export function voiceDictSettings(settings: OwnDictSettings | undefined, meta: Pick<OwnMeta, 'pitchRange'>): OwnDictSettings {
  const s = settings ?? DEFAULT_OWN_DICT_SETTINGS;
  return s.pitchRange || !meta.pitchRange ? s : { ...s, pitchRange: meta.pitchRange };
}

/** The F0 range (st) of a style: the blend's weighted mean of the per-emotion ranges (an all-zero blend = neutral). */
export function pitchRangeOf(blend: EmotionBlend, range: readonly number[] = DEFAULT_PITCH_RANGE): number {
  const sum = emotionNames.reduce((s, k) => s + Math.max(0, blend[k]), 0);
  if (sum <= 0) return range[0]!;
  return emotionNames.reduce((s, k, i) => s + Math.max(0, blend[k]) * range[i]!, 0) / sum;
}

/**
 * A relative contour on one word: phones a..b-1 (indices of f0, edges counted)
 * grouped into moras by moraEnds (phones after the last mora end are in none); level = the mean over all the word's
 * moras of each mora's mean F0; each mora k < values.length is shifted so its mean is level + values[k] * range (the
 * shape inside the mora kept), the others are kept. The level keeps the style's height and declination there,
 * range scales the shape to the style's spread.
 */
export function contourF0(f0: readonly number[], phones: readonly string[], a: number, b: number, values: readonly number[], range: number): number[] {
  const out = [...f0];
  const groups: [number, number][] = [];
  let from = a;
  for (let i = a; i < b; i++) if (MORA_END.has(phones[i]!)) { groups.push([from, i + 1]); from = i + 1; }
  if (!groups.length || !values.length) return out;
  const mean = ([s, e]: [number, number]): number => { let m = 0; for (let i = s; i < e; i++) m += f0[i]!; return m / (e - s); };
  const means = groups.map(mean);
  const level = means.reduce((x, y) => x + y, 0) / means.length;
  groups.slice(0, values.length).forEach(([s, e], k) => {
    const shift = level + values[k]! * range - means[k]!;
    for (let i = s; i < e; i++) out[i] = f0[i]! + shift;
  });
  return out;
}

/** Round half to even. */
export function roundHalfEven(v: number): number {
  const r = Math.round(v);
  return Math.abs(v - Math.trunc(v)) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** G2P argmax per char [phone x slots | mark x slots] -> phones, marks, per char [start, end) phone index. */
export function slotsToPhones(slots: number[][], meta: Pick<OwnMeta, 'phoneClasses'>, nSlots: number): { phones: string[]; marks: number[]; spans: [number, number][] } {
  const phones: string[] = [];
  const marks: number[] = [];
  const spans: [number, number][] = [];
  for (const s of slots) {
    const a = phones.length;
    for (let k = 0; k < nSlots; k++) {
      if (s[k]) { phones.push(meta.phoneClasses[s[k]!]!); marks.push(s[nSlots + k]!); }
    }
    spans.push([a, phones.length]);
  }
  return { phones, marks, spans };
}

/** Tones (high 1 / low 0) per phone from the accent marks (non-strict). */
export function tonesFromMarks(phones: readonly string[], marks: readonly number[], punct: ReadonlySet<string>): number[] {
  const tones = new Array<number>(phones.length).fill(0);
  let phrase: [number, number][] = [];
  let cur = 0;
  const close = (): void => {
    const vals = new Set(phrase.map(([, t]) => t));
    const only = (...ok: number[]): boolean => [...vals].every((v) => ok.includes(v));
    if (vals.size === 2 && vals.has(-1) && vals.has(0)) for (const [i, t] of phrase) tones[i] = t + 1;
    else if (only(0, 1)) for (const [i, t] of phrase) tones[i] = t;
    else { const top = Math.max(...vals); for (const [i, t] of phrase) tones[i] = t === top ? 1 : 0; }
    phrase = [];
  };
  phones.forEach((p, i) => {
    if (punct.has(p)) return;
    phrase.push([i, cur]);
    const m = marks[i]!;
    if (m === MARK.rise) cur++;
    else if (m === MARK.fall) cur--;
    else if (m === MARK.phrase || m === MARK.pause) { close(); cur = 0; }
  });
  close();
  return tones;
}

export interface AccentFeatures { tone: number[]; nucleus: number[]; boundary: number[]; pos: number[] }

/** Per phone: high, nucleus, first phone of an accent phrase, position 0..1 in its phrase. */
export function accentFeatures(phones: readonly string[], marks: readonly number[], punct: ReadonlySet<string>): AccentFeatures {
  const n = phones.length;
  const tone = tonesFromMarks(phones, marks, punct);
  const nucleus = marks.map((m) => (m === MARK.fall ? 1 : 0));
  const boundary = new Array<number>(n).fill(0);
  const pos = new Array<number>(n).fill(0);
  const groups: number[][] = [];
  let cur: number[] = [];
  let start = true;
  phones.forEach((p, i) => {
    if (punct.has(p)) return;
    if (start) { boundary[i] = 1; start = false; }
    cur.push(i);
    if (marks[i] === MARK.phrase || marks[i] === MARK.pause) { groups.push(cur); cur = []; start = true; }
  });
  if (cur.length) groups.push(cur);
  for (const g of groups) g.forEach((i, k) => { pos[i] = k / Math.max(1, g.length - 1); });
  return { tone, nucleus, boundary, pos };
}

/** B, H, D of the template mixed by the blend (emotionNames order = the model's style order). */
export function templateParams(meta: Pick<OwnMeta, 'template'>, blend: EmotionBlend): { B: number; H: number; D: number } {
  const dot = (v: number[]): number => emotionNames.reduce((s, k, i) => s + blend[k] * v[i]!, 0);
  return { B: dot(meta.template.B), H: dot(meta.template.H), D: dot(meta.template.D) };
}

export interface OwnProsody {
  /** with the edge '_' at both ends */
  phones: string[];
  marks: number[];
  acc: AccentFeatures;
  dur: number[];
  template: number[];
  dev: number[];
  f0: number[];
  energy: number[];
  voiced: number[];
}

/** [_ phones _] and their accent features (the predictor input besides the style). */
export function predictorInput(phones: readonly string[], marks: readonly number[], punct: ReadonlySet<string>): { phones: string[]; marks: number[]; acc: AccentFeatures } {
  const ph = ['_', ...phones, '_'];
  const mk = [0, ...marks, 0];
  return { phones: ph, marks: mk, acc: accentFeatures(ph, mk, punct) };
}

/** The predictor head [P, 4] (log(dur + 1), F0 deviation st, energy, voiced logit) -> prosody. */
export function prosodyFromHead(input: ReturnType<typeof predictorInput>, head: Float32Array, meta: Pick<OwnMeta, 'template'>, blend: EmotionBlend, strength = 1): OwnProsody {
  const { B, H, D } = templateParams(meta, blend);
  const P = input.phones.length;
  const dur: number[] = [];
  const template: number[] = [];
  const dev: number[] = [];
  const f0: number[] = [];
  const energy: number[] = [];
  const voiced: number[] = [];
  for (let i = 0; i < P; i++) {
    dur.push(Math.max(1, roundHalfEven(Math.exp(head[i * 4]!) - 1)));
    const t = B + H * input.acc.tone[i]! - D * input.acc.pos[i]!;
    template.push(t);
    dev.push(head[i * 4 + 1]!);
    f0.push(t + strength * head[i * 4 + 1]!);
    energy.push(head[i * 4 + 2]!);
    voiced.push(1 / (1 + Math.exp(-head[i * 4 + 3]!)));
  }
  return { ...input, dur, template, dev, f0, energy, voiced };
}

/** Per frame the phone index and [f0 / 12 (linear between phone centres), energy, position in the phone, voiced]. */
export function frameRows(pr: Pick<OwnProsody, 'dur' | 'f0' | 'energy' | 'voiced'>): { idx: Uint32Array; extra: Float32Array } {
  const F = pr.dur.reduce((s, d) => s + d, 0);
  const idx = new Uint32Array(F);
  const extra = new Float32Array(F * 4);
  const start: number[] = [];
  const centre: number[] = [];
  let f = 0;
  pr.dur.forEach((d, p) => { start.push(f); centre.push(f + d / 2); for (let k = 0; k < d; k++) idx[f++] = p; });
  let j = 0;
  for (let t = 0; t < F; t++) {
    const x = t + 0.5;
    while (j < centre.length - 1 && centre[j + 1]! <= x) j++;
    let v: number;
    if (x <= centre[0]!) v = pr.f0[0]!;
    else if (x >= centre[centre.length - 1]!) v = pr.f0[centre.length - 1]!;
    else v = pr.f0[j]! + (pr.f0[j + 1]! - pr.f0[j]!) * (x - centre[j]!) / (centre[j + 1]! - centre[j]!);
    const p = idx[t]!;
    extra[t * 4] = v / 12;
    extra[t * 4 + 1] = pr.energy[p]!;
    extra[t * 4 + 2] = (x - start[p]!) / pr.dur[p]!;
    extra[t * 4 + 3] = pr.voiced[p]!;
  }
  return { idx, extra };
}

// ---- the reading / intonation dictionary ----

export interface OwnDictEntry {
  surface: string;
  /** katakana or hiragana */
  reading: string;
  /** 0 flat, n = fall after mora n; omitted: the G2P reads the reading */
  accent?: number;
  phrase?: 'split' | 'join';
  /** the model's F0 deviation in the word's accent phrase (0 = the type only) */
  strength?: number;
  /** relative F0 per mora (-3..3): F0 = the word's mean model F0 + value * the style's range (pitchRangeOf), before pitch */
  contour?: number[];
  /** semitones per mora, added */
  pitch?: number[];
  /** duration factor per mora */
  length?: number[];
  note?: string;
}

/** Morae of phones (indices of the phone that ends each mora). */
export function moraEnds(phones: readonly string[]): number[] {
  const out: number[] = [];
  phones.forEach((p, i) => { if (MORA_END.has(p)) out.push(i); });
  return out;
}

/** Accent type typ written into marks a..b-1: rise / fall cleared, then [ after mora 1 and ] after mora typ. */
export function setAccent(phones: readonly string[], marks: readonly number[], a: number, b: number, typ: number): number[] {
  const mk = [...marks];
  for (let i = a; i < b; i++) if (mk[i] === MARK.rise || mk[i] === MARK.fall) mk[i] = 0;
  const ends = moraEnds(phones.slice(a, b)).map((i) => i + a);
  if (!ends.length) return mk;
  if (typ === 1) { if (mk[ends[0]!] === 0) mk[ends[0]!] = MARK.fall; return mk; }
  if (mk[ends[0]!] === 0) mk[ends[0]!] = MARK.rise;
  const n = ends[Math.min(typ, ends.length) - 1];
  if (typ > 1 && n !== undefined && mk[n] === 0) mk[n] = MARK.fall;
  return mk;
}

/** Accent type of a..b-1 by its marks (0 flat). */
export function accentTypeOf(phones: readonly string[], marks: readonly number[], a: number, b: number): number {
  let n = 0;
  for (let i = a; i < b; i++) {
    if (MORA_END.has(phones[i]!)) { n++; if (marks[i] === MARK.fall) return n; }
  }
  return 0;
}

/** Index after the accent phrase holding phone b - 1 (up to the next # or _ mark, inclusive). */
export function phraseEnd(marks: readonly number[], b: number): number {
  let i = b - 1;
  while (i < marks.length - 1 && marks[i] !== MARK.phrase && marks[i] !== MARK.pause) i++;
  return i + 1;
}

/**
 * F0 of the accent phrase a..e-1 (phone indices without the edge '_') of a word with an accent type:
 * template with H = height, + the model's level (F0 - template with that H) linear from the phone before to the one after.
 * base: the prosody the phrase is joined to; edited: its marks' features. Returns the whole F0 (edge included).
 */
export function dictPhraseF0(base: Pick<OwnProsody, 'f0'>, edited: Pick<OwnProsody, 'acc'>, a: number, e: number, meta: Pick<OwnMeta, 'template'>, blend: EmotionBlend, height: number, strength = 0, dev?: readonly number[]): number[] {
  const { B, D } = templateParams(meta, blend);
  const tone = edited.acc.tone;
  const pos = edited.acc.pos;
  const at = (i: number): number => B + height * tone[i]! - D * pos[i]!;
  const i0 = a + 1;
  const i1 = e + 1;
  const lv0 = base.f0[i0 - 1]! - at(i0 - 1);
  const lv1 = i1 < tone.length ? base.f0[i1]! - at(i1) : lv0;
  const f0 = [...base.f0];
  const n = i1 - i0;
  for (let k = 0; k < n; k++) {
    const t = (k + 1) / (n + 1);
    const i = i0 + k;
    f0[i] = at(i) + (1 - t) * lv0 + t * lv1 + (dev ? strength * dev[i]! : 0);
  }
  return f0;
}

export interface DictMatch { entry: OwnDictEntry; /** char range in the text the G2P reads */ start: number; end: number }

/**
 * L1 on normalized text: longest surface first, left to right; matched surfaces become their readings (katakana) in the
 * text the G2P reads (so it reads the word's context as usual). Returns that text and where each reading is.
 */
export function applyOwnDict(norm: string, entries: readonly OwnDictEntry[]): { text: string; matches: DictMatch[] } {
  if (!entries.length) return { text: norm, matches: [] };
  const sorted = entries.map((e) => ({ e, s: normalizeOwnText(e.surface).replace(/[.]$/, '') })).filter((x) => x.s !== '')
    .sort((x, y) => y.s.length - x.s.length);
  let text = '';
  const matches: DictMatch[] = [];
  let i = 0;
  outer: while (i < norm.length) {
    for (const { e, s } of sorted) {
      if (norm.startsWith(s, i)) {
        const r = hiraganaToKatakana(e.reading);
        matches.push({ entry: e, start: text.length, end: text.length + r.length });
        text += r;
        i += s.length;
        continue outer;
      }
    }
    text += norm[i];
    i++;
  }
  return { text, matches };
}

export interface DictWord { entry: OwnDictEntry; /** phone range [a, b) (no edge) */ a: number; b: number }

/**
 * L1 / L2 on the G2P output of applyOwnDict's text: a word with an accent type gets the mora table's phones and the
 * type's marks (a boundary before it when phrase is split); its rise / fall marks after it up to the phrase end are cleared.
 * aligned: every replaced word had the G2P's phone count (then the G2P's own prosody can hold the outside).
 */
export function dictLabels(g2p: { phones: string[]; marks: number[]; spans: [number, number][] }, matches: readonly DictMatch[], kana: OwnMeta['kana']): { phones: string[]; marks: number[]; words: DictWord[]; aligned: boolean } {
  const phones: string[] = [];
  const marks: number[] = [];
  const words: DictWord[] = [];
  let aligned = true;
  let next = 0;
  for (const m of matches) {
    const a0 = g2p.spans[m.start]![0];
    const b0 = g2p.spans[m.end - 1]![1];
    for (let i = next; i < a0; i++) { phones.push(g2p.phones[i]!); marks.push(g2p.marks[i]!); }
    next = b0;
    const a = phones.length;
    if (m.entry.accent === undefined) {
      for (let i = a0; i < b0; i++) { phones.push(g2p.phones[i]!); marks.push(g2p.marks[i]!); }
      words.push({ entry: m.entry, a, b: phones.length });
      continue;
    }
    const ph = kanaToPhones(m.entry.reading, kana);
    if (ph.length !== b0 - a0 || ph.some((p, k) => p !== g2p.phones[a0 + k])) aligned = false;
    // the word keeps the G2P's mark of its last phone when it is a phrase end or pause
    const last = g2p.marks[b0 - 1]!;
    for (const p of ph) { phones.push(p); marks.push(0); }
    if (last === MARK.phrase || last === MARK.pause) marks[marks.length - 1] = last;
    if ((m.entry.phrase ?? 'split') === 'split' && a > 0 && marks[a - 1] === 0) marks[a - 1] = MARK.phrase;
    words.push({ entry: m.entry, a, b: phones.length });
  }
  for (let i = next; i < g2p.phones.length; i++) { phones.push(g2p.phones[i]!); marks.push(g2p.marks[i]!); }
  let mk = marks;
  for (const w of words) {
    if (w.entry.accent === undefined) continue;
    mk = setAccent(phones, mk, w.a, w.b, w.entry.accent);
    const e = phraseEnd(mk, w.b);
    for (let i = w.b; i < e; i++) if (mk[i] === MARK.rise || mk[i] === MARK.fall) mk[i] = 0;
  }
  return { phones, marks: mk, words, aligned };
}

/**
 * L3: the final F0 and durations. base: the prosody of the G2P's own labels (when aligned; else the dictionary run), its
 * F0 and durations kept outside the accent phrases of words with a type; dict: the prosody of the dictionary labels.
 */
export function dictProsody(base: OwnProsody, dict: OwnProsody, words: readonly DictWord[], meta: Pick<OwnMeta, 'template'>, blend: EmotionBlend, settings: OwnDictSettings): { f0: number[]; dur: number[] } {
  let f0 = [...base.f0];
  const dur = [...base.dur];
  for (const w of words) {
    const e = w.entry;
    if (e.accent !== undefined) {
      const end = phraseEnd(dict.marks.slice(1, -1), w.b);
      f0 = dictPhraseF0({ f0 }, dict, w.a, end, meta, blend, settings.accentHeight, e.strength ?? 0, dict.dev);
    }
    if (e.contour?.length) f0 = contourF0(f0, dict.phones, w.a + 1, w.b + 1, e.contour, pitchRangeOf(blend, settings.pitchRange));
    const ends = moraEnds(dict.phones.slice(w.a + 1, w.b + 1)).map((i) => i + w.a + 1);
    let from = w.a + 1;
    ends.forEach((end, k) => {
      for (let i = from; i <= end; i++) {
        if (e.pitch?.[k]) f0[i] = f0[i]! + e.pitch[k]!;
        if (e.length?.[k] !== undefined && e.length[k] !== 1) dur[i] = Math.max(1, roundHalfEven(dur[i]! * e.length[k]!));
      }
      from = end + 1;
    });
  }
  return { f0, dur };
}

/** Prosody label string ("z e[ N ..."). */
export function labelString(phones: readonly string[], marks: readonly number[]): string {
  return phones.map((p, i) => p + ['', '[', ']', '#', '_'][marks[i]!]!).join(' ');
}
