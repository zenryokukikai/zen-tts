// The demo's intonation panel and the dictionary kept in this browser only. Everything here is pure (the page,
// src/demo/tune.ts, does the voice and the storage):
//   - the browser dictionary: entries (DictEntry), validated on load / import, merged by 表記, written as a file to move it;
//   - the sentence's sounds (moras) from the G2P's phones / marks, with their kana, accent phrase and phone range;
//   - the panel's edits (下がる位置 per phrase, 高さ / 長さ per sound) -> the speak options (labels, f0Add, durScale),
//     and one word's edited shape -> a dictionary entry (contour / length / accent).
import { MARK, accentTypeOf, setAccent, type OwnDictEntry, type OwnMeta } from '../runtime/tts-own-prosody.js';

export const BROWSER_DICT_STORAGE_KEY = 'zen-tts.dict.v2';
export const BROWSER_DICT_FORMAT = 'zen-tts-dictionary';
export const BROWSER_DICT_MAX_ENTRIES = 500;
const MAX_SURFACE = 40;
const MAX_READING = 80;
const MAX_MORA_VALUES = 80;
const MAX_NOTE = 200;
const KANA = /^[\u3041-\u3096\u30A1-\u30FA\u30FC\u309D\u309E\u30FD\u30FE]+$/;

export class BrowserDictError extends Error {}

function numberList(v: unknown, row: string, name: string, lo: number, hi: number): number[] {
  if (!Array.isArray(v)) throw new BrowserDictError(`${row}：${name}は数の並びで入れてください`);
  if (v.length > MAX_MORA_VALUES) throw new BrowserDictError(`${row}：${name}は ${MAX_MORA_VALUES} 個までです`);
  return v.map((x) => {
    const n = typeof x === 'number' ? x : Number.NaN;
    if (!Number.isFinite(n) || n < lo || n > hi) throw new BrowserDictError(`${row}：${name}は ${lo}〜${hi} の数で入れてください`);
    return n;
  });
}

/** One entry checked (trimmed; reading kana only; optional fields kept only when given and in range). */
export function validateDictEntry(x: unknown, row = '辞書'): OwnDictEntry {
  if (typeof x !== 'object' || x === null || Array.isArray(x)) throw new BrowserDictError(`${row}：語の形が正しくありません`);
  const o = x as Record<string, unknown>;
  const surface = typeof o['surface'] === 'string' ? o['surface'].trim() : '';
  const reading = typeof o['reading'] === 'string' ? o['reading'].trim() : '';
  if (surface === '' || reading === '') throw new BrowserDictError(`${row}：表記と読みを入れてください`);
  if ([...surface].length > MAX_SURFACE) throw new BrowserDictError(`${row}：表記は ${MAX_SURFACE} 文字までです`);
  if ([...reading].length > MAX_READING) throw new BrowserDictError(`${row}：読みは ${MAX_READING} 文字までです`);
  if (!KANA.test(reading)) throw new BrowserDictError(`${row}：読みはひらがなかカタカナで入れてください`);
  const e: OwnDictEntry = { surface, reading };
  if (o['accent'] !== undefined && o['accent'] !== null) {
    const a = o['accent'];
    if (typeof a !== 'number' || !Number.isInteger(a) || a < 0 || a > MAX_READING) throw new BrowserDictError(`${row}：下がる位置は 0 以上の整数で入れてください`);
    e.accent = a;
  }
  if (o['phrase'] !== undefined && o['phrase'] !== null) {
    if (o['phrase'] !== 'split' && o['phrase'] !== 'join') throw new BrowserDictError(`${row}：区切りの指定が正しくありません`);
    e.phrase = o['phrase'];
  }
  if (o['strength'] !== undefined && o['strength'] !== null) {
    const s = o['strength'];
    if (typeof s !== 'number' || !Number.isFinite(s) || s < 0 || s > 2) throw new BrowserDictError(`${row}：抑揚の強さは 0〜2 で入れてください`);
    e.strength = s;
  }
  if (o['contour'] !== undefined && o['contour'] !== null) e.contour = numberList(o['contour'], row, '高さの形', -3, 3);
  if (o['pitch'] !== undefined && o['pitch'] !== null) e.pitch = numberList(o['pitch'], row, '高さ', -12, 12);
  if (o['length'] !== undefined && o['length'] !== null) e.length = numberList(o['length'], row, '長さ', 0.25, 4);
  if (typeof o['note'] === 'string' && o['note'].trim() !== '') e.note = o['note'].trim().slice(0, MAX_NOTE);
  return e;
}

/** A list of entries checked: every entry valid, 表記 unique, at most BROWSER_DICT_MAX_ENTRIES. */
export function validateDictEntries(input: unknown): OwnDictEntry[] {
  if (!Array.isArray(input)) throw new BrowserDictError('辞書の形が正しくありません（語の並びがありません）');
  if (input.length > BROWSER_DICT_MAX_ENTRIES) throw new BrowserDictError(`登録できるのは ${BROWSER_DICT_MAX_ENTRIES} 語までです`);
  const seen = new Set<string>();
  return input.map((x, n) => {
    const e = validateDictEntry(x, `${n + 1} 語目`);
    if (seen.has(e.surface)) throw new BrowserDictError(`${n + 1} 語目：「${e.surface}」が重複しています`);
    seen.add(e.surface);
    return e;
  });
}

/** The file written by 書き出す (and kept in localStorage). */
export interface BrowserDictFile { format: typeof BROWSER_DICT_FORMAT; version: 1; entries: OwnDictEntry[] }

export function dictFile(entries: readonly OwnDictEntry[]): BrowserDictFile {
  return { format: BROWSER_DICT_FORMAT, version: 1, entries: [...entries] };
}

/** Text of a dictionary file (or the stored value) -> its entries; a bare list of entries is accepted too. */
export function parseDictFile(text: string): OwnDictEntry[] {
  let v: unknown;
  try { v = JSON.parse(text); } catch { throw new BrowserDictError('辞書のファイルとして読めません（JSON ではありません）'); }
  if (Array.isArray(v)) return validateDictEntries(v);
  if (typeof v !== 'object' || v === null) throw new BrowserDictError('辞書のファイルとして読めません');
  const o = v as Record<string, unknown>;
  if (o['format'] !== BROWSER_DICT_FORMAT) throw new BrowserDictError('このページの辞書のファイルではありません');
  if (o['version'] !== 1) throw new BrowserDictError('辞書のファイルの版が違います');
  return validateDictEntries(o['entries']);
}

/** add on top of base: the same 表記 is replaced (in place), new ones appended. Throws when the result is too long. */
export function mergeDictEntries(base: readonly OwnDictEntry[], add: readonly OwnDictEntry[]): { entries: OwnDictEntry[]; added: number; replaced: number } {
  const out = [...base];
  let added = 0;
  let replaced = 0;
  for (const e of add) {
    const i = out.findIndex((x) => x.surface === e.surface);
    if (i >= 0) { out[i] = e; replaced++; } else { out.push(e); added++; }
  }
  if (out.length > BROWSER_DICT_MAX_ENTRIES) throw new BrowserDictError(`登録できるのは ${BROWSER_DICT_MAX_ENTRIES} 語までです`);
  return { entries: out, added, replaced };
}

// ---- the sentence's sounds ----

const MORA_END = new Set(['a', 'i', 'u', 'e', 'o', 'N', 'a:', 'i:', 'u:', 'e:', 'o:', 'q']);
/** Kana that share their sounds with a commoner one (ヂ = ジ, ヲ = オ, small vowels ...): never the one shown. */
const RARE_KANA = new Set(['ヂ', 'ヅ', 'ヲ', 'ヰ', 'ヱ', 'ヮ', 'ヶ', 'ァ', 'ィ', 'ゥ', 'ェ', 'ォ', 'ャ', 'ュ', 'ョ']);

/** One sound (mora) of the sentence, or one punctuation mark. a..b-1: its phones (no edge '_'). */
export interface SoundUnit {
  /** katakana; a long vowel is its kana + ー */
  kana: string;
  punct: boolean;
  a: number;
  b: number;
  /** accent phrase index (a new one after a phrase / pause mark and after punctuation) */
  phrase: number;
}

function kanaOfPhones(kana: OwnMeta['kana']): Map<string, string> {
  const m = new Map<string, string>();
  for (const [k, [c, v]] of Object.entries(kana)) {
    const key = `${c}|${v}`;
    const cur = m.get(key);
    const better = cur === undefined || (RARE_KANA.has(cur) && !RARE_KANA.has(k)) || (RARE_KANA.has(cur) === RARE_KANA.has(k) && k.length < cur.length);
    if (better) m.set(key, k);
  }
  return m;
}

const PUNCT_SHOWN: Record<string, string> = { ',': '、', '.': '。', '!': '！', '?': '？' };

/** The G2P's phones / marks -> sounds with their kana and accent phrase. */
export function soundUnits(phones: readonly string[], marks: readonly number[], kana: OwnMeta['kana'], punct: ReadonlySet<string>): SoundUnit[] {
  const rev = kanaOfPhones(kana);
  const out: SoundUnit[] = [];
  let phrase = 0;
  let from = 0;
  const name = (ph: readonly string[]): string => {
    const v = ph[ph.length - 1]!;
    const c = ph.slice(0, -1).join('');
    if (v.endsWith(':')) return `${rev.get(`${c}|${v[0]}`) ?? ''}ー`;
    return rev.get(`${c}|${v}`) ?? ph.join('');
  };
  for (let i = 0; i < phones.length; i++) {
    const p = phones[i]!;
    if (punct.has(p)) {
      if (from < i) out.push({ kana: name(phones.slice(from, i)), punct: false, a: from, b: i, phrase });
      out.push({ kana: PUNCT_SHOWN[p] ?? p, punct: true, a: i, b: i + 1, phrase });
      from = i + 1;
      phrase++;
      continue;
    }
    if (!MORA_END.has(p)) continue;
    out.push({ kana: name(phones.slice(from, i + 1)), punct: false, a: from, b: i + 1, phrase });
    from = i + 1;
    if (marks[i] === MARK.phrase || marks[i] === MARK.pause) phrase++;
  }
  if (from < phones.length) out.push({ kana: name(phones.slice(from)), punct: false, a: from, b: phones.length, phrase });
  // phrase indices without gaps (a phrase of punctuation only is kept: it holds the mark)
  const ids = [...new Set(out.map((u) => u.phrase))];
  for (const u of out) u.phrase = ids.indexOf(u.phrase);
  return out;
}

const toKatakana = (s: string): string => s.replace(/[\u3041-\u3096\u309D\u309E]/g, (c) => String.fromCharCode(c.charCodeAt(0) + 0x60));

/** Katakana with long vowels written ー (コウ, コオ and コー all become コー; ケイ and ケー become ケー), to compare readings. */
export function readingKey(s: string, kana: OwnMeta['kana']): string {
  let out = '';
  let prev: string | null = null;
  for (const c of toKatakana(s)) {
    if (c === 'ー') { out += 'ー'; continue; }
    const m = kana[c];
    const v = m ? m[1] : null;
    const plainVowel = m !== undefined && m[0] === '' && /^[aiueo]$/.test(m[1]) && !RARE_KANA.has(c);
    if (plainVowel && prev && (v === prev || (v === 'u' && prev === 'o') || (v === 'i' && prev === 'e'))) { out += 'ー'; continue; }
    out += c;
    prev = v && /^[aiueo]$/.test(v) ? v : null;
  }
  return out;
}

/** Where a reading (hiragana / katakana) is in the sentence's sounds: the unit indices of its sounds, or undefined. */
export function findReadingUnits(units: readonly SoundUnit[], reading: string, kana: OwnMeta['kana']): number[] | undefined {
  const want = readingKey(reading.trim(), kana);
  if (!want) return undefined;
  const idx = units.map((u, i) => (u.punct ? -1 : i)).filter((i) => i >= 0);
  for (let s = 0; s < idx.length; s++) {
    let txt = '';
    for (let e = s; e < idx.length; e++) {
      txt += units[idx[e]!]!.kana;
      const k = readingKey(txt, kana);
      if (k === want) return idx.slice(s, e + 1);
      if (!want.startsWith(k)) break;
    }
  }
  return undefined;
}

/** The phone range [a, b) of an accent phrase (its sounds, punctuation excluded); undefined when it has none. */
export function phraseRange(units: readonly SoundUnit[], phrase: number): [number, number] | undefined {
  const us = units.filter((u) => !u.punct && u.phrase === phrase);
  return us.length ? [us[0]!.a, us[us.length - 1]!.b] : undefined;
}

/** 下がる位置 of every phrase as the G2P read it (0 = 下がらない, n = after its n-th sound); punctuation-only phrases are -1. */
export function phraseAccents(units: readonly SoundUnit[], phones: readonly string[], marks: readonly number[]): number[] {
  const n = units.length ? Math.max(...units.map((u) => u.phrase)) + 1 : 0;
  return Array.from({ length: n }, (_, p) => {
    const r = phraseRange(units, p);
    return r ? accentTypeOf(phones, marks, r[0], r[1]) : -1;
  });
}

/** The panel's edits of one sentence: 下がる位置 per phrase (null = as read), 高さ (semitones) and 長さ (factor) per sound. */
export interface SoundEdits { accent: (number | null)[]; pitch: number[]; length: number[] }

export function emptySoundEdits(units: readonly SoundUnit[]): SoundEdits {
  const phrases = units.length ? Math.max(...units.map((u) => u.phrase)) + 1 : 0;
  return { accent: Array<number | null>(phrases).fill(null), pitch: units.map(() => 0), length: units.map(() => 1) };
}

export function hasEdits(e: SoundEdits): boolean {
  return e.accent.some((a) => a !== null) || e.pitch.some((p) => p !== 0) || e.length.some((l) => l !== 1);
}

/** The edits -> speak options: labels only when a 下がる位置 changed; f0Add / durScale per phone. */
export function speakOptionsOf(units: readonly SoundUnit[], phones: readonly string[], marks: readonly number[], e: SoundEdits): { labels?: { phones: string[]; marks: number[] }; f0Add?: number[]; durScale?: number[] } {
  const out: { labels?: { phones: string[]; marks: number[] }; f0Add?: number[]; durScale?: number[] } = {};
  if (e.accent.some((a) => a !== null)) {
    let mk = [...marks];
    e.accent.forEach((a, p) => {
      const r = phraseRange(units, p);
      if (a !== null && r) mk = setAccent(phones, mk, r[0], r[1], a);
    });
    out.labels = { phones: [...phones], marks: mk };
  }
  if (e.pitch.some((v) => v !== 0)) {
    const f = new Array<number>(phones.length).fill(0);
    units.forEach((u, k) => { for (let i = u.a; i < u.b; i++) f[i] = e.pitch[k] ?? 0; });
    out.f0Add = f;
  }
  if (e.length.some((v) => v !== 1)) {
    const d = new Array<number>(phones.length).fill(1);
    units.forEach((u, k) => { for (let i = u.a; i < u.b; i++) d[i] = e.length[k] ?? 1; });
    out.durScale = d;
  }
  return out;
}

/** Mean F0 (semitones) of each sound from the speak result's f0 (edges included); null for punctuation. */
export function soundPitch(units: readonly SoundUnit[], f0WithEdges: readonly number[]): (number | null)[] {
  return units.map((u) => {
    if (u.punct) return null;
    let s = 0;
    for (let i = u.a; i < u.b; i++) s += f0WithEdges[i + 1]!;
    return s / (u.b - u.a);
  });
}

/**
 * 辞書に登録: one word (its sounds idx in the sentence) -> an entry with its reading and always its pitch shape:
 * contour = the made voice's pitch of its sounds about their own mean / the style's range. When read, it is put back on
 * the model's level there and scaled to the range of whatever emotion it is read with, so the shape is relative and
 * applies on top of any emotion and position. length: the word's 長さ (base: the entry the sentence was read with, its
 * lengths are in the made voice already, so the edits multiply them). base's accent / phrase / strength / note are
 * kept; its pitch (semitones added) is in the made pitch, so it is dropped.
 */
export function wordEntryOf(surface: string, reading: string, units: readonly SoundUnit[], idx: readonly number[], edits: SoundEdits,
  madePitch: readonly (number | null)[], range: number, base?: OwnDictEntry): OwnDictEntry {
  const e: OwnDictEntry = { surface: surface.trim(), reading: reading.trim() };
  if (base && base.reading === e.reading) {
    if (base.accent !== undefined) e.accent = base.accent;
    if (base.phrase !== undefined) e.phrase = base.phrase;
    if (base.strength !== undefined) e.strength = base.strength;
  }
  if (base?.note !== undefined) e.note = base.note;
  const p = idx.map((i) => madePitch[i]);
  if (p.every((v): v is number => v !== null && v !== undefined) && range > 0) e.contour = wordContour(p.map((v) => v / range));
  const baseLength = base && base.reading === e.reading ? base.length : undefined;
  const length = idx.map((i, k) => Math.round((edits.length[i] ?? 1) * (baseLength?.[k] ?? 1) * 1000) / 1000);
  if (length.some((v) => v !== 1)) e.length = length.map((v) => Math.max(0.25, Math.min(4, v)));
  return e;
}

/** The y of each sound of an entry's pitch sketch (its contour; undefined when it has none), and where it falls the most. */
export function contourSketch(e: OwnDictEntry): { values: number[]; fallAfter: number } | undefined {
  const c = e.contour;
  if (!c?.length) return undefined;
  let fallAfter = -1;
  let drop = 0.15;
  for (let k = 0; k + 1 < c.length; k++) if (c[k]! - c[k + 1]! > drop) { drop = c[k]! - c[k + 1]!; fallAfter = k; }
  return { values: [...c], fallAfter };
}

/** A short id of entries (the dictionary's version). */
export function dictVersionOf(entries: readonly OwnDictEntry[]): string {
  const s = JSON.stringify(entries);
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return `browser-${entries.length}-${h.toString(16)}`;
}

/** A dictionary contour for a word from relative values of its moras: about their own mean, rounded to 0.001 and
 *  clamped to -3..3. */
export function wordContour(rel: readonly number[]): number[] {
  const m = rel.reduce((a, b) => a + b, 0) / (rel.length || 1);
  return rel.map((v) => Math.max(-3, Math.min(3, Math.round((v - m) * 1000) / 1000)));
}
