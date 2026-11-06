// Japanese text normalisation for the G2P model, implemented from docs/SPEC-text-frontend.md §1.
// Copyright (c) 2026 Zenryoku Kikai. Licensed under the ZEN-TTS License (LICENSE.md).
import { foldLongVowels, readLatin } from './latin-reading.js';

const DIGIT_NAMES = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'] as const;
// Units of successive 4-digit groups, least significant first (spec §1.7).
const GROUP_UNITS = ['', '万', '億', '兆', '京', '垓', '秭', '穣', '溝', '澗', '正', '載', '極'] as const;
const PLACE_UNITS = ['千', '百', '十', ''] as const;

const CURRENCY_UNIT: Readonly<Record<string, string>> = { '$': 'ドル', '£': 'ポンド', '¥': '円', '€': 'ユーロ' };

const TO_APOSTROPHE = new Set([..."'\"()[]", '‘', '’', '“', '”', '「', '」', '《', '》', '【', '】']);
const TO_COMMA = new Set(['、', '・', '·']);
const TO_HYPHEN = new Set([
  '\u2010', '\u2012', '\u2013', '\u2014', '\u2015', '\u2043', '\u2212', '\u2500', '\u2501', '\u02d7', '\u23af', '\u23e4', '\u2e3a', '\u2e3b',
]);
const TO_LONG_MARK = new Set(['~', '〜']);

function isAllowed(cp: number): boolean {
  if ((cp >= 0x41 && cp <= 0x5a) || (cp >= 0x61 && cp <= 0x7a)) return true;
  if (cp === 0x21 || cp === 0x3f || cp === 0x27 || cp === 0x2c || cp === 0x2d || cp === 0x2e) return true;
  if ((cp >= 0x370 && cp <= 0x3ff) || (cp >= 0x1f00 && cp <= 0x1fff)) return true;
  if ((cp >= 0x3040 && cp <= 0x3098) || cp === 0x309d || cp === 0x309e) return true;
  if (cp === 0x3005) return true;
  if ((cp >= 0x30a0 && cp <= 0x30fa) || (cp >= 0x30fc && cp <= 0x30fe)) return true;
  return (cp >= 0x3400 && cp <= 0x4dbf) || (cp >= 0x4e00 && cp <= 0x9fff);
}

/** One 4-digit group (1..9999) in kanji; a leading 1 is silent before 千/百/十. */
function readGroup(group: string): string {
  const padded = group.padStart(4, '0');
  let out = '';
  for (let p = 0; p < 4; p++) {
    const d = padded.charCodeAt(p) - 48;
    if (d === 0) continue;
    const unit = PLACE_UNITS[p] ?? '';
    out += (d === 1 && unit !== '' ? '' : DIGIT_NAMES[d]) + unit;
  }
  return out;
}

/** Integer digit string -> kanji, or null when the value is 10^51 or more. */
function readInteger(digits: string): string | null {
  const s = digits.replace(/^0+/, '');
  if (s === '') return DIGIT_NAMES[0];
  const groupCount = Math.ceil(s.length / 4);
  if (groupCount > GROUP_UNITS.length) return null;
  const head = s.length - (groupCount - 1) * 4;
  let out = '';
  for (let g = 0; g < groupCount; g++) {
    const part = g === 0 ? s.slice(0, head) : s.slice(head + (g - 1) * 4, head + g * 4);
    if (/^0+$/.test(part)) continue;
    const unit = GROUP_UNITS[groupCount - 1 - g] ?? '';
    out += (part.replace(/^0+/, '') === '1' && unit !== '' ? '一' : readGroup(part)) + unit;
  }
  return out;
}

const NUMERAL = /^([0-9]+)(?:\.([0-9]+))?$/;

/** Reads one numeral (digits, optionally `.digits`) in Japanese kanji. Throws when `s` is not a numeral. */
export function numberToKanji(s: string): string {
  const m = NUMERAL.exec(s);
  if (!m) throw new Error(`not a numeral: ${s}`);
  const whole = readInteger(m[1] ?? '');
  const frac = (m[2] ?? '').replace(/0+$/, '');
  if (whole === null) return m[2] === undefined ? '' : '.';
  if (frac === '') return whole;
  let out = whole + '点';
  for (const ch of frac) out += DIGIT_NAMES[ch.charCodeAt(0) - 48];
  return out;
}

// Phone-style digit names (spec §1.5b): a number read digit by digit (090-1234-5678, 〒123-4567).
const DIGIT_KANA = ['ゼロ', 'イチ', 'ニー', 'サン', 'ヨン', 'ゴー', 'ロク', 'ナナ', 'ハチ', 'キュー'] as const;
// Hyphen-like characters between digit groups (after NFKC: -, ‐, ‑, ‒, –, —, −, ー).
const NUM_HYPHEN = '[-\u2010\u2011\u2012\u2013\u2014\u2212\u30fc]';
const HYPHEN_GROUPS = new RegExp(`(?<![0-9A-Za-z.\\-])[0-9]+(?:${NUM_HYPHEN}[0-9]+)+(?![0-9A-Za-z]|\\.[0-9])`, 'g');
const KANJI = /[\u3400-\u4dbf\u4e00-\u9fff]/;
const POSTAL_MARK = new RegExp(`〒\\s*([0-9]{3})(?:${NUM_HYPHEN})?([0-9]{4})(?![0-9])`, 'g');

const digitsToKana = (groups: readonly string[]): string =>
  groups.map((g) => [...g].map((d) => DIGIT_KANA[d.charCodeAt(0) - 48]).join('')).join('、');

/** A date written with hyphens (2026-11-05, 2026-1-5, 2026-11): left to the G2P. */
function isHyphenDate(groups: readonly string[]): boolean {
  const [y, m, d] = groups;
  if (groups.length < 2 || groups.length > 3 || y?.length !== 4 || !m || m.length > 2) return false;
  const mm = Number(m);
  if (mm < 1 || mm > 12) return false;
  if (d === undefined) return true;
  const dd = Number(d);
  return d.length <= 2 && dd >= 1 && dd <= 31;
}

/**
 * Digit groups joined by hyphens that are read digit by digit (spec §1.5b): a group starting with 0 with 6 or more digits in
 * all (090-1234-5678, 03-1234-5678, 0120-123-456), the postal-code shape 123-4567, or 10 or more digits in all; not a
 * date (2026-11-05), and (when not starting with 0) not right before a kanji (100-2000円: a range or an amount). Everything
 * else (1-2-3, 3-1, 10-20, 1234, 88866円) is left to the G2P.
 */
function readsDigitByDigit(groups: readonly string[]): boolean {
  if (isHyphenDate(groups)) return false;
  const total = groups.reduce((n, g) => n + g.length, 0);
  if (groups[0]!.startsWith('0') && total >= 6) return true;
  if (groups.length === 2 && groups[0]!.length === 3 && groups[1]!.length === 4) return true;
  return total >= 10;
}

/** Phone numbers and postal codes -> katakana digit names, groups joined by 、 (spec §1.5b). */
export function readDigitNumbers(s: string): string {
  s = s.replace(POSTAL_MARK, (_m, a: string, b: string) => digitsToKana([a, b]));
  return s.replace(HYPHEN_GROUPS, (run, at: number, all: string) => {
    const groups = run.split(new RegExp(NUM_HYPHEN));
    // a run not starting with 0 right before a kanji is a range or an amount (100-2000円, 10-20人): left to the G2P
    if (!run.startsWith('0') && KANJI.test(all[at + run.length] ?? '')) return run;
    return readsDigitByDigit(groups) ? digitsToKana(groups) : run;
  });
}

export interface NormalizeOptions {
  /** romaji and short acronyms by rule, English to lower case (spec §3); default true. false: Latin letters as written. */
  latin?: boolean;
}

/**
 * Japanese text -> the character set the G2P model reads (spec §1). Digits stay digits: the G2P (1.2.0 and later) reads
 * numbers itself (10分 ジュップン, 十分 ジュウブン, 3分の2 サンブンノニ). Before it: thousand separators are dropped,
 * a currency sign moves after the number ($5 -> 5ドル), a range between numbers is から (1〜3 -> 1から3), % is パーセント
 * and a decimal point between digits is 点 (3.14 -> 3点14), as in the training data. Phone numbers and postal codes
 * (090-1234-5678, 〒123-4567) are read digit by digit by rule (readDigitNumbers).
 */
export function normalizeOwnText(text: string, o: NormalizeOptions = {}): string {
  let s = o.latin === false ? text.normalize('NFKC') : readLatin(foldLongVowels(text.normalize('NFKC')));
  s = readDigitNumbers(s);
  s = s.replace(/(?<=[0-9]),(?=[0-9]{3})/g, '');
  s = s.replace(/([$£¥€])([0-9][0-9.]*)/g, (_m, sign: string, run: string) => run + (CURRENCY_UNIT[sign] ?? ''));
  s = s.replace(/(?<=[0-9])\s*[〜~～]\s*(?=[0-9])/g, 'から');
  s = s.replace(/%/g, 'パーセント');
  s = s.replace(/(?<=[0-9])\.(?=[0-9])/g, '点');
  s = s.replace(/・・・/g, '...');
  let out = '';
  for (const ch of s) {
    if (TO_APOSTROPHE.has(ch)) out += "'";
    else if (ch === '。' || ch === '$' || ch === '\n') out += '.';
    else if (TO_COMMA.has(ch)) out += ',';
    else if (TO_HYPHEN.has(ch)) out += '-';
    else if (TO_LONG_MARK.has(ch)) out += 'ー';
    else if (isAllowed(ch.codePointAt(0) ?? 0) || (ch >= '0' && ch <= '9')) out += ch;
  }
  return out;
}
