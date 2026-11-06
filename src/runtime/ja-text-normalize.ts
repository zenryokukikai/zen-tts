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

// Amounts (spec §1.5c): a number with a currency word after it or a 万 / 億 / 兆 unit in it is written in kanji numerals
// before the G2P (88866円 -> 八万八千八百六十六円, 3億2000万円 -> 三億二千万円, 1万2千円 -> 一万二千円, ¥1,980 -> 千九百八十円):
// the G2P reads long digit runs badly (ハッピャウ) and kanji numerals well. Runs on text after the separators are dropped
// and the currency signs moved. Left alone: a decimal (1.5万円), a run touching a letter, a digit or '.' (v2, 1.2.0円),
// a zero amount (0円: ゼロ), and anything without a currency word or a large unit (dates, times, years, counters).
const AMOUNT_UNIT = '兆億万千百';
const CURRENCY_WORD = '(?:円|ドル|ユーロ|ウォン|元|ポンド|セント|ルピー|バーツ|ペソ|フラン)';
const AMOUNT = new RegExp(`(?<![0-9A-Za-z.]|[0-9A-Za-z]-)(-?)([0-9]+(?:[${AMOUNT_UNIT}][0-9]+)*[${AMOUNT_UNIT}]?)(?![0-9]|\\.[0-9])(?=(${CURRENCY_WORD})?)`, 'g');

/** Amounts in kanji numerals (spec §1.5c). */
export function readAmounts(s: string): string {
  return s.replace(AMOUNT, (whole, minus: string, chain: string, currency: string | undefined, at: number, all: string) => {
    if (currency === undefined && !/[兆億万]/.test(chain)) return whole;
    // a minus counts only when it is not a hyphen between digits or letters (100-2000円 is a range)
    if (minus && /[0-9A-Za-z]/.test(all[at - 1] ?? '')) return whole;
    if (/^0+$/.test(chain)) return whole;
    if (!LONG_RUN.test(chain)) return whole;
    const kanji = chain.replace(/[0-9]+/g, (d) => numberToKanji(d));
    return (minus ? 'マイナス' : '') + kanji;
  });
}

// Currency amounts read by rule (spec §1.5c, kana form): a whole number (digits, optionally with 万 / 億 / 兆 / 千 / 百 units)
// right before 円 / ドル / ユーロ / ウォン / 元 / ポンド becomes its katakana reading with the sound changes (三百 サンビャク,
// 六百 ロッピャク, 八百 ハッピャク, 三千 サンゼン, 八千 ハッセン, 千万 イッセンマン, 一兆 イッチョウ, 十兆 ジュッチョウ, 4円 ヨエン).
// The reading of a currency amount does not depend on context, so the rule is exact where the G2P misreads long digit runs.
// 千円 is センエン. Not touched: decimals (1.5ドル), runs touching a letter, a digit or '.', other counters (人, 日, 分).
const LONG_RUN = /[0-9]{5}/;
const KANA_DIGIT = ['', 'イチ', 'ニ', 'サン', 'ヨン', 'ゴ', 'ロク', 'ナナ', 'ハチ', 'キュウ'] as const;
const KANA_GROUP_UNIT = ['', 'マン', 'オク', 'チョウ'] as const;
const CURRENCY_KANA: Readonly<Record<string, string>> = { '円': 'エン', 'ドル': 'ドル', 'ユーロ': 'ユーロ', 'ウォン': 'ウォン', '元': 'ゲン', 'ポンド': 'ポンド' };
const KANJI_UNIT_VALUE: Readonly<Record<string, bigint>> = { '兆': 10n ** 12n, '億': 10n ** 8n, '万': 10n ** 4n, '千': 1000n, '百': 100n };
const CURRENCY_AMOUNT = /(?<![0-9A-Za-z.]|[0-9A-Za-z]-)(-?)([0-9]+(?:[兆億万千百][0-9]*)*)(円|ドル|ユーロ|ウォン|元(?![\u3400-\u9fff])|ポンド)(?![0-9])/g;
// the first number of a range of amounts (1,000〜3,000円, 1000から3000円): read like the second, without the currency word
const AMOUNT_RANGE_HEAD = /(?<![0-9A-Za-z.]|[0-9A-Za-z]-)([0-9]+(?:[兆億万千百][0-9]*)*)(?=\s*(?:[〜~～]|から)\s*[0-9]+(?:[兆億万千百][0-9]*)*(?:円|ドル|ユーロ|ウォン|元|ポンド))/g;

/** 1..9999 in katakana before a group unit (マン, オク, チョウ or none). */
function kanaGroup(n: number, unit: string): string {
  const th = Math.floor(n / 1000), hu = Math.floor(n / 100) % 10, te = Math.floor(n / 10) % 10, on = n % 10;
  let out = '';
  if (th) out += th === 1 ? (unit !== '' && hu + te + on === 0 ? 'イッセン' : 'セン') : th === 3 ? 'サンゼン' : th === 8 ? 'ハッセン' : KANA_DIGIT[th] + 'セン';
  if (hu) out += hu === 1 ? 'ヒャク' : hu === 3 ? 'サンビャク' : hu === 6 ? 'ロッピャク' : hu === 8 ? 'ハッピャク' : KANA_DIGIT[hu] + 'ヒャク';
  const chou = unit === 'チョウ';
  if (te) out += (te === 1 ? '' : KANA_DIGIT[te]) + (chou && !on ? 'ジュッ' : 'ジュウ');
  if (on) out += chou && (on === 1 || on === 8) ? (on === 1 ? 'イッ' : 'ハッ') : KANA_DIGIT[on];
  return out + unit;
}

/** A whole amount (BigInt, 1 .. < 10^16) in katakana, or null. */
function amountKana(v: bigint): string | null {
  if (v <= 0n || v >= 10n ** 16n) return null;
  const s = v.toString();
  const groups: number[] = [];
  for (let e = s.length; e > 0; e -= 4) groups.unshift(Number(s.slice(Math.max(0, e - 4), e)));
  let out = '';
  groups.forEach((g, i) => { if (g) out += kanaGroup(g, KANA_GROUP_UNIT[groups.length - 1 - i] ?? ''); });
  return out;
}

/** "3億2000万" / "1万2千" / "3千5百万" / "12345" -> its value, or null when the units are out of order. */
function chainValue(chain: string): bigint | null {
  let total = 0n, section = 0n, lastBig = 10n ** 16n, lastSmall = 10000n;
  for (const m of chain.matchAll(/([0-9]*)([兆億万千百]?)/g)) {
    if (m[0] === '') continue;
    const digits = m[1] ? BigInt(m[1]) : null;
    const u = m[2] ? KANJI_UNIT_VALUE[m[2]]! : 1n;
    if (u >= 10000n) {
      if (u >= lastBig) return null;
      const part = section + (digits ?? 0n);
      total += (part === 0n ? 1n : part) * u;
      section = 0n; lastBig = u; lastSmall = 10000n;
    } else if (u > 1n) {
      if (u >= lastSmall) return null;
      section += (digits ?? 1n) * u; lastSmall = u;
    } else {
      section += digits ?? 0n; lastSmall = 1n;
    }
  }
  return total + section;
}

/** Currency amounts in katakana (spec §1.5c). */
export function readCurrencyAmounts(s: string): string {
  s = s.replace(AMOUNT_RANGE_HEAD, (whole, chain: string, at: number, all: string) => {
    const tail = /^\s*(?:から)\s*([0-9]+(?:[兆億万千百][0-9]*)*)/.exec(all.slice(at + chain.length));
    if (!LONG_RUN.test(chain) && !(tail && LONG_RUN.test(tail[1] ?? ''))) return whole;
    const v = chainValue(chain);
    return v === null ? whole : (v === 0n ? 'ゼロ' : amountKana(v)) ?? whole;
  });
  return s.replace(CURRENCY_AMOUNT, (whole, minus: string, chain: string, cur: string, at: number, all: string) => {
    if (minus && /[0-9A-Za-z]/.test(all[at - 1] ?? '')) return whole;
    if (!LONG_RUN.test(chain)) return whole; // up to 4 digits per run: the G2P reads them right, with their accent
    const v = chainValue(chain);
    if (v === null) return whole;
    let k = v === 0n ? 'ゼロ' : amountKana(v);
    if (k === null) return whole;
    if (cur === '円' && v % 10n === 4n) k = k.replace(/ヨン$/, 'ヨ'); // 4円 ヨエン, 14円 ジュウヨエン
    return (minus ? 'マイナス' : '') + k + (CURRENCY_KANA[cur] ?? cur);
  });
}

// Long numbers before a unit (spec §1.5c): the G2P reads up to 4 digits before a unit right (sound changes included) but
// breaks longer runs (88866メートル ハッピャウ). A number of 5 digits or more right before a unit of this table is read by
// rule: its kana with the unit's sound change on the last element (standard counter patterns). Units where the reading
// depends on context (日, 月, 人, 分, 時, 年, 階, 軒) are not in the table and stay with the G2P.
// k: イッ / ロッ / ハッ / ジュッ / ヒャッ before the unit, except the digits listed in keep (read as is: 6歳 ロクサイ, 100点 ヒャクテン);
// h: like k, and the unit's p / b forms (1本 イッポン, 3本 サンボン, 1000本 センボン, 10000本 イチマンボン).
type SoundChange = 'none' | 'k' | 'h';
interface UnitRule { cls: SoundChange; base: string; p?: string; b?: string; keep?: string }
const UNIT_RULES: Readonly<Record<string, UnitRule>> = {
  'メートル': { cls: 'none', base: 'メートル' }, 'キロメートル': { cls: 'k', base: 'キロメートル' }, 'キロ': { cls: 'k', base: 'キロ' },
  'キログラム': { cls: 'k', base: 'キログラム' }, 'センチ': { cls: 'k', base: 'センチ' }, 'センチメートル': { cls: 'k', base: 'センチメートル' },
  'ミリ': { cls: 'none', base: 'ミリ' }, 'ミリメートル': { cls: 'none', base: 'ミリメートル' }, 'グラム': { cls: 'none', base: 'グラム' },
  'リットル': { cls: 'none', base: 'リットル' }, 'パーセント': { cls: 'k', base: 'パーセント', keep: '1,6,8,100' }, '度': { cls: 'none', base: 'ド' },
  '秒': { cls: 'none', base: 'ビョウ' }, 'ページ': { cls: 'k', base: 'ページ' }, '個': { cls: 'k', base: 'コ' }, '回': { cls: 'k', base: 'カイ' },
  '歳': { cls: 'k', base: 'サイ', keep: '6,100' }, '才': { cls: 'k', base: 'サイ', keep: '6,100' }, '枚': { cls: 'none', base: 'マイ' },
  '本': { cls: 'h', base: 'ホン', p: 'ポン', b: 'ボン' }, '杯': { cls: 'h', base: 'ハイ', p: 'パイ', b: 'バイ' }, '匹': { cls: 'h', base: 'ヒキ', p: 'ピキ', b: 'ビキ' },
  '台': { cls: 'none', base: 'ダイ' }, '件': { cls: 'k', base: 'ケン' }, '点': { cls: 'k', base: 'テン', keep: '6,100' }, '位': { cls: 'none', base: 'イ' },
  '番': { cls: 'none', base: 'バン' },
};
const UNIT_ALT = Object.keys(UNIT_RULES).sort((a, b) => b.length - a.length).join('|');
const LONG_BEFORE_UNIT = new RegExp(`(?<![0-9A-Za-z.]|[0-9A-Za-z]-)([0-9]{5,16})(${UNIT_ALT})(?![0-9])`, 'g');

/** n (5..16 digits) + unit in katakana, the unit's sound change on n's last element. */
function unitKana(v: bigint, unit: string): string | null {
  const r = UNIT_RULES[unit];
  const k = amountKana(v);
  if (!r || k === null) return null;
  if (r.cls === 'none') return k + r.base;
  const keep = new Set((r.keep ?? '').split(','));
  const p = r.p ?? r.base, b = r.b ?? r.base;
  const lastTwo = Number(v % 100n), lastThree = Number(v % 1000n);
  if (/イチ$/.test(k) && !keep.has('1')) return k.slice(0, -2) + 'イッ' + p;
  if (/ロク$/.test(k) && !keep.has('6')) return k.slice(0, -2) + 'ロッ' + p;
  if (/ハチ$/.test(k) && !keep.has('8')) return k.slice(0, -2) + 'ハッ' + p;
  if (/ジュウ$/.test(k) && lastTwo % 10 === 0) return k.slice(0, -3) + 'ジュッ' + p;
  if (/ャク$/.test(k) && lastThree % 100 === 0 && !keep.has('100')) return k.slice(0, -1) + 'ッ' + p;
  if (r.cls === 'h' && (/サン$/.test(k) || /[セゼ]ン$|マン$/.test(k))) return k + b;
  return k + r.base;
}

/** Numbers of 5 digits or more right before a unit of UNIT_RULES, in katakana (spec §1.5c). */
export function readLongUnitNumbers(s: string): string {
  return s.replace(LONG_BEFORE_UNIT, (whole, digits: string, unit: string) => unitKana(BigInt(digits), unit) ?? whole);
}

/** How long amounts and numbers before a unit (5+ digits) are read: 'rule' (spec §1.5c, the default) or 'model' (the G2P). */
export type NumberReading = 'rule' | 'model';
export const NUMBER_READINGS: readonly NumberReading[] = ['rule', 'model'];

/** The speak option `numbers` -> normalizeOwnText's `amounts`. */
export function numbersToAmounts(n: NumberReading | undefined): 'kana' | false {
  if (n === undefined || n === 'rule') return 'kana';
  if (n === 'model') return false;
  throw new RangeError(`zen-tts: numbers must be 'rule' or 'model' (got ${String(n)})`);
}

export interface NormalizeOptions {
  /** romaji and short acronyms by rule, English to lower case (spec §3); default true. false: Latin letters as written. */
  latin?: boolean;
  /** currency amounts and long numbers before a unit (spec §1.5c): 'kana' by rule (readCurrencyAmounts, readLongUnitNumbers),
   *  'kanji' numerals (readAmounts, comparison only); default off here (the engines pass the speak option `numbers`) */
  amounts?: 'kana' | 'kanji' | false;
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
  if (o.amounts === 'kanji') s = readAmounts(s);
  else if (o.amounts === 'kana') s = readLongUnitNumbers(readCurrencyAmounts(s));
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
