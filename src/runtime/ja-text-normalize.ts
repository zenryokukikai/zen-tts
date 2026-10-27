// Japanese text normalisation for the G2P model, implemented from docs/SPEC-text-frontend.md §1.
// Copyright (c) 2026 Zenryoku Kikai. Licensed under the ZEN-TTS License (LICENSE.md).

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

/** Reads one numeral (digits, optionally `.digits`) in Japanese. Throws when `s` is not a numeral. */
export function numberToKanji(s: string): string {
  const m = NUMERAL.exec(s);
  if (!m) throw new Error(`not a numeral: ${s}`);
  return readNumeral(m[1] ?? '', m[2]);
}

function readNumeral(intDigits: string, fracDigits: string | undefined): string {
  const whole = readInteger(intDigits);
  const frac = (fracDigits ?? '').replace(/0+$/, '');
  if (whole === null) return fracDigits === undefined ? '' : '.';
  if (frac === '') return whole;
  let out = whole + '点';
  for (const ch of frac) out += DIGIT_NAMES[ch.charCodeAt(0) - 48];
  return out;
}

/** Japanese text -> the character set the G2P model reads (spec §1). */
export function normalizeOwnText(text: string): string {
  let s = text.normalize('NFKC');
  s = s.replace(/(?<=[0-9]),(?=[0-9]{3})/g, '');
  s = s.replace(/([$£¥€])([0-9.]*[0-9])/g, (_m, sign: string, run: string) => run + (CURRENCY_UNIT[sign] ?? ''));
  s = s.replace(/・・・/g, '...');
  s = s.replace(/([0-9]+)(?:\.([0-9]+))?/g, (_m, i: string, f: string | undefined) => readNumeral(i, f));
  let out = '';
  for (const ch of s) {
    if (TO_APOSTROPHE.has(ch)) out += "'";
    else if (ch === '。' || ch === '$' || ch === '\n') out += '.';
    else if (TO_COMMA.has(ch)) out += ',';
    else if (TO_HYPHEN.has(ch)) out += '-';
    else if (TO_LONG_MARK.has(ch)) out += 'ー';
    else if (isAllowed(ch.codePointAt(0) ?? 0)) out += ch;
  }
  return out;
}
