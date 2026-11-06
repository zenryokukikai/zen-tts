// Latin-letter words for the G2P model (docs/SPEC-text-frontend.md §3). English words go to the G2P as written: their
// readings are learned by the reading model (no English word list at run time). Japanese words written in romaji become
// hiragana by rule, short acronyms are spelled with the katakana letter names (SDK -> エスディーケー), and URLs / addresses
// stay as written. The only list here is ROMAJI_EXCLUDE: English words that also parse as romaji (game, nose, again), used
// to decide romaji vs English; it holds no readings.
// Copyright (c) 2026 Zenryoku Kikai. Licensed under the ZEN-TTS License (LICENSE.md).
import { ROMAJI_EXCLUDE } from './latin-words.js';

let excludeSet: Set<string> | null = null;

/** English words that parse as romaji but are not read as romaji (romaji-exclude.txt). */
function romajiExcluded(word: string): boolean {
  excludeSet ??= new Set(ROMAJI_EXCLUDE.split('\n'));
  return excludeSet.has(word.toLowerCase());
}

// ---- romaji (Hepburn and Kunrei) -> hiragana ----

const VOWEL: Readonly<Record<string, string>> = { a: 'あ', i: 'い', u: 'う', e: 'え', o: 'お' };

/** Consonant (+ y) and vowel -> hiragana. Longest spelling first. */
const SYLLABLE: Readonly<Record<string, string>> = (() => {
  const t: Record<string, string> = {};
  const row = (c: string, kana: string): void => { [...'aiueo'].forEach((v, i) => { const k = kana[i]; if (k && k !== '・') t[c + v] = k; }); };
  row('k', 'かきくけこ'); row('g', 'がぎぐげご'); row('s', 'さしすせそ'); row('z', 'ざじずぜぞ');
  row('t', 'たちつてと'); row('d', 'だぢづでど'); row('n', 'なにぬねの'); row('h', 'はひふへほ');
  row('b', 'ばびぶべぼ'); row('p', 'ぱぴぷぺぽ'); row('m', 'まみむめも'); row('r', 'らりるれろ');
  row('y', 'や・ゆ・よ'); row('w', 'わ・・・を');
  const yoon = (c: string, i: string): void => { t[c + 'ya'] = i + 'ゃ'; t[c + 'yu'] = i + 'ゅ'; t[c + 'yo'] = i + 'ょ'; };
  yoon('k', 'き'); yoon('g', 'ぎ'); yoon('s', 'し'); yoon('z', 'じ'); yoon('t', 'ち'); yoon('d', 'ぢ'); yoon('n', 'に');
  yoon('h', 'ひ'); yoon('b', 'び'); yoon('p', 'ぴ'); yoon('m', 'み'); yoon('r', 'り'); yoon('j', 'じ');
  Object.assign(t, {
    shi: 'し', sha: 'しゃ', shu: 'しゅ', she: 'しぇ', sho: 'しょ',
    chi: 'ち', cha: 'ちゃ', chu: 'ちゅ', che: 'ちぇ', cho: 'ちょ',
    tsu: 'つ', fu: 'ふ', fa: 'ふぁ', fi: 'ふぃ', fe: 'ふぇ', fo: 'ふぉ',
    ji: 'じ', ja: 'じゃ', ju: 'じゅ', je: 'じぇ', jo: 'じょ', ye: 'いぇ',
  });
  return t;
})();

const SPELLINGS = Object.keys(SYLLABLE).sort((a, b) => b.length - a.length);
const CONSONANT = /^[bcdfghjkmnprstwyz]$/;

/**
 * A lower-case word -> hiragana when the whole word is romaji, else undefined. Rules (spec §3.4): syllables longest
 * first; n before a consonant, an apostrophe or the end is ん (nn before a consonant is one ん); m before b / p / m is ん;
 * a doubled consonant (and tch) is っ; oh before a consonant or the end is おお.
 */
export function romajiToHiragana(word: string): string | undefined {
  const w = word.toLowerCase();
  if (w.length < 2 || !/^[a-z']+$/.test(w)) return undefined;
  let out = '';
  let i = 0;
  while (i < w.length) {
    const c = w[i]!;
    const next = w[i + 1] ?? '';
    if (c === "'") { i++; continue; }
    if (VOWEL[c]) {
      // oh + consonant or end (Ohno, Ohtani): a long o
      if (c === 'o' && next === 'h' && !VOWEL[w[i + 2] ?? ''] && w[i + 2] !== 'y') { out += 'おお'; i += 2; continue; }
      out += VOWEL[c]; i++; continue;
    }
    if (c === 'n' && (next === '' || next === "'" || (CONSONANT.test(next) && next !== 'y'))) {
      out += 'ん';
      i += next === 'n' && !VOWEL[w[i + 2] ?? ''] && w[i + 2] !== 'y' ? 2 : 1;
      continue;
    }
    if (c === 'm' && (next === 'b' || next === 'p' || next === 'm')) { out += 'ん'; i++; continue; }
    if (c === 't' && next === 'c' && w[i + 2] === 'h') { out += 'っ'; i++; continue; }
    if (c === next && CONSONANT.test(c) && c !== 'n') { out += 'っ'; i++; continue; }
    const s = SPELLINGS.find((x) => w.startsWith(x, i));
    if (!s) return undefined;
    out += SYLLABLE[s];
    i += s.length;
  }
  return out;
}

// ---- classification of a Latin run ----

/** acronym: spelled with the letter names; romaji: hiragana; english: passed to the G2P as written. */
export type LatinKind = 'acronym' | 'romaji' | 'english';

/** The katakana name of each letter, as said in Japanese (A エー … Z ゼット). */
const LETTER_NAME: Readonly<Record<string, string>> = {
  a: 'エー', b: 'ビー', c: 'シー', d: 'ディー', e: 'イー', f: 'エフ', g: 'ジー', h: 'エイチ', i: 'アイ', j: 'ジェー', k: 'ケー',
  l: 'エル', m: 'エム', n: 'エヌ', o: 'オー', p: 'ピー', q: 'キュー', r: 'アール', s: 'エス', t: 'ティー', u: 'ユー', v: 'ブイ',
  w: 'ダブリュー', x: 'エックス', y: 'ワイ', z: 'ゼット',
};

/** A run spelled with the letter names (the - and ' inside are dropped): SDK -> エスディーケー. */
export function spellLetters(run: string): string {
  return [...run.toLowerCase()].map((c) => LETTER_NAME[c] ?? '').join('');
}

export interface LatinReading { text: string; kind: LatinKind }

/**
 * One run of letters (with - or ' inside, no other characters) -> how it is read (spec §3.3):
 * 1. all upper case, up to 3 letters: an acronym, spelled with the letter names (SDK エスディーケー, VR ブイアール);
 * 2. romaji (hiragana), for a run of 2 letters or more that parses completely as romaji and is not in the romaji exclusion
 *    list (sushi すし, Yamada やまだ, SUSHI すし; game, nose, again are English);
 * 3. else English: passed to the G2P in lower case (unity, anthropic, html, iphone, wi-fi), as the reading model was
 *    trained on English words (lower case, - and ' kept); the model reads it.
 */
export function readLatinRun(run: string): LatinReading {
  if (!/[a-z]/.test(run) && run.replace(/[-']/g, '').length <= 3) return { text: spellLetters(run), kind: 'acronym' };
  const plain = !/[A-Z]/.test(run.slice(1)) || !/[a-z]/.test(run); // lower case, Capitalised or all capitals
  if (plain && !run.includes('-') && !romajiExcluded(run)) {
    const r = romajiToHiragana(run);
    if (r) return { text: r, kind: 'romaji' };
  }
  return { text: run.toLowerCase(), kind: 'english' };
}

const RUN = /[A-Za-z]+(?:[-'][A-Za-z]+)*/g;
const LETTER = /[A-Za-z]/;

/** Characters that make a run part of a URL, a mail address, a file name or code when next to it. */
const ADDRESS = /[.@/:_\\#=&%+~]/;
const NAME_CHAR = /[A-Za-z0-9]/;

/**
 * Text after NFKC (spec §3.2) -> the same text with each romaji run in hiragana and each short acronym spelled with the
 * letter names; English runs stay as written (the G2P reads them). A run touching a digit or a letter-like neighbour through
 * . @ / : _ and similar (a URL, a mail address, example.com, a_b) is kept as written. Text without Latin letters is
 * returned unchanged.
 */
export function readLatin(text: string): string {
  if (!LETTER.test(text)) return text;
  return text.replace(RUN, (run, at: number) => {
    const b1 = text[at - 1] ?? '', b2 = text[at - 2] ?? '';
    const a1 = text[at + run.length] ?? '', a2 = text[at + run.length + 1] ?? '';
    if ((ADDRESS.test(b1) && NAME_CHAR.test(b2)) || (ADDRESS.test(a1) && NAME_CHAR.test(a2))) return run;
    if (b1 === '/' || a1 === '/' || b1 === '@' || a1 === '@' || (a1 === ':' && a2 === '/')) return run; // also the scheme of a URL (https://)
    return readLatinRun(run).text;
  });
}

/** Macrons and circumflexes on vowels (Tōkyō, Ōsaka, kâ) -> long vowels in romaji (ou, oo, aa, ...), before NFKC. */
export function foldLongVowels(text: string): string {
  return text.replace(/[āâáàīîíìūûúùēêéèōôóòĀÂÁÀĪÎÍÌŪÛÚÙĒÊÉÈŌÔÓÒ]/g, (ch, at: number) => {
    const [base = '', mark = ''] = ch.normalize('NFD');
    const lower = base.toLowerCase();
    if (!'\u0304\u0302'.includes(mark) || mark === '') return base;
    const long = lower === 'o' ? 'u' : lower;
    // the added vowel is upper case only in an all-capitals word (TŌKYŌ -> TOUKYOU, Ōsaka -> Ousaka)
    const word = /[\p{L}]+/u.exec(text.slice(Math.max(0, text.slice(0, at).search(/\p{L}*$/u))))?.[0] ?? ch;
    return base + (base !== lower && word === word.toUpperCase() && word.length > 1 ? long.toUpperCase() : long);
  });
}
