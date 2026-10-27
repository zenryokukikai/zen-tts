// Kana reading -> phonemes with the model's mora table, implemented from docs/SPEC-text-frontend.md §2.
// Copyright (c) 2026 Zenryoku Kikai. Licensed under the ZEN-TTS License (LICENSE.md).

/** Mora table: katakana spelling (1 or 2 chars) -> [consonant or '', vowel / N / q]. */
export type MoraTable = Readonly<Record<string, readonly [string, string]>>;

const LONG_VOWEL_MARK = 'ー';

/** Hiragana (U+3041..U+3096, ゝ, ゞ) -> katakana; every other character unchanged. */
export function hiraganaToKatakana(s: string): string {
  let out = '';
  for (const ch of s) {
    const cp = ch.codePointAt(0) ?? 0;
    out += (cp >= 0x3041 && cp <= 0x3096) || cp === 0x309d || cp === 0x309e ? String.fromCodePoint(cp + 0x60) : ch;
  }
  return out;
}

/** Phonemes of a kana reading; the long-vowel mark repeats the previous phoneme. Throws on a character not in the table. */
export function kanaToPhones(reading: string, kana: MoraTable): string[] {
  const units = [...hiraganaToKatakana(reading)];
  const phones: string[] = [];
  for (let pos = 0; pos < units.length; ) {
    const one = units[pos] ?? '';
    if (one === LONG_VOWEL_MARK) {
      const last = phones[phones.length - 1];
      if (last !== undefined) phones.push(last);
      pos += 1;
      continue;
    }
    const pair = one + (units[pos + 1] ?? '');
    const key = pos + 1 < units.length && pair in kana ? pair : one in kana ? one : null;
    if (key === null) throw new Error(`読みに使えない文字です：${one}`);
    const [consonant, vowel] = kana[key] ?? ['', ''];
    if (consonant !== '') phones.push(consonant);
    phones.push(vowel);
    pos += key.length;
  }
  return phones;
}
