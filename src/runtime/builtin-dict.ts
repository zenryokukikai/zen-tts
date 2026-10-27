// Built-in reading dictionary: words the reading model gets wrong in common sentences, applied before the user's
// dictionary (a user entry with the same surface replaces the built-in one).
import type { OwnDictEntry } from './tts-own-prosody.js';

/**
 * 日中 is read ニチチュウ after a comma; it is ニッチュウ (heiban). The longer words below keep the text as it is (reading =
 * surface), so that 日中 inside them is not matched and the model's own reading is kept (一日中, 本日中, 来日中, ...).
 */
const KEEP_AS_WRITTEN = [
  '一日中', '今日中', '明日中', '昼日中', '毎日中', '終日中', '連日中', '後日中', '即日中', '同日中', '初日中', '期日中', '月日中',
  '本日中', '当日中', '翌日中', '前日中', '近日中', '数日中', '平日中', '休日中', '祝日中', '土日中',
  '在日中', '来日中', '訪日中', '滞日中',
] as const;

export const BUILTIN_DICT: readonly OwnDictEntry[] = [
  { surface: '日中', reading: 'ニッチュウ', accent: 0 },
  // お気軽 is read with a fall after キ (オキ＼ガル); it is flat like 気軽
  { surface: 'お気軽', reading: 'オキガル', accent: 0 },
  ...KEEP_AS_WRITTEN.map((w) => ({ surface: w, reading: w })),
];

/** The built-in entries whose surface the user's dictionary does not have, then the user's entries. */
export function withBuiltinDict(user: readonly OwnDictEntry[] = []): OwnDictEntry[] {
  const own = new Set(user.map((e) => e.surface));
  return [...BUILTIN_DICT.filter((e) => !own.has(e.surface)), ...user];
}
