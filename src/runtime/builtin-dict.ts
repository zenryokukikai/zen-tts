// Built-in reading dictionary: empty. Readings come from the reading model (G2P); the runtime carries no built-in reading
// list (docs/SPEC-text-frontend.md §3.6). The user's dictionary is applied as before.
import type { OwnDictEntry } from './tts-own-prosody.js';

export const BUILTIN_DICT: readonly OwnDictEntry[] = [];

/** The built-in entries whose surface the user's dictionary does not have, then the user's entries. */
export function withBuiltinDict(user: readonly OwnDictEntry[] = []): OwnDictEntry[] {
  const own = new Set(user.map((e) => e.surface));
  return [...BUILTIN_DICT.filter((e) => !own.has(e.surface)), ...user];
}
