// Builds data/latin-words/cmu-auto.tsv: katakana for common English words that are not in curated.tsv / neologd.tsv, made
// from their CMUdict pronunciation by the loanword rules of arpabet-katakana.mjs (docs/SPEC-text-frontend.md §3.7).
// Usage: node scripts/latin-words/cmu-words.mjs <cmudict.dict> <3esl.txt>   (then node scripts/build-latin-words.mjs)
// An English word that also parses as romaji (date, fine, idea) is added too, so it is read as English; the few Japanese
// words in the English list (karate, kimono, …) are left out and stay romaji.
// Inputs (not in the repository, see README.md): cmudict.dict (CMUdict, BSD-style) and 3esl.txt (Alan Beale's 12dicts
// "3esl" list of about 21,000 common English words, public domain).
import fs from 'node:fs';
import path from 'node:path';
import { arpabetToKatakana } from './arpabet-katakana.mjs';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..', '..');
const dir = path.join(root, 'data', 'latin-words');

const cmu = new Map();
for (const l of fs.readFileSync(process.argv[2], 'utf8').split('\n')) {
  const [w0, ...ph] = l.split(' ');
  if (!w0) continue;
  const w = w0.replace(/\(\d+\)$/, '');
  if (!cmu.has(w)) cmu.set(w, ph.filter((p) => p && !p.startsWith('#'))); // the first (main) pronunciation
}
const common = new Set(fs.readFileSync(process.argv[3], 'latin1').split(/\r?\n/).map((x) => x.trim()).filter((x) => /^[a-z]{2,}$/.test(x)));
const rows = (f) => fs.readFileSync(path.join(dir, f), 'utf8').split('\n').filter((l) => l && !l.startsWith('#'));
const listed = new Set([...rows('curated.tsv'), ...rows('neologd.tsv')].map((l) => l.split('\t')[0].toLowerCase()));

// Japanese words in the English list: they stay romaji (the others that parse as romaji are English words such as date,
// fine, idea, into, join, mine, rate: read as English, not 「だて」「ふぃね」).
const JAPANESE = new Set(['karate', 'kimono', 'geisha', 'tofu', 'honcho', 'sake', 'sushi', 'tsunami', 'sumo', 'origami', 'bonsai', 'judo', 'kendo', 'ninja', 'samurai', 'emoji', 'anime', 'manga', 'otaku', 'kawaii', 'ramen', 'futon', 'tatami', 'wasabi', 'miso', 'shogun', 'zen', 'haiku', 'kanji', 'sensei', 'dojo', 'tycoon', 'typhoon']);
const out = [];
let romaji = 0;
for (const w of [...common].sort()) {
  if (listed.has(w) || !cmu.has(w)) continue;
  if (JAPANESE.has(w)) { romaji++; continue; }
  const k = arpabetToKatakana(cmu.get(w), w);
  if (!/^[ァ-ヴー]+$/.test(k)) throw new Error(`${w}: ${k}`);
  out.push(`${w}\t${k}`);
}
const head = `# Made by scripts/latin-words/cmu-words.mjs (2026-11-03): common English words (12dicts 3esl, public domain) that are not in
# curated.tsv or neologd.tsv, read from their CMUdict pronunciation (BSD-style) by the loanword rules of arpabet-katakana.mjs.
# Japanese words in the English list (karate, kimono, …) are left out (romaji). Do not edit by hand: fix a word in curated.tsv (it wins).
# surface<TAB>reading
`;
fs.writeFileSync(path.join(dir, 'cmu-auto.tsv'), head + out.join('\n') + '\n');
console.log(JSON.stringify({ common: common.size, added: out.length, japaneseLeftAsRomaji: romaji }));
