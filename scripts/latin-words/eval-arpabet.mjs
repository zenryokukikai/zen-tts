// Checks arpabet-katakana.mjs against the curated and neologd lists (the readings people use): exact matches and the
// typical differences. Usage: node scripts/latin-words/eval-arpabet.mjs <cmudict.dict> [n diffs]
import fs from 'node:fs';
import path from 'node:path';
import { arpabetToKatakana } from './arpabet-katakana.mjs';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..', '..');
const cmu = new Map();
for (const l of fs.readFileSync(process.argv[2], 'utf8').split('\n')) {
  const [w0, ...ph] = l.split(' ');
  if (!w0) continue;
  const w = w0.replace(/\(\d+\)$/, '');
  if (!cmu.has(w)) cmu.set(w, []);
  cmu.get(w).push(ph.filter((p) => p && p !== '#' && !p.startsWith('#')));
}
// the merged list (curated wins, as in build-latin-words.mjs), one row per word
const merged = new Map();
for (const f of ['neologd.tsv', 'curated.tsv']) {
  for (const l of fs.readFileSync(path.join(root, 'data', 'latin-words', f), 'utf8').split('\n')) {
    if (!l || l.startsWith('#')) continue;
    const [w, k] = l.split('\t');
    if (/^[a-z]+$/.test(w) && cmu.has(w)) merged.set(w, [w, k, f]);
  }
}
const only = process.env.ONLY;
const rows = [...merged.values()].filter((r) => !only || r[2] === only);
// loose: ignore ー and the ッ / ヴ-バ / small kana differences
const loose = (s) => s.replace(/[ーッ]/g, '').replace(/ヴァ/g, 'バ').replace(/ヴィ/g, 'ビ').replace(/ヴ/g, 'ブ');
let exact = 0, near = 0;
const diffs = [];
for (const [w, k] of rows) {
  const outs = cmu.get(w).map((ph) => arpabetToKatakana(ph, w));
  if (outs.includes(k)) exact++;
  else if (outs.some((o) => loose(o) === loose(k))) { near++; diffs.push(['near', w, k, outs[0]]); }
  else diffs.push(['diff', w, k, outs[0]]);
}
console.log(JSON.stringify({ words: rows.length, exact, exactPct: +(100 * exact / rows.length).toFixed(1), nearOnly: near, exactOrNearPct: +(100 * (exact + near) / rows.length).toFixed(1) }));
const n = +(process.argv[3] ?? 40);
const pick = (t) => diffs.filter((d) => d[0] === t);
for (const t of ['near', 'diff']) {
  const d = pick(t);
  const step = Math.max(1, Math.floor(d.length / n));
  console.log(`-- ${t} (${d.length})`);
  for (let i = 0; i < d.length; i += step) console.log(`${d[i][1]}\t${d[i][2]}\t${d[i][3]}`);
}
