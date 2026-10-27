// zen-tts/node check (needs models/ and, for native, `pnpm run build:native`):
//  1. parity: the browser's CPU path (CpuOwnTts on cpu/tts-cpu.wasm with the worker's curve sampling) vs zen-tts/node
//     with every kernel build that loads here, on the test sentences (scripts/sentences.json) + an emotion curve + a dictionary entry;
//  2. RTF on 1 thread (synthesis time / audio length), per kernel, median over the sentences after a warm-up;
//  3. speakStream on a 3-sentence text: chunks, first chunk latency.
// usage: node scripts/node-check.mjs [out.json] [reps]
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { CpuOwnTts } from '../dist/runtime/cpu-tts.js';
import { nsfExportToWeights } from '../dist/runtime/nsf-weights.js';
import { withBuiltinDict } from '../dist/runtime/builtin-dict.js';
import { curveFn, curvePoints, engineOptionsOf } from '../dist/runtime/speak-options.js';
import { ZenTtsNode } from '../dist/node/index.js';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const out = process.argv[2] ?? '/tmp/zen-tts-node-check.json';
const reps = Number(process.argv[3] ?? 3);
const buf = (p) => { const b = fs.readFileSync(path.join(root, p)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const json = (p) => JSON.parse(fs.readFileSync(path.join(root, p), 'utf8'));
const sents = json('scripts/sentences.json');
const list = [...sents.short, ...sents.medium, ...sents.long];
const cases = [
  ...list.map((text) => ({ text, o: {} })),
  { text: list[5], o: { emotion: { neutral: 1 }, change: { to: { angry: 1 }, curve: 'ease' } } },
  { text: list[6], o: { emotion: { happy: 70, sad: 30 }, strength: 1.3 } },
  { text: '本日は日中も気温が上がりません。', o: { dict: [{ surface: '気温', reading: 'キオン', accent: 0 }] } },
];
const sha = (a) => crypto.createHash('sha256').update(Buffer.from(a.buffer, a.byteOffset, a.byteLength)).digest('hex').slice(0, 16);

// 1a. the browser CPU path (cpu-worker.ts: CpuOwnTts.speak with the curve as 129 points, the built-in dictionary)
const ref = await CpuOwnTts.create(buf('cpu/tts-cpu.wasm'), json('models/tts/own.json'), buf('models/tts/own.bin'), nsfExportToWeights(json('models/decoder/nsf.json'), buf('models/decoder/nsf.bin')));
const browserCpu = (text, o) => {
  const { blendAt, ...rest } = engineOptionsOf(o);
  const opts = blendAt ? { ...rest, blendAt: curveFn(curvePoints(blendAt)) } : rest;
  return ref.speak(text, { ...opts, dict: withBuiltinDict(rest.dict) }).audio;
};
const refAudio = cases.map((c) => browserCpu(c.text, c.o));

const result = { node: process.version, cpu: (fs.readFileSync('/proc/cpuinfo', 'utf8').match(/model name\s*:\s*(.*)/) ?? [])[1], kernels: {} };
for (const kernel of ['avx512', 'avx2', 'sse41', 'wasm']) {
  let tts;
  try { tts = await ZenTtsNode.load({ modelsPath: path.join(root, 'models'), ...(kernel === 'wasm' ? { backend: 'wasm' } : { backend: 'native', kernel }) }); }
  catch (e) { result.kernels[kernel] = { error: String(e.message ?? e) }; console.log(kernel, 'skip:', e.message); continue; }
  // parity
  let same = 0, maxAbs = 0;
  const diffs = [];
  for (const [i, c] of cases.entries()) {
    const a = (await tts.speak(c.text, c.o)).pcm, b = refAudio[i];
    let m = a.length === b.length ? 0 : Infinity;
    if (a.length === b.length) for (let j = 0; j < a.length; j++) { const d = Math.abs(a[j] - b[j]); if (d > m) m = d; }
    if (m === 0) same++; else diffs.push({ i, lenA: a.length, lenB: b.length, maxAbs: m, a: sha(a), b: sha(b) });
    if (m > maxAbs) maxAbs = m;
  }
  // RTF (1 thread): warm-up, then reps per sentence, median per sentence and over sentences
  await tts.speak('こんにちは。');
  const rtf = [];
  for (const text of list) {
    const runs = [];
    let sec = 0;
    for (let r = 0; r < reps; r++) { const t0 = performance.now(); const x = await tts.speak(text); runs.push(performance.now() - t0); sec = x.pcm.length / 44100; }
    runs.sort((p, q) => p - q);
    rtf.push(runs[Math.floor(runs.length / 2)] / 1000 / sec);
  }
  rtf.sort((p, q) => p - q);
  // speakStream on 3 sentences
  const text3 = `${sents.short[0]}${sents.medium[0]}${sents.short[1]}`;
  const t0 = performance.now();
  const chunks = [];
  for await (const c of tts.speakStream(text3)) chunks.push({ index: c.index, text: c.text, sec: +(c.pcm.length / 44100).toFixed(3), atMs: +(performance.now() - t0).toFixed(1) });
  result.kernels[kernel] = {
    loaded: tts.kernel, parity: { cases: cases.length, bitExact: same, maxAbs, diffs }, rtfMedian: +rtf[Math.floor(rtf.length / 2)].toFixed(4),
    rtfMin: +rtf[0].toFixed(4), rtfMax: +rtf[rtf.length - 1].toFixed(4), memoryMB: +(tts.memoryBytes / 1048576).toFixed(1), stream3: chunks,
  };
  console.log(kernel, `bit-exact ${same}/${cases.length}`, 'maxAbs', maxAbs, 'RTF median', result.kernels[kernel].rtfMedian, 'first chunk ms', chunks[0]?.atMs);
}
fs.writeFileSync(out, JSON.stringify(result, null, 1));
console.log('wrote', out);
