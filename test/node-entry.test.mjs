// Unit tests of the Node entry's pure parts (no models needed): sentence split, option mapping, curve sampling, WAV.
import test from 'node:test';
import assert from 'node:assert/strict';
import { splitSentences, emotionBlend, wavBytes, SAMPLE_RATE, EMOTION_CURVES } from '../dist/node/index.js';
import { curveFn, curvePoints, engineOptionsOf, CURVE_POINTS } from '../dist/runtime/speak-options.js';

test('splitSentences keeps the end marks and drops empty pieces', () => {
  assert.deepEqual(splitSentences('こんにちは。今日はいい天気ですね！ 散歩しましょうか？'), ['こんにちは。', '今日はいい天気ですね！', '散歩しましょうか？']);
  assert.deepEqual(splitSentences('一行目\n二行目。'), ['一行目', '二行目。']);
  assert.deepEqual(splitSentences('。。'), []);
  assert.deepEqual(splitSentences(''), []);
});

test('splitSentences cuts a long sentence after a comma before the limit', () => {
  const s = 'あ'.repeat(50) + '、' + 'い'.repeat(50) + '。';
  const parts = splitSentences(s, 60);
  assert.equal(parts.length, 2);
  assert.equal(parts[0], 'あ'.repeat(50) + '、');
  assert.equal(parts.join(''), s);
  assert.ok(splitSentences('う'.repeat(130), 60).every((p) => p.length <= 60));
});

test('emotionBlend normalises amounts; empty or zero is neutral', () => {
  assert.deepEqual(emotionBlend(undefined), { neutral: 1, happy: 0, sad: 0, angry: 0, apologetic: 0 });
  assert.deepEqual(emotionBlend({ happy: 0 }), { neutral: 1, happy: 0, sad: 0, angry: 0, apologetic: 0 });
  const b = emotionBlend({ happy: 70, neutral: 30 });
  assert.ok(Math.abs(b.happy - 0.7) < 1e-12 && Math.abs(b.neutral - 0.3) < 1e-12);
});

test('engineOptionsOf maps change to a blendAt curve; curve sampling is read back at the points', () => {
  const o = engineOptionsOf({ emotion: { neutral: 1 }, change: { to: { angry: 1 }, curve: 'linear' }, strength: 1.2 });
  assert.equal(o.strength, 1.2);
  assert.ok(o.blendAt);
  const pts = curvePoints(o.blendAt);
  assert.equal(pts.length, CURVE_POINTS);
  const f = curveFn(pts);
  for (const x of [0, 0.25, 0.5, 1]) assert.ok(Math.abs(f(x).angry - o.blendAt(x).angry) < 1e-9, `x=${x}`);
  assert.ok(EMOTION_CURVES.includes('ease'));
});

test('wavBytes writes a 16-bit mono WAV at 44.1 kHz', () => {
  const w = wavBytes(new Float32Array([0, 1, -1, 2]));
  const v = new DataView(w.buffer);
  assert.equal(new TextDecoder().decode(w.subarray(0, 4)), 'RIFF');
  assert.equal(v.getUint32(24, true), SAMPLE_RATE);
  assert.equal(v.getUint32(40, true), 8);
  assert.equal(v.getInt16(46, true), 0x7fff);
  assert.equal(v.getInt16(50, true), 0x7fff); // clipped
});
