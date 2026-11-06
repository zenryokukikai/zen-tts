// Numbers go to the reading model (G2P) as digits (docs/SPEC-text-frontend.md §1.5, since 1.2.0): the normaliser keeps 0-9 and
// only rewrites what the training data rewrote (separators, currency signs, ranges, %, the decimal point).
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { normalizeOwnText, readDigitNumbers } from '../dist/runtime/ja-text-normalize.js';

const n = (s) => normalizeOwnText(s);

test('numbers: digits stay digits (no kanji numerals, no counter table)', () => {
  assert.equal(n('10分待ってください。'), '10分待ってください.');
  assert.equal(n('3本ください'), '3本ください');
  assert.equal(n('2026年11月5日'), '2026年11月5日');
  assert.equal(n('3分の2'), '3分の2');
  assert.equal(n('十分です。'), '十分です.');
});

test('numbers: separators, currency, ranges, percent and the decimal point as in the training data', () => {
  assert.equal(n('1,280円'), '1280円');
  assert.equal(n('$5です'), '5ドルです');
  assert.equal(n('¥1,000'), '1000円');
  assert.equal(n('1〜3人'), '1から3人');
  assert.equal(n('10%'), '10パーセント');
  assert.equal(n('3.14'), '3点14');
});

test('numbers: full-width digits become half-width (NFKC)', () => {
  assert.equal(n('１０分'), '10分');
});

test('numbers: phone numbers and postal codes are read digit by digit by rule (§1.5b)', () => {
  assert.equal(n('電話番号は090-1234-5678です。'), '電話番号はゼロキューゼロ,イチニーサンヨン,ゴーロクナナハチです.');
  assert.equal(n('03-1234-5678'), 'ゼロサン,イチニーサンヨン,ゴーロクナナハチ');
  assert.equal(n('0120-123-456'), 'ゼロイチニーゼロ,イチニーサン,ヨンゴーロク');
  assert.equal(n('〒123-4567'), 'イチニーサン,ヨンゴーロクナナ');
  assert.equal(n('〒1234567'), 'イチニーサン,ヨンゴーロクナナ');
  assert.equal(n('郵便番号は123-4567です'), '郵便番号はイチニーサン,ヨンゴーロクナナです');
  assert.equal(n('０９０－１２３４－５６７８'), 'ゼロキューゼロ,イチニーサンヨン,ゴーロクナナハチ');
  for (const h of ['\u2010', '\u2212', '\u2013', 'ー']) assert.equal(n(`090${h}1234${h}5678`), 'ゼロキューゼロ,イチニーサンヨン,ゴーロクナナハチ', h);
});

test('numbers: everything else stays for the G2P (plain runs, dates, versions, decimals, ranges, amounts)', () => {
  assert.equal(n('1234'), '1234');
  assert.equal(n('88866円'), '88866円');
  assert.equal(n('2026-11-05'), '2026-11-05');
  assert.equal(n('1.2.0'), '1点2点0');
  assert.equal(n('3.14'), '3点14');
  assert.equal(n('1〜3'), '1から3');
  assert.equal(n('8,000円'), '8000円');
  assert.equal(n('1,234,567円'), '1234567円');
  assert.equal(n('12,345人'), '12345人');
  assert.equal(n('1-2-3'), '1-2-3');
  assert.equal(n('10-20'), '10-20');
  assert.equal(n('100-2000円'), '100-2000円');
  assert.equal(n('価格は300-5000円です'), '価格は300-5000円です');
  assert.equal(n('090-1234-5678宛'), 'ゼロキューゼロ,イチニーサンヨン,ゴーロクナナハチ宛');
});

test('numbers: the digit-by-digit rule leaves text without such numbers unchanged (fixture sentences)', () => {
  const fx = JSON.parse(fs.readFileSync(new URL('./fixtures/latin-regression.json', import.meta.url), 'utf8'));
  for (const s of [...fx.japanese, ...fx.latin]) assert.equal(readDigitNumbers(s.normalize('NFKC')), s.normalize('NFKC'), s);
});
