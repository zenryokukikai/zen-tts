// Currency amounts and long numbers before a unit, read by rule (normalizeOwnText amounts: 'kana', spec §1.5c).
import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeOwnText, readCurrencyAmounts, readLongUnitNumbers } from '../dist/runtime/ja-text-normalize.js';

const k = (s) => normalizeOwnText(s, { amounts: 'kana' });

test('amounts: off by default (digits stay digits)', () => {
  assert.equal(normalizeOwnText('88866円です。'), '88866円です.');
});

test('amounts: long currency amounts become katakana with the sound changes', () => {
  assert.equal(k('88866円です。'), 'ハチマンハッセンハッピャクロクジュウロクエンです.');
  assert.equal(k('12,345,678円'), 'センニヒャクサンジュウヨンマンゴセンロッピャクナナジュウハチエン');
  assert.equal(k('¥36,800'), 'サンマンロクセンハッピャクエン');
  assert.equal(k('$12,000'), 'イチマンニセンドル');
  assert.equal(k('10000000円'), 'イッセンマンエン');
  assert.equal(k('1000000000000円'), 'イッチョウエン');
  assert.equal(k('10000000000000円'), 'ジュッチョウエン');
  assert.equal(k('80000000円'), 'ハッセンマンエン');
  assert.equal(k('30000円'), 'サンマンエン');
  assert.equal(k('3億20000円'), 'サンオクニマンエン');
  assert.equal(k('10004円'), 'イチマンヨエン');
  assert.equal(k('12345ユーロ'), 'イチマンニセンサンビャクヨンジュウゴユーロ');
  assert.equal(k('-50000円'), 'マイナスゴマンエン');
});

test('amounts: up to 4 digits per run stay digits (the G2P reads them right, with their accent)', () => {
  for (const t of ['500円', '1,980円', '3億2000万円', '1万2千円', '12万3400円', '千円', '0円']) assert.equal(k(t), normalizeOwnText(t));
});

test('amounts: a range of amounts reads both ends', () => {
  assert.equal(k('10,000〜30,000円'), 'イチマンからサンマンエン');
  assert.equal(k('1000から30000円'), 'センからサンマンエン');
  assert.equal(k('1,000〜3,000円'), '1000から3000円');
});

test('amounts: not touched (phone, postal, dates, versions, decimals, ranges with hyphens, other counters, 元 in a word)', () => {
  for (const t of ['電話番号は090-1234-5678です。', '〒123-4567', '2026年11月5日', 'v12345円', '12345.5円', '10000-20000円', '12345人', '12345日', '12345分', '30000元素'])
    assert.equal(k(t), normalizeOwnText(t), t);
});

test('units: a long number before a unit of the table reads with the unit sound change', () => {
  assert.equal(readLongUnitNumbers('88866メートル'), 'ハチマンハッセンハッピャクロクジュウロクメートル');
  assert.equal(readLongUnitNumbers('43776キロ'), 'ヨンマンサンゼンナナヒャクナナジュウロッキロ');
  assert.equal(readLongUnitNumbers('10000本'), 'イチマンボン');
  assert.equal(readLongUnitNumbers('12341本'), 'イチマンニセンサンビャクヨンジュウイッポン');
  assert.equal(readLongUnitNumbers('33333匹'), 'サンマンサンゼンサンビャクサンジュウサンビキ');
  assert.equal(readLongUnitNumbers('20010杯'), 'ニマンジュッパイ');
  assert.equal(readLongUnitNumbers('12348回'), 'イチマンニセンサンビャクヨンジュウハッカイ');
  assert.equal(readLongUnitNumbers('10100個'), 'イチマンヒャッコ');
  assert.equal(readLongUnitNumbers('50000歳'), 'ゴマンサイ');
  assert.equal(k('12345%'), 'イチマンニセンサンビャクヨンジュウゴパーセント');
});

test('units: short numbers and context-dependent counters stay with the G2P', () => {
  for (const t of ['1234メートル', '10キロ', '3本', '12345人', '12345日', '12345分', '12345年', '12345時間']) assert.equal(k(t), normalizeOwnText(t), t);
});

test('currency: readCurrencyAmounts on its own', () => {
  assert.equal(readCurrencyAmounts('88866円'), 'ハチマンハッセンハッピャクロクジュウロクエン');
});
