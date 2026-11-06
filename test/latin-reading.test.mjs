// Latin letters in the text front end (docs/SPEC-text-frontend.md §3): the routing of a run (romaji -> hiragana by rule,
// a short acronym -> letter names, English -> the G2P as written, in lower case), the romaji parser, the whole
// normalisation, and that text without Latin letters is unchanged (fixtures/latin-regression.json: 300 Japanese sentences
// of the G2P training corpus with no Latin letters; 60 with Latin letters). English readings come from the G2P model.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { readLatinRun, readLatin, romajiToHiragana, foldLongVowels } from '../dist/runtime/latin-reading.js';
import { normalizeOwnText } from '../dist/runtime/ja-text-normalize.js';

const fixture = JSON.parse(fs.readFileSync(new URL('./fixtures/latin-regression.json', import.meta.url), 'utf8'));

// run -> [kind, text]
const RUNS = {
  // romaji: hiragana by rule
  arigatou: ['romaji', 'ありがとう'], konnichiwa: ['romaji', 'こんにちわ'], sayonara: ['romaji', 'さよなら'], sensei: ['romaji', 'せんせい'],
  sushi: ['romaji', 'すし'], SUSHI: ['romaji', 'すし'], kimono: ['romaji', 'きもの'], karate: ['romaji', 'からて'], tsunami: ['romaji', 'つなみ'],
  matcha: ['romaji', 'まっちゃ'], kanpai: ['romaji', 'かんぱい'], kampai: ['romaji', 'かんぱい'], shinbun: ['romaji', 'しんぶん'],
  "kon'ya": ['romaji', 'こんや'], gakkou: ['romaji', 'がっこう'], Ohno: ['romaji', 'おおの'], Shinkansen: ['romaji', 'しんかんせん'],
  tanaka: ['romaji', 'たなか'], Yamada: ['romaji', 'やまだ'], Suzuki: ['romaji', 'すずき'], chotto: ['romaji', 'ちょっと'],
  // an English word whose katakana is its romaji reading may stay romaji (the same sound)
  banana: ['romaji', 'ばなな'],
  // English words that also parse as romaji: in the exclusion list, so English (to the G2P), not がめ / のせ / だて
  game: ['english', 'game'], made: ['english', 'made'], time: ['english', 'time'], home: ['english', 'home'], nose: ['english', 'nose'],
  again: ['english', 'again'], mike: ['english', 'mike'], date: ['english', 'date'], idea: ['english', 'idea'],
  // place names the G2P was trained on as English words (トーキョー, not ときょ)
  Tokyo: ['english', 'tokyo'], tokyo: ['english', 'tokyo'], TOKYO: ['english', 'tokyo'], osaka: ['english', 'osaka'], ramen: ['english', 'ramen'],
  // English: to the G2P as written, in lower case
  Unity: ['english', 'unity'], Anthropic: ['english', 'anthropic'], Twitter: ['english', 'twitter'], YouTube: ['english', 'youtube'],
  iPhone: ['english', 'iphone'], 'Wi-Fi': ['english', 'wi-fi'], ChatGPT: ['english', 'chatgpt'], "McDonald's": ['english', "mcdonald's"],
  HTML: ['english', 'html'], JSON: ['english', 'json'], meeting: ['english', 'meeting'], hello: ['english', 'hello'], Xyzzq: ['english', 'xyzzq'],
  // acronyms (all capitals, 3 letters or fewer): the katakana letter names
  PDF: ['acronym', 'ピーディーエフ'], NHK: ['acronym', 'エヌエイチケー'], OK: ['acronym', 'オーケー'], AI: ['acronym', 'エーアイ'],
  USB: ['acronym', 'ユーエスビー'], SDK: ['acronym', 'エスディーケー'], VR: ['acronym', 'ブイアール'], XYZ: ['acronym', 'エックスワイゼット'],
};

for (const [w, [kind, text]] of Object.entries(RUNS)) {
  test(`readLatinRun ${w} -> ${kind} ${text}`, () => assert.deepEqual(readLatinRun(w), { text, kind }));
}

test('no English word list at run time: English runs reach the G2P as letters, unchanged but for case', () => {
  for (const w of ['Kubernetes', 'beautiful', 'Perplexity', 'smartwatch']) assert.deepEqual(readLatinRun(w), { text: w.toLowerCase(), kind: 'english' });
});

test('romajiToHiragana: n / nn / n\', m before b p m, small tsu, long vowels, non-romaji', () => {
  assert.equal(romajiToHiragana('ken\'ichi'), 'けんいち');
  assert.equal(romajiToHiragana('kenichi'), 'けにち');
  assert.equal(romajiToHiragana('konnichiwa'), 'こんにちわ');
  assert.equal(romajiToHiragana('sanpo'), 'さんぽ');
  assert.equal(romajiToHiragana('sampo'), 'さんぽ');
  assert.equal(romajiToHiragana('zasshi'), 'ざっし');
  assert.equal(romajiToHiragana('itchi'), 'いっち');
  assert.equal(romajiToHiragana('tsukue'), 'つくえ');
  assert.equal(romajiToHiragana('fuji'), 'ふじ');
  assert.equal(romajiToHiragana('kyou'), 'きょう');
  assert.equal(romajiToHiragana('ohayou'), 'おはよう');
  assert.equal(romajiToHiragana('tsuzuki'), 'つずき'); // zu is ず (the same sound as づ)
  assert.equal(romajiToHiragana('si'), 'し'); // Kunrei
  assert.equal(romajiToHiragana('tyotto'), 'ちょっと');
  assert.equal(romajiToHiragana('hello'), undefined); // ll + o: no
  assert.equal(romajiToHiragana('twitter'), undefined);
  assert.equal(romajiToHiragana('a'), undefined); // one letter
  assert.equal(romajiToHiragana('xyz'), undefined);
});

test('foldLongVowels: macrons and circumflexes are long vowels', () => {
  assert.equal(foldLongVowels('Tōkyō'), 'Toukyou');
  assert.equal(foldLongVowels('Ōsaka'), 'Ousaka');
  assert.equal(foldLongVowels('kâ'), 'kaa');
  assert.equal(foldLongVowels('café'), 'cafe');
  assert.equal(foldLongVowels('TŌKYŌ'), 'TOUKYOU');
});

const TEXTS = [
  ['Twitterでarigatouって送った', 'twitterでありがとうって送った'],
  ['iPhone15を買った', 'iphone15を買った'],
  ['PDFをDLして', 'ピーディーエフをディーエルして'],
  ['NHKのニュース', 'エヌエイチケーのニュース'],
  ['Tōkyō駅', 'とうきょう駅'],
  ['sushiとramenが好き', 'すしとramenが好き'],
  ['今日はOKです', '今日はオーケーです'],
  ['Unity の SDK を使います。', 'unityのエスディーケーを使います.'],
  // addresses: kept as written (then the normalisation drops / : @)
  ['https://example.com/abc を見て', 'httpsexample.comabcを見て'],
  ['メールは info@zenryoku.work まで', 'メールはinfozenryoku.workまで'],
  ['ファイルは report_final.pdf です', 'ファイルはreportfinal.pdfです'],
];
for (const [t, want] of TEXTS) test(`normalizeOwnText ${t}`, () => assert.equal(normalizeOwnText(t), want));

test('latin: false keeps the 1.1.0 behaviour', () => {
  assert.equal(normalizeOwnText('Twitterでarigatou', { latin: false }), 'Twitterでarigatou');
});

test('text without Latin letters: the same output with and without the Latin step (300 corpus sentences)', () => {
  assert.equal(fixture.japanese.length, 300);
  for (const t of fixture.japanese) assert.equal(normalizeOwnText(t), normalizeOwnText(t, { latin: false }), t);
  for (const t of fixture.japanese) assert.equal(readLatin(t.normalize('NFKC')), t.normalize('NFKC'), t);
});

test('sentences with Latin letters: only the Latin runs change (romaji / acronyms to kana, English to lower case)', () => {
  for (const t of fixture.latin) {
    const a = normalizeOwnText(t);
    const b = normalizeOwnText(t, { latin: false });
    const strip = (s) => s.replace(/[A-Za-z'-]+|[ァ-ヴーぁ-ゖ]+/g, '');
    assert.equal(strip(a), strip(b), t);
  }
});
