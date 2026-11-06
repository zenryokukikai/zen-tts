// English word + CMUdict pronunciation (ARPAbet) -> katakana, following Japanese loanword conventions (docs/SPEC-text-frontend.md
// §3.7). Used at build time only (scripts/latin-words/cmu-words.mjs); the runtime ships the resulting word list.
//
// Rules (each was checked against the curated and neologd lists, see README.md):
// - Vowels: AA ア (オ when spelled o), AE ア (キャ・ギャ after K / G when stressed), AH ア (unstressed: follows the spelling:
//   a ア, e エ, i イ, o オ, u ア), AO オー (オ before a stop when spelled o), AW アウ, AY アイ, EH エ, ER アー, EY エー (エイ at
//   the end of a word), IH イ, IY イー (イ before another vowel or when unstressed inside a word), OW オー (オウ never), OY
//   オイ, UH ウ, UW ウー (ユー after Y).
// - A vowel + R not before a vowel lengthens the vowel (car カー, port ポート); ER is アー.
// - Consonant + vowel: the kana of that row, with ティ ディ トゥ ドゥ for T / D + イ / ウ, シ ジ for S / Z + イ, ファ フィ フェ
//   フォ for F, ウィ ウェ ウォ for W, イェ for Y + エ. V is written in the バ row (ビ for vi), the usual form in IT words
//   (server サーバー, video ビデオ). TH is the サ row (シ for thi), DH the ザ row (ジ for dhi). L and R are the ラ row.
// - Consonant + Y + vowel: the イ kana + small ャ ュ ョ (cute キュート, computer コンピューター).
// - A consonant with no vowel after it: T ト, D ド, CH チ, JH ジ, SH シュ, ZH ジュ, TS ツ, DZ ズ, NG ング (ン before K / G),
//   N ン, M ム (ン before P / B / M), L ル, others + ウ (ク グ プ ブ フ ブ ス ズ).
// - Small ッ before a final or pre-consonant P T K B D G CH SH JH after a stressed short vowel (cat キャット, big ビッグ), and
//   before such a consonant spelled with a double letter or ck (happy ハッピー, ticket チケット is checked in the list).
// - Endings: -tion -sion ション ジョン, -le (consonant + AH0 + L) ル without the schwa (table テーブル).
const VOWELS = new Set(['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']);

// consonant -> kana for a, i, u, e, o
const ROW = {
  K: ['カ', 'キ', 'ク', 'ケ', 'コ'], G: ['ガ', 'ギ', 'グ', 'ゲ', 'ゴ'], S: ['サ', 'シ', 'ス', 'セ', 'ソ'], Z: ['ザ', 'ジ', 'ズ', 'ゼ', 'ゾ'],
  T: ['タ', 'ティ', 'トゥ', 'テ', 'ト'], D: ['ダ', 'ディ', 'ドゥ', 'デ', 'ド'], N: ['ナ', 'ニ', 'ヌ', 'ネ', 'ノ'],
  HH: ['ハ', 'ヒ', 'フ', 'ヘ', 'ホ'], B: ['バ', 'ビ', 'ブ', 'ベ', 'ボ'], P: ['パ', 'ピ', 'プ', 'ペ', 'ポ'], M: ['マ', 'ミ', 'ム', 'メ', 'モ'],
  Y: ['ヤ', 'イ', 'ユ', 'イェ', 'ヨ'], R: ['ラ', 'リ', 'ル', 'レ', 'ロ'], L: ['ラ', 'リ', 'ル', 'レ', 'ロ'], W: ['ワ', 'ウィ', 'ウ', 'ウェ', 'ウォ'],
  F: ['ファ', 'フィ', 'フ', 'フェ', 'フォ'], V: ['バ', 'ビ', 'ブ', 'ベ', 'ボ'], TH: ['サ', 'シ', 'ス', 'セ', 'ソ'], DH: ['ザ', 'ジ', 'ズ', 'ゼ', 'ゾ'],
  SH: ['シャ', 'シ', 'シュ', 'シェ', 'ショ'], ZH: ['ジャ', 'ジ', 'ジュ', 'ジェ', 'ジョ'], CH: ['チャ', 'チ', 'チュ', 'チェ', 'チョ'],
  JH: ['ジャ', 'ジ', 'ジュ', 'ジェ', 'ジョ'], '': ['ア', 'イ', 'ウ', 'エ', 'オ'],
};
const AIUEO = { a: 0, i: 1, u: 2, e: 3, o: 4 };
// consonant -> its kana with no vowel after it
const CODA = { T: 'ト', D: 'ド', CH: 'チ', JH: 'ジ', SH: 'シュ', ZH: 'ジュ', K: 'ク', G: 'グ', P: 'プ', B: 'ブ', F: 'フ', V: 'ブ', S: 'ス', Z: 'ズ', TH: 'ス', DH: 'ズ', L: 'ル', M: 'ム', N: 'ン', HH: 'フ', W: 'ウ', Y: 'イ', R: 'ル', NG: 'ング' };
const SOKUON_AFTER = new Set(['P', 'T', 'K', 'D', 'G', 'CH', 'SH', 'JH']);
const SHORT = new Set(['AE', 'EH', 'IH', 'AH', 'UH', 'AA', 'AO']);
const SMALL = { a: 'ャ', u: 'ュ', o: 'ョ' };
// the イ-column kana of a consonant (for consonant + Y + vowel)
const I_KANA = { K: 'キ', G: 'ギ', S: 'シ', Z: 'ジ', T: 'チ', D: 'デ', N: 'ニ', HH: 'ヒ', B: 'ビ', P: 'ピ', M: 'ミ', R: 'リ', L: 'リ', F: 'フ', V: 'ビ' };

/** Vowel letter groups of a spelling, in order (y is a vowel when not at the start or before a vowel). */
function vowelGroups(word) {
  const out = [];
  const re = /[aeiou]+y?|(?<=[^aeiou])y(?![aeiou])/g;
  let m;
  while ((m = re.exec(word))) out.push(m[0]);
  return out;
}

/** The vowel letter a vowel phone was spelled with ('' when unknown). */
function letterOf(groups, k) { return (groups[k] ?? '')[0] ?? ''; }

/**
 * phones: ARPAbet with stress digits (CMUdict), word: lower-case spelling. Returns katakana.
 */
export function arpabetToKatakana(phones, word) {
  const ph0 = phones.map((p) => ({ base: p.replace(/\d$/, ''), stress: /\d$/.test(p) ? +p.slice(-1) : -1 }));
  // ER before a vowel is a vowel + R (bakery ベーカリー, very)
  const ph = [];
  ph0.forEach((p, k) => {
    const nx = ph0[k + 1];
    if (p.base === 'ER' && nx && VOWELS.has(nx.base)) { ph.push({ base: 'AH', stress: Math.max(p.stress, 1) }, { base: 'R', stress: -1 }); }
    else ph.push(p);
  });
  const nVowels = ph.filter((p) => VOWELS.has(p.base)).length;
  let groups = vowelGroups(word);
  // a silent final e (awesome, make): drop it when it makes one group too many
  if (groups.length === nVowels + 1 && /[^aeiou]e$/.test(word)) groups = groups.slice(0, -1);
  const aligned = groups.length === nVowels;
  // -tion / -sion: SH|ZH AH0 N at the end
  // -le: C AH0 L at the end, spelled ...le
  let out = '';
  let vi = -1; // index of the vowel phone
  const vowelKana = (i) => {
    // returns [column letter, extra] for vowel phone i
    const p = ph[i];
    vi++;
    const letter = aligned ? letterOf(groups, vi) : '';
    const next = ph[i + 1];
    const afterNext = ph[i + 2];
    const rNotVowel = next && next.base === 'R' && !(afterNext && VOWELS.has(afterNext.base));
    const lastVowel = !ph.slice(i + 1).some((q) => VOWELS.has(q.base));
    const beforeVowel = next && VOWELS.has(next.base);
    switch (p.base) {
      case 'AA': {
        if (aligned && /^(aw|au)$/.test(groups[vi] ?? '')) return ['o', 'ー']; // awesome オーサム, sauce
        return [letter === 'o' || (!aligned && /o/.test(word) && !/a/.test(word)) ? 'o' : 'a', rNotVowel ? 'ー' : ''];
      }
      case 'AE': return ['a', rNotVowel ? 'ー' : ''];
      case 'AH': {
        // a final -ia spelled with one schwa (russia, malaysia, tunisia): イア
        if (p.stress === 0 && lastVowel && aligned && /^ia$/.test(groups[vi] ?? '')) return ['i', 'ア'];
        if (p.stress === 0) {
          const prevY = ph[i - 1] && ph[i - 1].base === 'Y';
          const col = { a: 'a', e: 'e', i: 'i', o: 'o', u: prevY ? 'u' : 'a', y: 'i' }[letter] ?? 'a';
          return [col, ''];
        }
        return [letter === 'o' && !/^ou/.test(groups[vi] ?? '') ? 'o' : 'a', ''];
      }
      case 'AO': {
        if (rNotVowel) return ['o', 'ー'];
        const stop = next && SOKUON_AFTER.has(next.base) && !(afterNext && VOWELS.has(afterNext.base));
        if (aligned && /^a$/.test(groups[vi] ?? '') && next && next.base === 'L') return ['o', 'ー']; // also, salt: オール
        if (letter === 'o' || p.stress !== 1 || (!aligned && /o/.test(word) && !/au|aw|al|ou/.test(word))) return ['o', ''];
        return ['o', 'ー'];
      }
      case 'AW': return ['a', next && next.base === 'ER' ? '' : 'ウ'];
      case 'AY': return ['a', 'イ'];
      case 'EH': {
        if (letter === 'a' && !rNotVowel) return ['e', ''];
        return ['e', rNotVowel ? 'ア' : ''];
      }
      case 'ER': return [ph[i - 1] && ph[i - 1].base === 'AW' ? 'wa' : 'a', 'ー'];
      case 'EY': return ['e', lastVowel && !next ? 'イ' : (p.stress === 1 && aligned && /^(ai|ay|ei|ey)$/.test(groups[vi] ?? '') ? 'イ' : 'ー')];
      case 'IH': {
        if (p.stress === 0 && ({ a: 1, e: 1, o: 1 })[letter]) return [letter, ''];
        return ['i', rNotVowel ? 'ア' : ''];
      }
      case 'IY': {
        if (beforeVowel) return ['i', ''];
        if (p.stress === 0 && !lastVowel) return ['i', ''];
        // a final unstressed i spelled i (mini ミニ, taxi タクシー is in the list) or -dy -ty after a consonant (lady, city ティ)
        if (p.stress === 0 && lastVowel && aligned && /^i$/.test(groups[vi] ?? '')) return ['i', ''];
        // a final unstressed y after T / D is short (body ボディ, city シティ, university ユニバーシティ)
        if (p.stress === 0 && lastVowel && !next && ph[i - 1] && ['T', 'D'].includes(ph[i - 1].base)) return ['i', ''];
        return ['i', 'ー'];
      }
      case 'OW': {
        // a final o in a longer word is short (demo デモ, video ビデオ); a one-syllable word keeps it long (show ショー)
        if (lastVowel && !next) return ['o', nVowels === 1 ? 'ー' : ''];
        // a stressed o before -tion (motion モーション, promotion, notion ノーション)
        if (p.stress === 1 && next && next.base === 'SH' && ph[i + 2] && ph[i + 2].base === 'AH') return ['o', 'ー'];
        // a stressed o inside a longer word is usually short when spelled with a single o (robot ロボット, yoga ヨガ)
        return ['o', p.stress >= 1 && (lastVowel || /^(oa|ow|oe)$/.test(aligned ? groups[vi] ?? '' : '') || !aligned) ? 'ー' : ''];
      }
      case 'OY': return ['o', 'イ'];
      case 'UH': return ['u', rNotVowel ? 'ア' : ''];
      case 'UW': return ['u', (p.stress === 1 || lastVowel && !next) ? 'ー' : ''];
      default: return ['a', ''];
    }
  };
  const n = ph.length;
  const DOUBLE = { P: /pp/, T: /tt/, K: /ck|kk|cq/, D: /dd/, G: /gg/, CH: /tch/, JH: /dg/ };
  // ッ before the consonant at i + 1 when it starts a new syllable, the vowel at i is short and stressed, and the spelling doubles it
  const needSokuon = (i) => {
    const v = ph[i], c = ph[i + 1], after = ph[i + 2];
    return !!(v && c && after && VOWELS.has(v.base) && SHORT.has(v.base) && v.stress === 1 && DOUBLE[c.base] && VOWELS.has(after.base) && DOUBLE[c.base].test(word));
  };
  for (let i = 0; i < n; i++) {
    const p = ph[i];
    const b = p.base;
    if (VOWELS.has(b)) {
      const [col, extra] = vowelKana(i);
      out += (col === 'wa' ? 'ワ' : ROW[''][AIUEO[col]]) + extra;
      if (ph[i + 1] && ph[i + 1].base === 'R' && !(ph[i + 2] && VOWELS.has(ph[i + 2].base))) i++; // R absorbed
      else if (needSokuon(i)) out += 'ッ';
      continue;
    }
    const next = ph[i + 1];
    // -tion / -sion
    if ((b === 'SH' || b === 'ZH') && next && next.base === 'AH' && next.stress === 0 && ph[i + 2] && ph[i + 2].base === 'N' && i + 3 === n) {
      out += b === 'SH' ? 'ション' : 'ジョン'; vi++; break;
    }
    // -ful: F AH0 L at the end (beautiful ビューティフル, careful)
    if (b === 'F' && next && next.base === 'AH' && next.stress === 0 && ph[i + 2] && ph[i + 2].base === 'L' && i + 3 === n && word.endsWith('ful')) {
      out += 'フル'; vi++; break;
    }
    // -le: C AH0 L at the end, spelled le
    if (next && next.base === 'AH' && next.stress === 0 && ph[i + 2] && ph[i + 2].base === 'L' && i + 3 === n && word.endsWith('le') && CODA[b]) {
      out += CODA[b] + 'ル'; vi++; break;
    }
    if (b === 'NG') { out += next && (next.base === 'K' || next.base === 'G') ? 'ン' : (next && VOWELS.has(next.base) ? 'ング' : 'ング'); if (next && VOWELS.has(next.base)) { out = out.slice(0, -1); out += 'ン'; i += 0; const [col, extra] = vowelKana(i + 1); out += ROW.G[AIUEO[col]] + extra; i++; } continue; }
    // TS / DZ clusters with no vowel after
    if ((b === 'T' && next && next.base === 'S') && !(ph[i + 2] && VOWELS.has(ph[i + 2].base))) { out += 'ツ'; i++; continue; }
    if ((b === 'D' && next && next.base === 'Z') && !(ph[i + 2] && VOWELS.has(ph[i + 2].base))) { out += 'ズ'; i++; continue; }
    // consonant + Y + vowel
    if (next && next.base === 'Y' && I_KANA[b] && ph[i + 2] && VOWELS.has(ph[i + 2].base)) {
      const [col, extra] = vowelKana(i + 2);
      const small = SMALL[col];
      out += small ? I_KANA[b] + small + extra : ROW[b][AIUEO[col]] + extra;
      if (ph[i + 3] && ph[i + 3].base === 'R' && !(ph[i + 4] && VOWELS.has(ph[i + 4].base))) i++;
      i += 2;
      continue;
    }
    // S / T + W + i: スイ ツイ (sweet スイート, twitter ツイッター)
    if ((b === 'S' || b === 'T') && next && next.base === 'W' && ph[i + 2] && (ph[i + 2].base === 'IH' || ph[i + 2].base === 'IY')) {
      const [, extra] = vowelKana(i + 2);
      out += (b === 'S' ? 'ス' : 'ツ') + 'イ' + extra;
      i += 2;
      if (needSokuon(i)) out += 'ッ';
      continue;
    }
    // K / G + W + vowel: クイ クエ クオ
    if ((b === 'K' || b === 'G') && next && next.base === 'W' && ph[i + 2] && VOWELS.has(ph[i + 2].base)) {
      const [col, extra] = vowelKana(i + 2);
      out += (b === 'K' ? 'ク' : 'グ') + ROW[''][AIUEO[col]] + extra;
      if (ph[i + 3] && ph[i + 3].base === 'R' && !(ph[i + 4] && VOWELS.has(ph[i + 4].base))) i++;
      i += 2;
      continue;
    }
    if (next && VOWELS.has(next.base)) {
      const [col, extra] = vowelKana(i + 1);
      let kana = ROW[b][AIUEO[col]];
      // spelled ew / eu / u for a Y-less UW after N T D (news ニュース, tuesday, duty): the ュ sound
      if (['N', 'T', 'D'].includes(b) && next.base === 'UW' && aligned && /^e/.test(groups[vi] ?? '')) kana = { N: 'ニュ', T: 'テュ', D: 'デュ' }[b];
      if (b === 'K' && next.base === 'AE' && next.stress === 1) kana = 'キャ';
      if (b === 'G' && next.base === 'AE' && next.stress === 1) kana = 'ギャ';
      out += kana + extra;
      if (ph[i + 2] && ph[i + 2].base === 'R' && !(ph[i + 3] && VOWELS.has(ph[i + 3].base))) i++;
      else if (needSokuon(i + 1)) out += 'ッ';
      i++;
      continue;
    }
    // a consonant with no vowel after it
    if (b === 'N') { out += 'ン'; continue; }
    if (b === 'M') { out += next && ['P', 'B', 'M'].includes(next.base) ? 'ン' : 'ム'; continue; }
    // small ッ before a stop (and CH SH JH) after a short vowel, where the stop ends a syllable:
    // - at the end of the word, or before a final -s (app アップ, ticket, kids キッズ, netflix ネットフリックス);
    // - before the consonant of a later stressed syllable (a compound: macbook マックブック, network, upload アップロード);
    // - before another consonant when the spelling doubles the stop (pp tt ck).
    const prev = ph[i - 1];
    const syllableEnd = !next || (['S', 'Z'].includes(next.base) && !ph[i + 2])
      || (!VOWELS.has(next.base) && !['S', 'R', 'L', 'Y', 'W'].includes(next.base) && ['K', 'T', 'P', 'D'].includes(b) && prev && prev.stress === 1 && ph.slice(i + 1).some((q) => VOWELS.has(q.base) && q.stress >= 1))
      || (!VOWELS.has(next.base) && ['K', 'T', 'P'].includes(b) && prev && prev.stress === 1 && DOUBLE[b] && DOUBLE[b].test(word));
    // after an unstressed vowel only T D K CH SH JH (ticket チケット, -ic クリニック), not P G (gossip ゴシップ is in the list)
    const stressOk = prev && (prev.stress >= 1 || ['T', 'D', 'K', 'CH', 'SH', 'JH'].includes(b));
    if (syllableEnd && SOKUON_AFTER.has(b) && VOWELS.has(prev?.base) && SHORT.has(prev.base) && stressOk && !(b === 'G' && prev.base !== 'IH') && !out.endsWith('ー') && !out.endsWith('ッ')) out += 'ッ';
    out += CODA[b] ?? '';
  }
  return out;
}
