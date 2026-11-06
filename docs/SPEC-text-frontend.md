# Text front end: behavioural specification

This document specifies three steps the runtime runs before the G2P model:

1. **Text normalisation** (`normalizeOwnText`): it turns arbitrary Japanese text into the restricted character set the G2P model reads.
2. **Kana reading to phonemes** (`kanaToPhones`): it turns a dictionary reading written in kana into the phoneme sequence the predictor uses.
3. **Latin-letter words** (`readLatin`, part of step 1): English words and Japanese words written in romaji become kana.

They are implemented in `src/runtime/ja-text-normalize.ts`, `src/runtime/kana-phones.ts` and `src/runtime/latin-reading.ts`.

The rules use:
- Unicode NFKC;
- the reading rules of Japanese numerals;
- the Open JTalk mora table that ships in `models/tts/own.json` (`kana`).

## 1. Text normalisation

`normalizeOwnText(text: string): string` applies the steps below in this order.

### 1.1 Compatibility folding
Apply Unicode NFKC to the whole string. This step:
- folds full-width ASCII and half-width katakana;
- composes kana with voiced marks;
- expands ligatures, circled and parenthesised numbers, and squared words (㌔ → キロ, ㍻ → 平成);
- turns `…` into `...` and `‥` into `..`;
- turns superscript digits into ASCII digits.

From here on, "digit" means the ASCII digits `0`–`9` only. Other scripts' digits are not numbers, and are dropped in 1.6.

### 1.1a Latin-letter words
Unless the caller turns it off (`latin: false`), §3 runs here, on the NFKC text, before every step below. Text without
ASCII letters is not changed by it.

### 1.2 Thousands separators
A comma `,` is deleted when the character right before it is a digit **and** the three characters right after it are digits.

Examples:
- `1,000` → `1000`
- `12,345,678` → `12345678`
- `1,2345` → `12345`
- `1,00`, `1,23`, `12,34` and `3,4` stay as they are.
- `x,000` stays as it is: there is no digit before the comma.

This step comes before 1.6. So a `、` or `・` between digits is never treated as a separator: `1、000` → `一,零`.

### 1.3 Currency
These signs are currency signs. Each one names a unit:

| Sign | Unit |
|---|---|
| `$` | ドル |
| `£` | ポンド |
| `¥` | 円 |
| `€` | ユーロ |

When a currency sign is directly followed by a numeric run, the sign is removed and its unit word is placed right after the run. A numeric run is the longest sequence of digits and `.`, as long as it ends with a digit.

Examples:
- `$5` → `5ドル`
- `$1.5.3` → `1.5.3ドル`
- `$..5` → `..5ドル`
- `$5..` → `5ドル..` (the run ends at the last digit)
- `$1,5` → `1ドル,5`. Step 1.2 did not remove that comma, so the run stops at it.
- `1$5` → `15ドル`. Nothing prevents a digit right before the sign.

A sign that is not followed by a numeric run is left alone. In 1.6, a lone `$` becomes `.`, and the other three lone signs are dropped.

### 1.4 Katakana middle dots
After the currency step:
- every run of three `・` (U+30FB) becomes `...`, counting from the start of the run;
- any `・` left over becomes `,` later, in 1.6.

Examples:
- `・・・` → `...`
- `・・` → `,,`
- `・・・・` → `...,`
- `・・・・・・` → `......`

A space between dots breaks the run, and is removed later.

This step runs after the currency step. So `$・・・0` keeps its `$`: the `$` is not followed by a numeric run when the currency step runs. The `$` then becomes `.` (→ `....零`).

### 1.5 Numbers (since 1.2.0: digits are read by the model)
Digits stay digits; the reading model (G2P, 1.2.0 and later) has `0`-`9` in its vocabulary and was trained on number text labelled
from pyopenjtalk on the digit form (`10分` ジュップン, `3本` サンボン, `1杯` イッパイ, `2026年11月5日`, `1280円`), with hand
labels where pyopenjtalk is wrong (`3分の2` サンブンノニ) and decimals digit by digit after 点. There is no counter table and no conversion to kanji numerals. Only these rewrites remain,
the same as in the training data:
- a thousands separator between digits is dropped (`1,280` → `1280`);
- a currency sign before a number moves after it as a word (`$5` → `5ドル`, `¥1,000` → `1000円`, `€` ユーロ, `£` ポンド);
- `〜` / `~` between numbers is `から` (`1〜3人` → `1から3人`);
- `%` is `パーセント`;
- a `.` between digits is `点` (`3.14` → `3点14`).
The written `十分` (ジュウブン) is not a number and is read as text.

### 1.5b Phone numbers and postal codes (read by rule, since 1.2.0)
A number whose form shows it is read digit by digit is rewritten before the steps above into katakana digit names, the groups
joined by `、` (a short pause): `0` ゼロ, `1` イチ, `2` ニー, `3` サン, `4` ヨン, `5` ゴー, `6` ロク, `7` ナナ, `8` ハチ, `9` キュー
(`090-1234-5678` → ゼロキューゼロ、イチニーサンヨン、ゴーロクナナハチ; `〒123-4567` → イチニーサン、ヨンゴーロクナナ). That is:
- digit groups joined by hyphens (`-`, `‐`, `‑`, `‒`, `–`, `—`, `−`, `ー`; full-width forms after NFKC), not touching a letter, a
  digit or a `.`-number on either side, when
  - the first group starts with `0` and there are 6 or more digits in all (`03-1234-5678`, `0120-123-456`), or
  - there are two groups of 3 and 4 digits (the postal-code shape, `123-4567`), or
  - there are 10 or more digits in all;
  and it is not a date (`2026-11-05`, `2026-11`: 4 digits, a month 1-12 and a day 1-31 are left to the model), and, when it does
  not start with `0`, it is not right before a kanji (`100-2000円`, a range or an amount, is left to the model);
- `〒` and 7 digits, with or without a hyphen (`〒1234567`); the `〒` itself is dropped.
Everything else (`1234`, `88866円`, `1-2-3`, `10-20`, `1.2.0`) is left to the model.

### 1.5c Long amounts and numbers before a unit (read by rule, since 1.3.0)
Speak option `numbers` (`'rule'`, the default, or `'model'`; also a load option). With `'rule'`, after §1.2-1.3 and before the
decimal point step, katakana by rule (`readCurrencyAmounts`, `readLongUnitNumbers` in `src/runtime/ja-text-normalize.ts`):
- a currency amount (円, ドル, ユーロ, ウォン, 元, ポンド) with a run of 5 or more digits, alone or in a 万 / 億 / 兆 chain, and both
  ends of a range of such amounts (`88866円` → ハチマンハッセンハッピャクロクジュウロクエン, `10,000〜30,000円` → イチマンからサンマンエン);
- a run of 5 or more digits right before a unit of the table (メートル, キロ, グラム, パーセント, 本, 杯, 匹, 個, 回, 歳, ... with the
  counter sound changes: `43776キロ` → …ナナジュウロッキロ, `12341本` → …ヨンジュウイッポン).
Runs of up to 4 digits, decimals, ranges with a hyphen and the counters whose reading depends on the context (日, 月, 人, 分,
時, 年, 階, 軒) are left to the model. `'model'` leaves all numbers to the model (as 1.2.x).

### 1.6 Character classes
After the numbers, map the following characters, then delete every character that is not allowed (digits `0`-`9` are allowed).

**Mapped to `'`** (quotes and brackets):
- `'` `"` `(` `)` `[` `]`
- `‘` `’` `“` `”`
- `「` `」` `《` `》` `【` `】`

**Mapped to `.`:**
- `。` (ideographic full stop)
- a lone `$`
- line feed U+000A (one `.` per line feed)

**Mapped to `,`:**
- `、` (ideographic comma)
- `・` (left over from 1.4)
- `·` (U+00B7)

**Mapped to `-`:** U+2010, U+2012–U+2015, U+2043, U+2212, U+2500, U+2501, U+02D7, U+23AF, U+23E4, U+2E3A and U+2E3B (hyphens, dashes, minus, box-drawing horizontals).

**Mapped to `ー`:** `~` and `〜` (wave dash).

**Allowed, kept as they are:**
- ASCII `A`–`Z`, `a`–`z`, `!`, `?`, `'`, `,`, `-`, `.`
- Greek: U+0370–U+03FF and U+1F00–U+1FFF
- Hiragana block U+3040–U+3098, plus `ゝ` `ゞ`
- `々`
- Katakana U+30A0–U+30FA and U+30FC–U+30FE (`ー`, `ヽ`, `ヾ`)
- CJK ideographs U+3400–U+4DBF and U+4E00–U+9FFF

**Deleted:** everything else. This includes:
- spaces, tabs, carriage return and other control characters;
- `『』〈〉〔〕{}«»‹›„‚`, `〇`, `〆`;
- `%#&*+=<>/\:;@^_`|`;
- Latin letters with diacritics after NFKC (`é`, `Š`, `ß`);
- other scripts' letters and digits;
- emoji, and ideographs outside the two ranges above.

Nothing else is changed. Runs of punctuation are kept as they are (`..`, `,,`, `''`, `!?`, `...?`).

### 1.7 Reading a numeral
(Not used by the runtime since 1.2.0; kept for `numberToKanji`, which tools use. Numbers are read by the G2P, §1.5.)
A numeral is `I` or `I.F`, where `I` and `F` are digit strings.

**The integer part `I`.** Leading zeros are ignored. If the value is 0, it reads `零`. Otherwise:
- Split the value into groups of four digits from the right.
- Reading from the most significant group, give each non-zero group its reading followed by the group's unit:
  - (none), 万, 億, 兆, 京, 垓, 秭, 穣, 溝, 澗, 正, 載, 極
- Groups of `0000` are skipped.
- Inside a group, the thousands, hundreds, tens and ones digits read as the digit plus `千`, `百`, `十`, or nothing. Zero digits are skipped.
- A digit `1` is not read before `千`, `百` or `十`. So 1000 is `千`, 110 is `百十`, and 10 is `十`.
- A whole group equal to 1 still reads `一` before its unit: 10000 is `一万`, and 100000000 is `一億`.

The digit names are 零 一 二 三 四 五 六 七 八 九.

Examples:
- `1000` → 千
- `11000` → 一万千
- `10000000` → 千万
- `12345678` → 千二百三十四万五千六百七十八
- `100010001` → 一億一万一

Values of 10^51 or more cannot be named with these units. Such a numeral is replaced by:
- `.` if it has a fraction part;
- nothing otherwise.

**The fraction part `F`.** Trailing zeros are dropped. If nothing is left, the numeral reads as its integer part only (`1.0` → `一`, `1,000.00` → `千`). Otherwise the reading is: the integer reading, then `点`, then each fraction digit read on its own by its digit name.

Examples:
- `0.05` → 零点零五
- `15541.08` → 一万五千五百四十一点零八
- `100.001` → 百点零零一

The fraction is read exactly as written. It is never rounded through a binary floating-point value.

Numerals of any length are read from their digits, so an integer part of 2^53 or more with a fraction, for example `12345678901234567.5`, is also read exactly.

`numberToKanji(s)` reads one numeral string `s` by these rules. If `s` is not a numeral, it throws.


### 1.9 End mark for the reading model
The reading model was trained on sentences ending with punctuation (96% end with `.`). Text without a final mark of the
punctuation set is read with a `.` after it, and the phones of that `.` are dropped again. Without this, a short text was
read badly, even into a consonant with no vowel (`二色` → `n i e o k`, shown as 「ニエオkundefined」 by a kana display).

## 2. Kana reading to phonemes

`kanaToPhones(reading: string, kana: Record<string, [string, string]>): string[]`

- `kana` is the mora table from the model file. It maps a katakana spelling of one or two characters to a pair: a consonant phoneme (empty for vowel-only morae, `ン`, and `ッ`) and a vowel phoneme (`N` for `ン`, `q` for `ッ`).
- The reading is first converted to katakana. Hiragana U+3041–U+3096 and `ゝ` `ゞ` move up by 0x60 code points (`ゕ` → `ヵ`, `ゝ` → `ヽ`). No other character changes.

Then scan left to right:
1. If the next two characters form a key in the table, use that key. Otherwise, if the next character alone is a key, use it.
2. A matched key appends its consonant, when it is non-empty, and then its vowel.
3. `ー` (the long-vowel mark) repeats the last phoneme appended so far. If nothing has been appended yet, it adds nothing.
4. Any other character is an error: `Error('読みに使えない文字です：' + ch)`, where `ch` is that character after the katakana conversion.

Examples, with the shipped table:
- `ぜんりょくきかい` → z e N ry o k u k i k a i
- `キット` → k i q t o
- `コーカ` → k o o k a
- `ッー` → q q
- `ーア` → a
- `フュ` → f u y u (`フュ` is not a key, so it reads as `フ` then `ュ`)

The same katakana conversion is used by the dictionary step, `applyOwnDict`, when it puts a reading into the text that the G2P model reads.

## 3. Latin-letter words

`readLatin(text)` (after `foldLongVowels`) runs as step 1.1a. Since 1.2.0 **English readings come from the reading model**
(G2P): it was trained on English words in Japanese sentences (the word lists of §3.5 as training labels), so an English run
reaches it as letters, in lower case, and the model reads it. The runtime has no English word list. Two kinds of runs are
still rewritten by rule: romaji (Japanese written in Latin letters) becomes hiragana (§3.4), and a short all-capitals acronym is
spelled with the katakana letter names (§3.8). Addresses (§3.2) are kept as written.

### 3.1 Long vowels written with a mark
`foldLongVowels` runs before `readLatin`. A vowel with a macron or circumflex (`ā â ī î ū û ē ê ō ô`, either case) becomes
the vowel followed by its long form: `o` by `u`, others by themselves (`Tōkyō` → `Toukyou`, `kâ` → `kaa`). The added
vowel is upper case only in an all-capitals word (`TŌKYŌ` → `TOUKYOU`). Acute and grave accents are dropped (`café` → `cafe`).

### 3.2 Runs
A run is a maximal sequence of ASCII letters with single `-` or `'` between letters (`Wi-Fi`, `McDonald's`, `kon'ya`). A run
is kept as written when it is part of an address:
- the character right before it is one of `. @ / : _ \ # = & % + ~` and the one before that is a letter or digit;
- or the character right after it is one of those and the one after that is a letter or digit;
- or it is directly next to `/` or `@`.

So URLs, mail addresses, file names (`report_final.pdf`) and `example.com` are kept; step 1.6 then removes `/ : @ _`.

### 3.3 Reading a run
1. **Acronym**: all capitals, 3 letters or fewer (`-` and `'` not counted) is spelled with the letter names (§3.8: `PDF`
   ピーディーエフ, `SDK` エスディーケー, `VR` ブイアール).
2. **Romaji**: a lower-case, Capitalised or all-capitals run without `-` that parses completely by §3.4 and is not in the romaji
   exclusion list (§3.5) becomes hiragana (`sushi` すし, `Yamada` やまだ, `SUSHI` すし, `kon'ya` こんや).
3. **English**: everything else goes to the G2P as written, in lower case (`Unity` → `unity`, `Anthropic`, `HTML`, `iPhone`,
   `Wi-Fi`, `McDonald's`); the model gives the reading (ユニティ, アンスロピック, …). A word the model has not learned may be
   read imperfectly; a user dictionary entry fixes it.

Examples: `game` → `game` (English; in the exclusion list, so not がめ), `Twitterでarigatou` → `twitterでありがとう`,
`Unity の SDK を使います。` → `unityのエスディーケーを使います.`, `Tokyo` → `tokyo` (the model reads トーキョー).

### 3.4 Romaji
`romajiToHiragana(word)`: the word in lower case, two letters or more, letters and `'` only. Left to right:
- a vowel `a i u e o` is あいうえお; `oh` before a consonant (not `y`) or the end is おお (`Ohno`);
- `n` before the end, `'`, or a consonant other than `y` is ん; `nn` there is one ん (`konnichiwa`); `m` before `b p m` is ん;
- a doubled consonant (not `n`) and `tch` are っ;
- otherwise the longest syllable of the Hepburn and Kunrei tables (`kya`, `shi`, `chi`, `tsu`, `fu`, `ji`, `si`, `ti`, `tya`,
  `fa`, `je`, ...); `zu` and `du` are ず and づ;
- anything else: not romaji (the whole word).

### 3.5 Data
`src/runtime/latin-words.ts` holds one list, `ROMAJI_EXCLUDE`: English words that parse as romaji and must not be read as romaji
(no readings). It is generated by `scripts/build-latin-words.mjs` from `data/latin-words/`:
- `romaji-exclude.txt`: English words (CMUdict) that parse as romaji but whose CMUdict vowels do not fit that romaji and
  whose Japanese reading in neologd is not the romaji one (nose, again, mike);
- plus the words of the English lists below whose katakana is not exactly their romaji reading (date デート, not だて; tokyo
  トーキョー, not ときょ). A listed word whose katakana is its romaji reading (banana バナナ) stays romaji: the same sound.

The English lists are the G2P's training labels for English words (not used at run time):
- `curated.tsv`: written for ZEN-TTS (services, brands, IT and everyday loanwords; wins on conflicts);
- `neologd.tsv`: extracted from mecab-ipadic-neologd (Apache-2.0; THIRD_PARTY_NOTICES.md), kept only for English words
  (in CMUdict) whose reading starts with a kana that fits the first CMUdict phone, minus a reviewed list of homophones and names;
- `cmu-auto.tsv`: common English words (the 12dicts 3esl list, public domain) not in the two lists above, read from their
  CMUdict pronunciation by the loanword rules of §3.7 (lowest priority).
How these were made: `scripts/latin-words/`. Priority on a conflict: curated > neologd > cmu-auto.

### 3.6 No built-in reading dictionary (since 1.2.0)
The runtime has no built-in reading list: `BUILTIN_DICT` (`src/runtime/builtin-dict.ts`) is empty and only the user's dictionary
is applied. Words that 1.1.0 and the demo fixed by a list (日中 ニッチュウ, お気軽 オキガル, polite words with お such as お兄様 /
お祖母ちゃん / お米, single-kanji words such as 雷 カミナリ / 唇 クチビル / 鳩 ハト, numbers with a counter such as 10分 ジュップン)
are read by the reading model, which was trained on them in sentences.

### 3.7 English pronunciation to katakana (training labels)
`scripts/latin-words/arpabet-katakana.mjs` turns a CMUdict pronunciation (ARPAbet with stress) and the spelling into
katakana, following the usual Japanese loanword forms:
- vowels: AA ア (オ when spelled o), AE ア (キャ ギャ after a stressed K G), AH ア (unstressed: by the spelling a ア e エ i イ o オ),
  AO オー (オ when spelled o or unstressed), AW アウ, AY アイ, EH エ, ER アー, EY エー (エイ at the end or spelled ai/ay/ei/ey and
  stressed), IH イ, IY イー (イ when unstressed inside a word, before a vowel, or a final unstressed -i / -ty / -dy), OW オー (short when
  unstressed, or stressed but spelled with a single o inside the word, or final in a longer word), OY オイ, UH ウ, UW ウー
  (short when unstressed inside the word);
- a vowel + R not before a vowel is long (car カー, port ポート); ER before a vowel is a vowel + R (bakery ベーカリー);
- consonant + vowel: the kana of that row, with ティ ディ トゥ ドゥ, シ ジ for S Z + イ, ファ フィ フェ フォ, ウィ ウェ ウォ, イェ;
  **V is the バ row** (server サーバー, video ビデオ; the common form in IT words, ヴ is not used); **TH is the サ row**, DH the ザ row;
  L and R are the ラ row; S T + W + i are スイ ツイ (sweet スイート, twitter ツイッター); N T D + UW spelled e… are ニュ テュ デュ (news);
- consonant + Y + vowel: the イ kana + small ャ ュ ョ (cute キュート, computer コンピューター);
- a consonant with no vowel: T ト, D ド, CH チ, JH ジ, SH シュ, ZH ジュ, TS ツ, DZ ズ, NG ング (ン before K G), N ン, M ム (ン before
  P B M), L ル, others + ウ (ク グ プ ブ フ ス ズ);
- endings: -tion ション, -sion ジョン, consonant + -le ル (table テーブル), -ful フル;
- small ッ before a stop (and CH SH JH) after a short vowel where the stop ends a syllable: at the end or before a final -s (app
  アップ, kids キッズ), before the consonant of a later stressed syllable (macbook マックブック), or before a consonant when the
  spelling doubles it; and between vowels when the spelling doubles the stop after a stressed short vowel (happy ハッピー).
Checked against curated.tsv + neologd.tsv (1,786 words also in CMUdict): 67.8% exact, 78.2% the same but for ー / ッ, 91.6%
within one kana of the list (`scripts/latin-words/eval-arpabet.mjs`, `eval-accept.mjs`). The typical differences are long vowels
(ー), the list's established forms (camera カメラ, not キャマラ) and names.

### 3.8 Letter names
A spelled run uses one katakana name per letter: A エー, B ビー, C シー, D ディー, E イー, F エフ, G ジー, H エイチ, I アイ, J ジェー,
K ケー, L エル, M エム, N エヌ, O オー, P ピー, Q キュー, R アール, S エス, T ティー, U ユー, V ブイ, W ダブリュー, X エックス, Y ワイ,
Z ゼット (`-` and `'` are dropped). Used for acronyms (§3.3).
