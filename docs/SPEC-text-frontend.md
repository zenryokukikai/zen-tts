# Text front end: behavioural specification

This document specifies two steps the runtime runs before the G2P model:

1. **Text normalisation** (`normalizeOwnText`): it turns arbitrary Japanese text into the restricted character set the G2P model reads.
2. **Kana reading to phonemes** (`kanaToPhones`): it turns a dictionary reading written in kana into the phoneme sequence the predictor uses.

They are implemented in `src/runtime/ja-text-normalize.ts` and `src/runtime/kana-phones.ts`.

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

### 1.5 Numbers
Find every maximal numeral, scanning left to right. A numeral is one or more digits, optionally followed by `.` and one or more digits. Each one is replaced by its Japanese reading (§1.7).

So:
- `1.2.3` → `一点二.三`
- `.5` → `.五`
- `1.` → `一.`
- `1..2` → `一..二`

### 1.6 Character classes
After the numbers, map the following characters, then delete every character that is not allowed.

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
