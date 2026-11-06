# How the Latin word list was made (2026-11-03)

Since 1.2.0 the runtime has no English word list: the 19,099 words of curated.tsv / neologd.tsv / cmu-auto.tsv are the training
labels the reading model (G2P) learned English words from. `node scripts/build-latin-words.mjs` builds only the romaji exclusion
list (`src/runtime/latin-words.ts`, `ROMAJI_EXCLUDE`; see docs/SPEC-text-frontend.md §3.5).

Before 1.2.0: `data/latin-words/` → `src/runtime/latin-words.ts` (19,099 words, 3,902 romaji exclusions, 571 KB, 157 KB gzip). The first version (2026-11-03 morning) had 1,741 words; cmu-auto.tsv and the recent words of curated.tsv were added the
same day (Anthropic, Unity, SDK were not read).

## curated.tsv (ours)
About 680 words written for ZEN-TTS: services and brands (Twitter, YouTube, iPhone, ChatGPT, …), IT and business words,
everyday loanwords, a few place names (tokyo → トーキョー, kobe → コーベ). Drafted with LLM help and reviewed by hand. It wins
over neologd.tsv on conflicts. Licensed with the runtime (ZEN-TTS License).

## neologd.tsv (mecab-ipadic-neologd, Apache-2.0)
Inputs (not in the repository):
- `seed/mecab-user-dict-seed.20200910.csv.xz` of https://github.com/neologd/mecab-ipadic-neologd (Apache-2.0)
- `cmudict.dict` of https://github.com/cmusphinx/cmudict (BSD-style)

Steps (selection scripts not included):
1. Entries whose surface is Latin letters (with - or ') and whose reading is katakana: 194,143.
2. Kept when the lower-case word is in CMUdict, the most frequent reading starts with a kana that fits the word's first
   CMUdict phone, it is the reading of at least half the entries, and it is not
   simply the romaji reading: 1,506.
3. Removed by review: homophone mismatches (an English word given another word's katakana, e.g. hitt → ヒット), names and
   rare spellings: 1,262 left.

## romaji-exclude.txt
`romaji-exclude.mjs`: CMUdict words that parse completely as romaji (§3.4 of docs/SPEC-text-frontend.md), are not given the
romaji reading by neologd, and whose CMUdict vowels cannot be written as the romaji's vowels (EY = e / ei, OW = o / ou / oo,
…). Such a word (nose, again, mike) is not read as romaji. 4,813 words, minus those in the word list.

## curated.tsv additions (2026-11-03, afternoon)
About 215 recent and tech words (AI companies and models, dev tools and languages, cloud, platforms and SNS, games and creative
tools, gadgets, everyday web words), drafted with LLM help and reviewed by hand; same licence. Where a word was already there, the
earlier reading was kept (nostr ノスター, browser ブラウザー, security セキュリティー).

## cmu-auto.tsv (`cmu-words.mjs`)
Inputs (not in the repository): `cmudict.dict` (as above) and `3esl.txt` of Alt12Dicts 2020.12.07 (public domain; about 21,000
common English words). For each 3esl word that is in CMUdict and not in curated.tsv / neologd.tsv, the first CMUdict pronunciation
is turned into katakana by `arpabet-katakana.mjs` (rules in docs/SPEC-text-frontend.md §3.7). A few Japanese words of the list
(karate, kimono, geisha, tofu, honcho, …) are left out so they stay romaji. 17,156 words.

How good the rules are: `node scripts/latin-words/eval-arpabet.mjs <cmudict.dict>` and `eval-accept.mjs <cmudict.dict>` compare
them with curated.tsv + neologd.tsv (1,786 words also in CMUdict): 67.8% exact, 78.2% equal but for ー / ッ, 91.6% within one
kana. A wrong reading is fixed by adding the word to curated.tsv (it wins).
