# ZEN-TTS

日本語のニューラル音声合成（TTS）。5 つの感情スタイルを連続値で混合し、文中で時間的に遷移させられます。G2P からデコーダまでブラウザ内（WebGPU / WASM SIMD）と Node.js（CPU）で動作します。

バージョン 1.0.0。4 voices（voice ごとに独立したモデル。G2P は共通）。

| voice | id | model | 特徴 |
|---|---|---|---|
| 全力 マリン（ぜんりょく まりん） | `marine` | marine-1.0 | やわらかく落ち着いた女性の声 |
| 全力 ハシロウ（ぜんりょく はしろう） | `hashiro` | hashiro-1.0 | 落ち着いた、深みのある男性の声 |
| 全力 カイ（ぜんりょく かい） | `kai` | kai-1.0 | 40代の男性。響きのある低音 |
| 全力 キッカ（ぜんりょく きっか） | `kikka` | kikka-1.0 | 素朴で透明感のある若い女性の声 |

ボイスクローン機能はありません。上記以外の voice のモデル作成は全力機械（https://zenryokukikai.com ）へご依頼ください。

利用条件として、アプリ・サイト・作品の利用者が見つけられる場所に、クレジット「音声合成：ZEN-TTS（https://zenryokukikai.com ）」を表示してください。URL はリンクにします（[LICENSE.md](LICENSE.md) 3 章）。

## 特徴

- **スタイル混合と遷移**：`neutral` / `happy` / `sad` / `angry` / `apologetic` を連続値の比率で混合（例：angry 30・apologetic 70）。1 文の中で開始配合から終了配合へ、9 種のカーブ（`EMOTION_CURVES`）と開始・終了位置で遷移。音単位の F0・長さの修正と辞書の相対ピッチ形状は、任意の配合・カーブ上で形状を保って適用されます。
- **オンデバイス**：G2P、韻律予測、生成器、NSF デコーダ（44.1 kHz）まで、すべてクライアント側で実行。ブラウザは WebGPU、非対応時は WASM SIMD（Web Worker）。
- **速度**：Intel Xeon 1 スレッドで RTF 0.042（「ほかの TTS との比較」）。ブラウザの CPU で RTF 約 0.09（Apple M3 Max）、iPhone で約 0.11。
- **辞書**：語ごとに読み・アクセント型・拍ごとの相対ピッチ（`contour`）・長さ・句境界を登録。相対値なので、感情や文中位置によらず登録した形状で読みます。
- **サイズ**：1 voice あたり約 31 MB（int8 / fp16）、CPU カーネル（`cpu/tts-cpu.wasm`）57 KB。

## 関連リポジトリ

| リポジトリ | 中身 |
|---|---|
| [zenryokukikai/zen-tts](https://github.com/zenryokukikai/zen-tts)（このリポジトリ） | ZEN-TTS：4 voices の重み、ブラウザ / Node ランタイム、デモ |
| [zenryokukikai/zen-lipsync](https://github.com/zenryokukikai/zen-lipsync) | ZEN-LipSync の SDK（`@zenryokukikai/zen-lipsync`） |
| [zenryokukikai/zen-lipsync-avatars](https://github.com/zenryokukikai/zen-lipsync-avatars) | アバターパックと examples |

## インストール

npm レジストリには公開していません。GitHub から指定します。

```sh
# npm
npm i github:zenryokukikai/zen-tts#v1.0.0
# pnpm
pnpm add github:zenryokukikai/zen-tts#v1.0.0
# yarn
yarn add github:zenryokukikai/zen-tts#v1.0.0
```

- Node.js 26.12.1（`.mise.toml` で固定、`mise install`）。開発は pnpm。
- `#v1.0.0` はタグまたはコミットハッシュで固定してください。
- 重み（`models/`）は npm パッケージに含まれません。リポジトリの `models/` を配置し、`modelsUrl`（ブラウザ）／`modelsPath`（Node）で指定します。ファイル一覧と sha256 は `models/manifest.json`。

## デモ

```bash
pnpm install         # devDependencies のみ（TypeScript、Playwright）
pnpm build           # tsc -> dist/
pnpm serve           # http://localhost:8080/demo/
```

`demo/index.html`：
- 合成（WAV 保存、処理時間・RTF 表示）、感情配合スライダー、`strength`
- 文中遷移（9 種のカーブ）
- バックエンド選択：auto / WebGPU / CPU（WASM）
- イントネーション編集：音ごとの読み・アクセント・F0 曲線・長さを表示し、F0・長さ・アクセント核を編集して再合成。配合を変えても編集は保持
- 辞書編集：登録・修正（読み、相対ピッチ形状、長さ、アクセント核）。localStorage に保存し、JSON で書き出し／読み込み

対応ブラウザ：
- WebGPU：Chrome / Edge 113+、Safari 26+、Android Chrome
- CPU：WASM SIMD + Web Worker 対応ブラウザ

E2E：`pnpm check`（`pnpm serve` 起動中に実行）。ヘッドレス Chromium で WebGPU / CPU の合成、イントネーション編集、辞書の登録・書き出し・読み込みを検証します。

## 使い方（API）

```ts
import { ZenTts, wavBlob } from './dist/index.js';

// backend: 'auto'（既定：WebGPU があれば GPU、なければ CPU）、'webgpu'、'cpu'
// voice: 'marine'（既定）| 'hashiro' | 'kai' | 'kikka'
const tts = await ZenTts.load({ modelsUrl: './models', cpuWasmUrl: './cpu/tts-cpu.wasm', voice: 'marine' });
console.log(tts.backend, tts.voice);   // 'webgpu' か 'cpu'、声の id

// 感情は比率で指定（合計は気にしなくてよい）
const r = await tts.speak('本日はご来店いただき、ありがとうございます。', {
  emotion: { happy: 70, neutral: 30 },
  strength: 1.2,                 // 抑揚の強さ（1 = 標準）
});
// r.audio: Float32Array（44.1 kHz、モノラル）、r.timing: 段ごとの時間
const url = URL.createObjectURL(wavBlob(r.audio));

// スタイルのグラデーション：はじめの配合（emotion）から、おわりの配合（change.to）へ、
// 文の 20%〜80% の間を「なめらか」（ease）なカーブで移る
await tts.speak('落ち着いて聞いてください。もう我慢できません。', {
  emotion: { neutral: 80, apologetic: 20 },
  change: { to: { angry: 70, sad: 30 }, curve: 'ease', from: 0.2, until: 0.8 },
});
```

- `speakStream(text, options)`：`AsyncIterable` で 1 文ずつ返す。分割は「。！？」と改行（`splitSentences()`）。
- `speakWith(text, { blend, labels, f0Add, durScale })`：音単位の F0 加算・長さ倍率・アクセントラベルを与えて合成（デモのイントネーション編集が使用）。
- 感情名：`neutral`、`happy`、`sad`、`angry`、`apologetic`。
- export：`VERSION`（`'1.0.0'`）、`VOICE`（既定 `'marine'`）、`VOICES`。`tts.voice` は使用中の voice id。
- `EMOTION_CURVES`：`ease`、`linear`、`fast-start`、`late-surge`、`switch`、`steps`、`hill`、`valley`、`wave`。

## Node.js で使う（サーバー、CPU）

`zen-tts/node`：Node.js 26.12.1+ の CPU で、ブラウザの CPU パスと同じ処理・同じ重みで合成します（テキスト前処理、G2P、韻律、感情配合・遷移、辞書）。

```sh
pnpm add github:zenryokukikai/zen-tts
pnpm run build:native                            # 任意：x86-64 の Linux で速いネイティブ版を作る（clang 16 以降）
```

```js
import { ZenTtsNode, wavBytes } from 'zen-tts/node';
import { writeFileSync } from 'node:fs';

const tts = await ZenTtsNode.load({ modelsPath: './models', voice: 'kikka' });   // voice を省くと 'marine'
console.log(tts.kernel, tts.voice);   // 'avx512' | 'avx2' | 'sse41'（ネイティブ版）か 'wasm'、声の id

const r = await tts.speak('本日はご来店いただき、ありがとうございます。', { emotion: { happy: 70, neutral: 30 } });
writeFileSync('voice.wav', wavBytes(r.pcm));   // r.pcm: Float32Array、44.1 kHz、モノラル

// 長い文章は 1 文ずつ：できた文から順に受け取れる（再生・配信・リップシンクに渡しながら、次の文を作る）
for await (const c of tts.speakStream('こんにちは。全力マリンです。今日はいい天気ですね。')) {
  console.log(c.index, c.text, c.pcm.length / c.sampleRate, '秒');
}
```

- カーネルはロード時に自動選択：AVX-512 → AVX2 → SSE4.1（`pnpm run build:native` のネイティブ版）→ WASM。`backend: 'wasm'`、`kernel: 'avx2'`（または `ZEN_TTS_KERNEL`）で固定可。どのカーネルも同一の出力（加算順序同一、FMA なし）。
- RTF（Intel Xeon Silver 4316、1 スレッド、15 文の中央値）：

| kernel | RTF |
|---|---|
| AVX-512 | 0.040 |
| AVX2 | 0.066 |
| SSE4.1 | 0.097 |
| WASM | 0.101 |

- `speakStream` の最初のチャンクまで約 0.1 s。
- ZEN-LipSync との連携：`@zenryokukikai/zen-lipsync/tts/node`。

## 辞書の形式

`speak(text, { dict })` に、次の形の配列を渡します（デモの「書き出す」も同じ形です）。

```json
[
  { "surface": "明日", "reading": "アス", "accent": 2 },
  { "surface": "全力機械", "reading": "ゼンリョクキカイ", "accent": 5, "length": [1, 1, 1, 1, 1, 1.2] }
]
```

| 項目 | 必須 | 意味 |
|---|---|---|
| `surface` | ○ | 文章の中の書き方（長いものから先に当てはめます） |
| `reading` | ○ | 読み（カタカナかひらがな） |
| `accent` |  | アクセント型：何拍目の後で下がるか。0 = 平板、1 = 頭高。省くと、読みだけを変えて、アクセントはモデルに任せます |
| `phrase` |  | `"split"`：この語の前で句を切る。`"join"`：前の句とつなげる |
| `strength` |  | この語の句の中での、モデルの抑揚の強さ（0 = 型どおり） |
| `contour` |  | 拍ごとの相対的な高さ（-3〜3）。語の中の平均からの上がり下がりで、1 は、その感情の声の幅（約 6〜8 半音）に相当します。読むときは、文の中でモデルがその語に予測した高さにこの形を重ねます。どの感情で読んでも、文のどこに出てきても同じ形になります |
| `pitch` |  | 拍ごとに足す高さ（半音） |
| `length` |  | 拍ごとの長さの倍率 |

全体の設定は `settings: { accentHeight: 4.5 }` です。アクセント型のある語の、高い拍の高さを半音で指定します。

デモの「ファイルに書き出す」は、`{ "format": "zen-tts-dictionary", "version": 1, "entries": [ ... ] }` の形で書き出します。`entries` の中身は上の表と同じです。

### 組み込みの辞書

`src/runtime/builtin-dict.ts`。`speak` / `analyze` はユーザー辞書の前に適用します。同じ `surface` のユーザー項目が優先されます。

## モデルファイル

重みは `models/` に同梱（4 voices、計 124.5 MB。npm パッケージには含まれない）。

| voice | パス |
|---|---|
| `marine`（既定） | `models/tts/`、`models/decoder/` |
| `hashiro` / `kai` / `kikka` | `models/voices/<id>/tts/`、`models/voices/<id>/decoder/` |

| ファイル | サイズ | 内容 |
|---|---|---|
| `tts/own.bin`、`tts/own.json` | 10.6 MB（int8） | G2P（全 voice 共通）、韻律予測器、生成器。`own.json` は層構成・`pitchRange` など |
| `decoder/nsf.bin`、`decoder/nsf.json` | 19.1 MB（fp16） | NSF デコーダ（44.1 kHz）。`nsf.json`：`logF0Mean` / `logF0Std`（F0 正規化）、`outputGain`（出力の線形ゲイン。省略時 1。> 1 のときピークリミッタ knee 0.8 / ceiling 0.95） |
| `models/manifest.json` | 1 KB | voice ごとの model 版・名前・ディレクトリ、全ファイルの bytes と sha256 |
| `cpu/tts-cpu.wasm` | 57 KB | CPU カーネル（WASM SIMD128）。ソースは `native/tts-cpu.c`、`pnpm build:wasm` で再生成（clang + wasm-ld、`WASM_LLVM_BIN`） |

### ネイティブカーネル（x86-64、Node.js）

`native/tts-cpu.c` を x86-64 向けにビルドした Node-API アドオン（AVX-512 / AVX2 / SSE4.1）。`pnpm run build:native` で `native/x86/build/` に出力すると `zen-tts/node` がロード時に使用します（clang 16+、`-ffp-contract=off`）。ブラウザでは使用しません。

## ほかの TTS との比較（大きさと速さ）

同じサーバーの CPU（1 スレッド）で、日本語の TTS を同じ文で測りました。RTF は「作るのにかかった時間 ÷ 音声の長さ」で、小さいほど速くなります。ニューラルネットワークで声を作る TTS に限って比べています。

| TTS | モデルサイズ | メモリのピーク | RTF 短い文 | RTF 中くらい | RTF 長い文 | 出力 | ブラウザで動く | スタイル・感情 | 音ごとの高さの指定 | 商用利用 |
|---|---|---|---|---|---|---|---|---|---|---|
| **ZEN-TTS** | **31.1 MB** | 255 MB | **0.045** | **0.042** | **0.041** | 44.1 kHz | ○ | 5 つを連続した割合で混ぜる。文の途中でカーブに沿って変える | ○ | ○（クレジット表示が必要） |
| Supertonic 3（fp32、公式） | 約 400 MB | 559 MB | 0.37 | 0.30 | 0.28 | 44.1 kHz | ○（onnxruntime-web） | 用意された声と表現タグ（混ぜられない） | × | ○（OpenRAIL-M。使い方の制限あり） |
| VOICEVOX CORE 0.17.1（ずんだもん） | 約 185 MB（モデル 59 MB ＋ 辞書 107 MB ＋ 実行部 18 MB） | 713 MB | 0.67 | 0.64 | 0.61 | 24 kHz | × | 話者ごとのスタイルを 1 つ選ぶ（混ぜられない） | ○ | ○（「VOICEVOX:キャラ名」のクレジットが必要） |
| Kokoro-82M（jf_alpha） | 約 1.2 GB（モデル 354 MB ＋ 読みの辞書 812 MB） | 1.07 GB | 0.78 | 0.70 | 0.66 | 24 kHz | △（ブラウザ版は日本語の読みに未対応） | 用意された声から選ぶ | × | ○（Apache-2.0） |
| AivisSpeech Engine 1.2.0（コハク） | 約 1.25 GB | 約 2 GB（エンジン） | — | 3.24 ※2 | — | 44.1 kHz | × | スタイルとその強さ（モデルによる） | ×（指定しても無視される）※3 | ○（モデルは ACML 1.0） |
| 参考（対象外）：Piper（ja_JP-hi_fi_captain-medium） | 約 184 MB（声 77 MB ＋ 辞書 107 MB）＋ 実行部 | 719 MB | 0.066 | 0.067 | 0.070 | 22.05 kHz | × | なし | × | ×（声のデータが CC BY-NC-SA 4.0） |

比べたもの：Supertonic 3、VOICEVOX、Kokoro-82M、AivisSpeech。Piper の日本語の声も測りましたが、声のデータが商用利用不可（CC BY-NC-SA 4.0）のため対象外です（ZEN-TTS のほうが速い結果）。

※2 AivisSpeech は HTTP の時間を含み、中くらいの文だけを 1 回（交互の 3 周ではない）測りました。
※3 AivisSpeech Engine の README に「`consonant_length` / `vowel_length` / `pitch` フィールドはサポートされていません」「常に無視されます」とあります。実際に 1 音の pitch を書き換えて読ませても、変わりませんでした。

**測った条件**
- 機械：Intel Xeon Silver 4316（2.3 GHz、仮想マシン）。CPU だけを使い、1 スレッド・1 つのコアに固定しました。
- 音ごとの高さの指定：真ん中の 1 音と、アクセントに逆らう 1 音を、+3 半音にして読ませ、できた音の高さを測りました。ZEN-TTS は +2.8〜+3.5 半音、VOICEVOX は +2.0〜+3.0 半音で、指定どおりに上がりました。AivisSpeech は変わりませんでした。

**出典**
- VOICEVOX：https://voicevox.hiroshiba.jp/ 、https://github.com/VOICEVOX/voicevox_core 、モデル（VVM）と利用規約 https://github.com/VOICEVOX/voicevox_vvm
- Piper：https://github.com/OHF-Voice/piper1-gpl 、日本語の声 https://huggingface.co/rhasspy/piper-voices/tree/main/ja/ja_JP/hi_fi_captain/medium
- Supertonic 3：https://huggingface.co/supertone-oss-archive/supertonic-3
- Kokoro-82M：https://huggingface.co/hexgrad/Kokoro-82M
- AivisSpeech：https://aivis-project.com/ 、AivisSpeech Engine の README（`Mora` 型の仕様変更、pitch が無視されること）https://github.com/Aivis-Project/AivisSpeech-Engine

## 聞き比べ

同じ文を、それぞれの TTS の標準の声で読ませた音声です（動画は、TTS の名前と文を表示した静止画に音声を付けたものです）。

- 加工は音量をそろえただけです（EBU R128 で -20 LUFS。どの音声にも同じ処理）。

**短い文**「夕方から風が強くなりそうです。」

<table>
<tr><th>TTS</th><th>音声</th></tr>
<tr><td>ZEN-TTS</td><td><video controls src="https://github.com/user-attachments/assets/7fc6c207-9c50-4df4-8060-a7997a7e340e"></video></td></tr>
<tr><td>VOICEVOX（ずんだもん ノーマル）<br><sub>VOICEVOX:ずんだもん</sub></td><td><video controls src="https://github.com/user-attachments/assets/a42acdbe-bd2d-4ecf-91df-c8661e56b9cd"></video></td></tr>
<tr><td>AivisSpeech（コハク ノーマル）<br><sub>AivisSpeech: コハク（ACML 1.0）</sub></td><td><video controls src="https://github.com/user-attachments/assets/d85f4af1-4bfc-461b-80ed-7da2f2387278"></video></td></tr>
<tr><td>Supertonic 3（F1）<br><sub>Supertonic 3 © Supertone Inc.（OpenRAIL-M）</sub></td><td><video controls src="https://github.com/user-attachments/assets/e383b6da-80ed-41ab-8314-7dafe2459b4d"></video></td></tr>
<tr><td>Kokoro-82M（jf_alpha）<br><sub>Kokoro-82M（Apache-2.0）</sub></td><td><video controls src="https://github.com/user-attachments/assets/54192558-7adf-411c-bb67-5c4d966219d2"></video></td></tr>
<tr><td>MeloTTS（JP）<br><sub>MeloTTS © MyShell.ai（MIT）</sub></td><td><video controls src="https://github.com/user-attachments/assets/5474d02d-9301-4690-a930-9a0044692ef4"></video></td></tr>
</table>

**中くらいの文**「全国的に晴れて、昼間は過ごしやすい気温になる見込みです。」

<table>
<tr><th>TTS</th><th>音声</th></tr>
<tr><td>ZEN-TTS</td><td><video controls src="https://github.com/user-attachments/assets/2f92765d-8ac5-4123-ae30-9ca9b0bee156"></video></td></tr>
<tr><td>VOICEVOX（ずんだもん ノーマル）<br><sub>VOICEVOX:ずんだもん</sub></td><td><video controls src="https://github.com/user-attachments/assets/bbecce6e-cccc-49a3-9aa2-795a6006dd34"></video></td></tr>
<tr><td>AivisSpeech（コハク ノーマル）<br><sub>AivisSpeech: コハク（ACML 1.0）</sub></td><td><video controls src="https://github.com/user-attachments/assets/7bf69933-113e-4027-8ca6-af1e075617ee"></video></td></tr>
<tr><td>Supertonic 3（F1）<br><sub>Supertonic 3 © Supertone Inc.（OpenRAIL-M）</sub></td><td><video controls src="https://github.com/user-attachments/assets/6420b6c5-a4e9-4250-accf-a4209f23e758"></video></td></tr>
<tr><td>Kokoro-82M（jf_alpha）<br><sub>Kokoro-82M（Apache-2.0）</sub></td><td><video controls src="https://github.com/user-attachments/assets/76ec00e1-0864-4e63-b0da-6242a0a19ed1"></video></td></tr>
<tr><td>MeloTTS（JP）<br><sub>MeloTTS © MyShell.ai（MIT）</sub></td><td><video controls src="https://github.com/user-attachments/assets/2b35021d-c591-40f9-b36d-d1940a5a6e72"></video></td></tr>
</table>

**感情を込めたい文**「えっ、本当ですか？ずっと待っていたので、とてもうれしいです。」

<table>
<tr><th>TTS</th><th>音声</th></tr>
<tr><td>ZEN-TTS</td><td><video controls src="https://github.com/user-attachments/assets/52ee528f-826e-4885-9531-d901ce8eebb2"></video></td></tr>
<tr><td>VOICEVOX（ずんだもん ノーマル）<br><sub>VOICEVOX:ずんだもん</sub></td><td><video controls src="https://github.com/user-attachments/assets/97011de3-57ce-4fbc-b799-6018c1ef43a0"></video></td></tr>
<tr><td>AivisSpeech（コハク ノーマル）<br><sub>AivisSpeech: コハク（ACML 1.0）</sub></td><td><video controls src="https://github.com/user-attachments/assets/6714b4b5-54b0-4ff5-9eb5-68acd3e9440b"></video></td></tr>
<tr><td>Supertonic 3（F1）<br><sub>Supertonic 3 © Supertone Inc.（OpenRAIL-M）</sub></td><td><video controls src="https://github.com/user-attachments/assets/534c4295-079a-41d5-a50b-4236d4e28707"></video></td></tr>
<tr><td>Kokoro-82M（jf_alpha）<br><sub>Kokoro-82M（Apache-2.0）</sub></td><td><video controls src="https://github.com/user-attachments/assets/41bfa31e-8553-47aa-90f2-4b554afc061d"></video></td></tr>
<tr><td>MeloTTS（JP）<br><sub>MeloTTS © MyShell.ai（MIT）</sub></td><td><video controls src="https://github.com/user-attachments/assets/ebf54c8f-09d3-47e8-9d7c-480a413d1469"></video></td></tr>
</table>

Piper の日本語の声（ja_JP-hi_fi_captain-medium）は、声のデータが CC BY-NC-SA 4.0（商用不可）なので、音声を載せていません。

**ZEN-TTS の機能サンプル**

文中遷移と比較用の文：「ずっと信じて待っていたのに、連絡ひとつないまま、約束の日が過ぎてしまいました。悲しくて、しばらく何も手につきませんでした。でも、もうこれ以上は我慢できません。」

<table>
<tr><th>内容</th><th>音声</th></tr>
<tr><td>感情混合<br><sub>happy 70 ＋ neutral 30</sub></td><td><video controls src="https://github.com/user-attachments/assets/8173c512-f2a7-4a81-a461-7b16b51f3415"></video></td></tr>
<tr><td>文中遷移<br><sub>sad → angry、<code>switch</code></sub></td><td><video controls src="https://github.com/user-attachments/assets/b7f0060c-8a86-42e0-ac56-7605ed185e72"></video></td></tr>
<tr><td>比較用<br><sub>同じ文を sad のみで</sub></td><td><video controls src="https://github.com/user-attachments/assets/c7f4b72c-befc-4493-9007-34602f7e8c59"></video></td></tr>
<tr><td>音単位 F0 編集：編集前</td><td><video controls src="https://github.com/user-attachments/assets/6aada753-16a9-4416-810f-4fc677f6c935"></video></td></tr>
<tr><td>音単位 F0 編集：編集後<br><sub>「すごしやすい」の「や」を +3 半音</sub></td><td><video controls src="https://github.com/user-attachments/assets/de1d2a31-ca7e-48c5-9b3f-cf940bafa24b"></video></td></tr>
</table>

## ライセンス

**ZEN-TTS ライセンス**（LICENSE.md）。

- 本ソフトウェアをそのまま自分のアプリ・サイト・サービス・作品に組み込んで使えます。作った音声も使えます（商用・非商用を問わず）。改変はできません（ソースコード・モデルの重みを含む。自分のアプリのビルドで依存として取り込み、バンドル・圧縮することは改変にあたりません）。
- **条件**：本ソフトウェアやそれで作った音声を使うアプリ・サイト・作品では、利用者が見つけられる場所に、次のクレジットを表示してください。URL はリンクにします（リンクを埋め込めない媒体では URL を文字で書けば足ります）。
  - クレジット：「音声合成：ZEN-TTS（https://zenryokukikai.com ）」
- **再配布はできません。**モデルや重みを、単体で、または再配布できる形で第三者に渡すことはできません。
- 本ソフトウェア、その重み、およびそれで作った音声を、他の機械学習モデルの学習、蒸留、評価用のデータとして使わないこと。
- 第三者コンポーネント：THIRD_PARTY_NOTICES.md。
