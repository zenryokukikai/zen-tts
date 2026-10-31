# .ztm（ZEN-TTS の声を 1 ファイルにしたもの）

1 つの voice を 1 ファイルにまとめた形式です。拡張子は `.ztm`（zen-tts-model）。1.1.0 から、voice はこの形式で CDN から配布します（README「モデルファイル」）。

## 構成

| 位置 | 中身 |
|---|---|
| 0〜3 バイト | `ZTM1` |
| 4〜7 バイト | ヘッダー長（uint32、リトルエンディアン） |
| 8 バイト〜 | ヘッダー（UTF-8 の JSON）。最初のセクションが 64 バイト境界から始まるよう空白で埋める |
| その後 | セクション `acoustic`、`vocoder`。それぞれ 64 バイト境界から配置し、間は 0 で埋める |

- **acoustic**：韻律予測器・生成器とスタイル（感情）。1.0.0 の `tts/own.bin` から G2P の部分を除いたもの。
- **vocoder**：NSF デコーダ。1.0.0 の `decoder/nsf.bin` と同じバイト列。
- 重みのバイト列は 1.0.0 の 4 ファイルから変えていません（同じ文から同じサンプルが出ます）。ファイル内では圧縮しません。

### ヘッダー（JSON）

- `format: "zen-tts-model"`、`version: 1`
- `voice`：`{ id, name?: { ja?, en? } }`
- `styles`：スタイル名（スタイル行の順）。このランタイムは `neutral`, `happy`, `sad`, `angry`, `apologetic` のみ読み込みます。
- `created`：作成日時。`source`：作成ツールと、元の 4 ファイルの SHA-256。
- `g2p`：`{ sha256, bytes }`。その voice を作成・確認したときの G2P のバイト列のハッシュ。
- `acoustic`：`{ meta, tensors }`。meta は own.json から G2P の表と `g2p.*` を除いたもの。テンソルの offset は acoustic セクションの先頭から。
- `vocoder`：`{ meta, tensors }`。meta は nsf.json の `tensors` 以外（`dtype`・`logF0Mean`・`logF0Std`・`outputGain` など）。テンソルの offset は vocoder セクションの先頭から。
- `sections`：`acoustic`・`vocoder` の `{ offset, bytes, sha256 }`。offset はファイル先頭から。

## G2P はランタイム側

- G2P（読みのモデル）は全 voice 共通なので .ztm には含めず、ランタイムの `models/g2p/g2p.json`・`g2p.bin`（4.99 MB、リポジトリに同梱）を使います。
  - g2p.json：テンソル一覧、`shape.g2p`、`vocab`・`charPad`・`charUnk`・`phoneClasses`・`marks`・`punctuation`・`kana`、g2p.bin の SHA-256。
- own.bin の先頭から `g2p.*` テンソルの終わり（4,987,392 バイト）までが G2P、その後ろが acoustic。
- voice の `g2p.sha256` とランタイムの G2P が違うときは、警告を出して読み込みます（読みが確認時と変わることがあるため）。1.1.0 の 4 voices はすべて同じ G2P（`1a8877…`）です。

## 読み込み

```js
import { ZenTts, ztmUrl } from 'zen-tts';
// ブラウザ：.ztm を 1 回の fetch で取得。G2P は <modelsUrl>/g2p/ から
const tts = await ZenTts.load({ modelsUrl: './models', cpuWasmUrl: './cpu/tts-cpu.wasm', model: ztmUrl('kikka') });

import { ZenTtsNode } from 'zen-tts/node';
// Node：model はローカルのパスか http(s) の URL。G2P はパッケージの models/g2p（modelsPath を指定したときはその g2p/）
const t = await ZenTtsNode.load({ model: '/path/kikka.ztm' });
```

- `tts.voice` は voice id、`tts.header` はヘッダー。
- デモは `demo/?model=<.ztm の URL>` でその voice を使います。

## ツール

```sh
node scripts/pack-ztm.mjs pack <フォルダ> -o kikka.ztm [--id kikka] [--name-ja "全力 キッカ"] [--name-en "Zenryoku Kikka"]
node scripts/pack-ztm.mjs unpack kikka.ztm -o <フォルダ> --g2p models/g2p   # 1.0.0 の 4 ファイルに戻す
node scripts/pack-ztm.mjs g2p <フォルダ> -o models/g2p                     # own.* から G2P を取り出す
node scripts/pack-ztm.mjs info kikka.ztm
```

- フォルダは 1.0.0 の構成（`tts/own.{json,bin}` と `decoder/nsf.{json,bin}`）。
- `pack` → `unpack` → `pack` で元とバイト単位で同じファイルになります。

## サイズ

| | サイズ |
|---|---|
| 1.0.0 の 4 ファイル（1 voice） | 31.1 MB |
| .ztm（1 voice） | 26.1 MB（G2P を含まない） |
| G2P（ランタイム、全 voice 共通） | 4.99 MB ＋ g2p.json 45 KB |

int8 / fp16 の重みはほとんど圧縮できません（gzip で約 92%）。
