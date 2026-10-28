# ZEN-TTS ライセンス 1.0 ／ ZEN-TTS License 1.0

Copyright (c) 2026 Zenryoku Kikai Inc. (全力機械株式会社) — https://zenryokukikai.com

## 日本語

### 1. 対象

このライセンスは、このリポジトリに含まれる次のもの（以下「本ソフトウェア」）に適用されます。
- ソースコード
- モデルの重み（`models/`）
- 文書

THIRD_PARTY_NOTICES.md に挙げた第三者の部分は、それぞれのライセンスに従います。

### 2. 許可すること

次の 3 章の条件を守る限り、誰でも、無料で、次のことができます。
- 本ソフトウェアを、自分のアプリ・Web サイト・サービス・作品の中で使う。
- 本ソフトウェアを、（改変せずに）自分のソフトウェアに組み込む。
- 本ソフトウェアで作った音声を使う（配信、販売、作品への利用を含む）。

これらは、商用・非商用を問いません。

### 3. 条件

1. **クレジットの表示（必須）**
   本ソフトウェア、またはそれで作った音声を使うアプリ・Web サイト・サービス・作品では、利用者が普通に見つけられる場所に、次のクレジットを表示してください。クレジット中の URL はリンクにしてください（リンクを埋め込めない媒体では URL を文字で書けば足ります）。
   - クレジット：「音声合成：ZEN-TTS（https://zenryokukikai.com ）」、または英語の「Speech synthesis: ZEN-TTS (https://zenryokukikai.com)」
   - ZEN-LipSync と併用する場合は、1 行にまとめて構いません（例：「音声合成：ZEN-TTS / リップシンク：ZEN-LipSync（https://zenryokukikai.com ）」）。
   - 表示する場所の例：画面、クレジット、「このアプリについて」のページ、説明欄、概要欄
   - 動画や音声だけの作品：説明欄や概要欄など、作品と一緒に示される場所に表示してください。

2. **改変の禁止**
   本ソフトウェア（ソースコード、モデルの重みを含む）を改変してはいけません。自分のアプリのビルドの中で、改変せずに依存として取り込み、バンドル・圧縮（minify）することは改変にあたりません。

3. **再配布の禁止**
   本ソフトウェア、その重みを、第三者に再配布、再許諾、販売、譲渡、貸与してはいけません。自分のアプリ・サービスの中で使うことはできますが、モデルや重みを単体で、または再配布できる形で第三者に渡すことはできません。

4. **学習・蒸留・評価への利用の禁止**
   本ソフトウェア、その重み、および本ソフトウェアで作った音声を、他の機械学習モデルの学習、蒸留、評価用のデータとして使ってはいけません。


### 4. 保証しないこと

本ソフトウェアは「現状のまま」提供します。全力機械株式会社は、次のことを保証しません。
- 品質が正しいこと
- 特定の目的に使えること
- 第三者の権利を侵害しないこと

本ソフトウェアの使用によって生じた損害について、法令で許される範囲で、全力機械株式会社は責任を負いません。

### 5. ライセンスの終了

3 章の条件に違反した場合、このライセンスで与えた許可は自動的に終わります。
ただし、違反に気づいてから 30 日以内に違反をやめた場合は、許可は元に戻ります。

### 6. 準拠法

このライセンスは、日本法に従って解釈します。

## English (translation; the Japanese text prevails)

### 1. Scope
This license covers the source code, the model weights (`models/`) and the documents in this repository (the "Software"). Third-party parts listed in THIRD_PARTY_NOTICES.md are under their own licenses.

### 2. Grant
Anyone may, free of charge and for commercial or non-commercial purposes:
- use the Software inside their own app, website, service or work;
- embed the Software, unmodified, in their own software;
- use audio produced with the Software (including publishing, selling it and using it in works),

subject to the conditions in Section 3.

### 3. Conditions
1. **Credit (required).** Any app, website, service or work that uses the Software or audio produced with it must show, in a place where its users can reasonably find it, the credit "Speech synthesis: ZEN-TTS (https://zenryokukikai.com)" (or the Japanese equivalent), with the URL as a hyperlink. Where links cannot be embedded, the URL written as text is enough. When used together with ZEN-LipSync, one combined line is fine (e.g. "Speech synthesis: ZEN-TTS / Lip-sync: ZEN-LipSync (https://zenryokukikai.com)").

   Suitable places include the UI, the credits, an about page or the description. For video or audio-only works, use the accompanying description.
2. **No modification.** Do not modify the Software (including its source code and model weights). Including it unmodified as a dependency in your own build, bundled or minified, is not a modification.
3. **No redistribution.** Do not redistribute, sublicense, sell, assign or lend the Software or its weights to any third party. You may use it inside your own app or service, but you may not hand the model or its weights to a third party, alone or in a form that can be redistributed.
4. **No use for training, distillation or evaluation.** Do not use the Software, its weights, or audio produced with the Software as data to train, distil or evaluate other machine-learning models.

### 4. No warranty
The Software is provided "as is". There is no warranty of any kind, including fitness for a particular purpose or non-infringement. To the extent permitted by law, Zenryoku Kikai Inc. is not liable for any damages arising from its use.

### 5. Termination
The grant ends automatically if Section 3 is breached. It is reinstated if the breach is cured within 30 days of becoming aware of it.

### 6. Governing law
This license is governed by the laws of Japan.
