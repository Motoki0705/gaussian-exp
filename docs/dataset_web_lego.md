# Dataset Notes: `data/web_lego`

## 概要

- 種別: NeRF-Synthetic 互換（Blender由来）データ
- 目的: `transforms_train.json` / `transforms_test.json` を使った 3DGS 入力の検証
- 参照先: `data/web_lego`

## 現在の構造

```text
data/web_lego/
  transforms_train.json
  transforms_test.json
  train/
    r_0.png ... r_99.png
  test/
    r_0.png ... r_199.png
```

## 現在の実データ品質（要注意）

現時点の実体を確認した結果:

- `train`: 100 枚中 100 枚が有効 PNG
- `test`: 200 枚中 36 枚が有効 PNG、164 枚が ASCII テキスト

ASCII テキストの中身は `Entry not found` で、画像取得元の欠損応答が保存されたものです。
この状態では、`transforms_test.json` を読む処理で `PIL.UnidentifiedImageError` が発生します。

## 学習時の挙動

- 3DGS は `transforms_train.json` だけでなく `transforms_test.json` 側も読み込みます。
- したがって、`--test_iterations -1` を指定しても、`test` 画像が壊れていると学習前に停止します。

## 開発向けの知見

- このディレクトリは「Blender形式ローダの挙動確認」には有用ですが、現状はベンチ用途に不適です。
- 壊れ画像が混ざるケースを再現できるため、ダウンロード検証や前処理バリデーション実装のテスト材料としては有用です。
- 本番学習には、`data/tandt_db` のような COLMAP 付きデータを優先する方が安定です。

## 復旧・運用ガイド

1. `test/*.png` の実体検証（PNG かどうか）を先に行う
2. 壊れファイルを削除して再ダウンロードする
3. `transforms_test.json` と一致する全ファイルが揃うまで学習を実行しない

検証の例:

```bash
cd /root/repos/gaussian-exp
find data/web_lego/test -type f -name '*.png' -print0 | \
  xargs -0 file | awk '/ASCII text/{a++} /PNG image data/{p++} END{print "png="p, "ascii="a}'
```

