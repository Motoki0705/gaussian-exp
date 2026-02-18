# Dataset Notes: `data/tandt_db`

## 概要

- 種別: 3DGS 公式配布の `Tanks&Temples + Deep Blending`（COLMAP 付き）
- 配布元: `https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip`
- 目的: 3DGS の安定学習・比較実験のベースライン

## 現在の構造

```text
data/tandt_db/
  tandt/
    truck/
      images/
      sparse/0/
        cameras.bin
        images.bin
        points3D.bin
        project.ini
    train/
      images/
      sparse/0/...
  db/
    playroom/
      images/
      sparse/0/...
    drjohnson/
      images/
      sparse/0/...
```

## シーン別の規模

- `tandt/truck`: 251 images
- `tandt/train`: 301 images
- `db/playroom`: 225 images
- `db/drjohnson`: 263 images

## 「COLMAP 付き」の意味

各シーンに以下が同梱されています。

- `cameras.bin`: カメラ内部パラメータ
- `images.bin`: 各画像の姿勢（外部パラメータ）
- `points3D.bin`: 疎点群

このため、`convert.py` を回さず `train.py -s <scene_root>` を直接実行できます。

## 開発向けの知見

- 実写で難易度が高く、3DGS の性能差が出やすい
- 公式配布で再現性が高い
- 画像枚数が多く、学習時間とVRAM消費は大きくなる
- ベンチ比較では `db/playroom` が扱いやすい（今回もこのシーンで 4000 iter 学習を実施）

## 推奨ワークフロー

1. まず `db/playroom` で設定検証（短めイテレーション）
2. 問題なければイテレーションを増やして高品質学習
3. 必要に応じて `tandt/train` など枚数の多いシーンへ展開

## 実行例（今回の高イテレーション設定）

```bash
cd /root/repos/gaussian-exp
.venv/bin/python train.py \
  -s /root/repos/gaussian-exp/data/tandt_db/db/playroom \
  -m /root/repos/gaussian-exp/output/web_playroom_4k \
  --iterations 4000 \
  --save_iterations 1000 2000 3000 4000 \
  --test_iterations -1
```

## 出力確認ポイント

- `output/.../point_cloud/iteration_*/point_cloud.ply` が指定反復で保存される
- 途中の保存タイミング（例: 1000, 2000, 3000, 4000）で一時的に学習速度が落ちるのは正常

