# 3DGS ローカル実行セットアップ済み

このワークスペースでは、公式 3D Gaussian Splatting をクローンして、依存ビルドまで完了しています。

- クローン先: `gaussian-splatting/`
- サブモジュール: 初期化済み
- Python 仮想環境: `.venv/`
- CUDA 拡張 (`simple-knn`, `fused-ssim`, `diff-gaussian-rasterization`): インストール済み

## 使い方

### 1) 仮想環境の構築・有効化

このリポジトリでは、ワークスペース直下の `.venv` を使用します。

#### 仮想環境を作る（初回のみ）

```bash
cd /root/repos/3dgs-exp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

#### 依存を入れる（今回の実行構成）

```bash
cd /root/repos/3dgs-exp/gaussian-splatting

# PyTorch (CUDA 12.8)
/root/repos/3dgs-exp/.venv/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Python依存
/root/repos/3dgs-exp/.venv/bin/python -m pip install plyfile tqdm opencv-python joblib

# CUDA拡張（3DGS必須）
/root/repos/3dgs-exp/.venv/bin/python -m pip install --no-build-isolation \
	./submodules/simple-knn \
	./submodules/fused-ssim \
	./submodules/diff-gaussian-rasterization
```

#### ふだん有効化するとき

```bash
source /root/repos/3dgs-exp/.venv/bin/activate
```

#### 動作確認（実施済み）

```bash
/root/repos/3dgs-exp/.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

期待値: `2.10.0+cu128 True`

### 2) 学習実行

以下は `gaussian-splatting/data/sample.mp4` で実際に実行・検証した手順です。

#### 実際に実行した前処理（動画 → 3DGSデータ）

```bash
cd /root/repos/3dgs-exp/gaussian-splatting

# 1) フレーム抽出（3fps、長辺縮小）
mkdir -p data/sample_scene/input
ffmpeg -y -i data/sample.mp4 -vf "fps=3,scale=1280:-1" -q:v 2 data/sample_scene/input/frame_%04d.jpg

# 2) COLMAPで再構成 + 3DGS向け形式へ変換
/root/repos/3dgs-exp/.venv/bin/python convert.py -s /root/repos/3dgs-exp/gaussian-splatting/data/sample_scene
```

#### 前処理コマンドで得られる出力構造

`ffmpeg` 後:

```text
gaussian-splatting/data/sample_scene/
	input/
		frame_0001.jpg
		frame_0002.jpg
		...
```

`convert.py` 後:

```text
gaussian-splatting/data/sample_scene/
	input/                  # 元フレーム
	images/                 # undistort済み画像（train.pyが読む）
	sparse/0/
		cameras.bin
		images.bin
		points3D.bin
	distorted/              # COLMAP中間成果物
	stereo/                 # COLMAP成果物
	run-colmap-geometric.sh
	run-colmap-photometric.sh
```

#### 実際に実行した学習

```bash
cd /root/repos/3dgs-exp/gaussian-splatting

# 実測学習（1000 iter）
/root/repos/3dgs-exp/.venv/bin/python train.py \
	-s /root/repos/3dgs-exp/gaussian-splatting/data/sample_scene \
	-m /root/repos/3dgs-exp/gaussian-splatting/output/sample_scene_1k \
	--iterations 1000 \
	--save_iterations 500 1000 \
	--test_iterations -1
```

#### 学習コマンドで得られる出力構造

```text
gaussian-splatting/output/sample_scene_1k/
	cfg_args
	cameras.json
	exposure.json
	input.ply
	point_cloud/
		iteration_500/
			point_cloud.ply
		iteration_1000/
			point_cloud.ply
```

#### 実際に実行したレンダリング確認

```bash
/root/repos/3dgs-exp/.venv/bin/python render.py \
	-m /root/repos/3dgs-exp/gaussian-splatting/output/sample_scene_1k \
	--iteration 1000
```

#### レンダリング時のパス指定（今回の例）

- `-m` には「学習出力ディレクトリ」を指定する
	- 今回: `/root/repos/3dgs-exp/gaussian-splatting/output/sample_scene_1k`
- `--iteration` は任意
	- `--iteration 1000` を付けると `point_cloud/iteration_1000` を明示的に使用
	- 省略時は最新反復が自動選択される（今回も実行確認済み）

相対パスで書く場合の例:

```bash
cd /root/repos/3dgs-exp/gaussian-splatting
python render.py -m output/sample_scene_1k
```

#### レンダリングコマンドで得られる出力構造

```text
gaussian-splatting/output/sample_scene_1k/
	train/
		ours_1000/
			renders/            # 生成画像
			gt/                 # 対応するGT画像
	test/
		ours_1000/
			renders/            # test splitがある場合
			gt/
```

今回の sample では train 側に 49枚の `renders` と 49枚の `gt` が生成されました。

#### 実行結果サマリ（sample.mp4）

- 動画情報: 1920x1080, 約16.3秒, 487フレーム
- 抽出後: 49枚（3fps）
- COLMAP: 49 images / 2234 points で再構成成功
- 学習: 1000 iter を約62秒で完了
- 学習ログ抜粋:
	- iter 500: Loss=0.0852573
	- iter 1000: Loss=0.0623016
- 出力: `output/sample_scene_1k/point_cloud/iteration_500`, `iteration_1000`
- レンダリング: train split 49/49 枚を出力

#### どのようなデータ構造にするべきか

最小構成は次の通りです。

```text
<scene_root>/
	input/                  # フレーム画像（JPG/PNG）
```

`convert.py` 実行後に、学習に必要な構造が自動で揃います。

```text
<scene_root>/
	images/                 # undistort済み画像（train.py が参照）
	sparse/0/               # COLMAP sparse（cameras.bin / images.bin / points3D.bin）
	distorted/              # COLMAP中間生成物
	stereo/                 # COLMAP生成物
```

`train.py` は基本的に `<scene_root>` を `-s` に渡せば動作します。

#### 動画をどのように加工するべきか

- フレーム間の視差を作るため、カメラはゆっくり移動しながら撮影する
- 強い手ブレ・モーションブラー・露出急変を避ける
- fps抽出は 2〜5fps 目安（似すぎた連番を減らし、COLMAPを安定化）
- 解像度は長辺 1280〜1600 程度が扱いやすい
- 反射・透明体・動体が多い動画は失敗しやすい
- 縦動画は回転メタデータの影響を受けるため、抽出画像の向きと解像度を確認する

### 3) 推論/レンダリング

```bash
cd /root/repos/3dgs-exp/gaussian-splatting
python render.py -m output/sample_scene_1k
```

## 補足

- 現在の環境では `torch 2.10.0+cu128` と CUDA が有効 (`torch.cuda.is_available() == True`) であることを確認済みです。
- 公式 README: `gaussian-splatting/README.md`
