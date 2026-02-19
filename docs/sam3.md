# SAM3 運用ガイド

このドキュメントは、`sam3` パッケージを Python コード内で安定運用するための要点をまとめる。

## 1. 必須要件

- GPU が使えること（`torch.cuda.is_available() == True`）
- Hugging Face 認証済みで `facebook/sam3` にアクセスできること
- `sam3` と互換のある `torch` を使うこと  
  (`torch==2.0.0` では `torch.nn.attention` がなく動作不可だったため、`2.10.0` で運用)

## 2. 依存関係の準備

任意の仮想環境で実行:

```bash
pip install "sam3>=0.1.2"
pip install "torch>=2.5.0" "torchvision>=0.20.0"
```

## 3. モデルファイルの準備（固定推奨）

以下は、`checkpoints/sam3/` に必要ファイルを固定配置する手順。

```bash
# 1) 保存先を作成
mkdir -p checkpoints/sam3

# 2) Hugging Face へログイン（未ログイン時）
hf auth login

# 3) sam3.pt を取得（facebook/sam3）
huggingface-cli download facebook/sam3 sam3.pt \
  --local-dir checkpoints/sam3 \
  --local-dir-use-symlinks False

# 4) CLIP BPE を取得
curl -L \
  -o checkpoints/sam3/bpe_simple_vocab_16e6.txt.gz \
  https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz

# 5) 配置確認
ls -lh checkpoints/sam3
```

期待されるファイル:
- `checkpoints/sam3/sam3.pt`
- `checkpoints/sam3/bpe_simple_vocab_16e6.txt.gz`

## 4. コードでの基本利用フロー

### 4.1 Build

動画推論器を直接使う場合:

```python
from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor(
    checkpoint_path="/path/to/sam3.pt",
    bpe_path="/path/to/bpe_simple_vocab_16e6.txt.gz",
    gpus_to_use=[0],
)
```

ポイント推論（画像）を使う場合:

```python
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model_builder import build_sam3_video_model

model = build_sam3_video_model(
    checkpoint_path="/path/to/sam3.pt",
    bpe_path="/path/to/bpe_simple_vocab_16e6.txt.gz",
    load_from_HF=False,
    device="cuda",
)

# image predictor は tracker.backbone を参照するため差し替える
model.tracker.backbone = model.detector.backbone
point_predictor = SAM3InteractiveImagePredictor(model.tracker)
```

### 4.2 動画推論

`start_session -> add_prompt -> propagate_in_video -> close_session` の順で使う。

```python
resp = video_predictor.handle_request(
    {"type": "start_session", "resource_path": "/path/to/video.mp4"}
)
session_id = resp["session_id"]

try:
    first = video_predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": "sky",
        }
    )
    # first["outputs"] を保存/後処理

    for step in video_predictor.handle_stream_request(
        {"type": "propagate_in_video", "session_id": session_id}
    ):
        frame_idx = step["frame_index"]
        outputs = step["outputs"]
        # frame_idx ごとに outputs を保存/後処理
finally:
    video_predictor.handle_request({"type": "close_session", "session_id": session_id})
```

### 4.3 Point 推論（画像）

ポイントは `point_coords: [B, N, 2]`、ラベルは `point_labels: [B, N]`。
単一画像で1プロンプトを投げる場合は `B=1`。

```python
import numpy as np

image_np = ...  # HxWx3 RGB
point_predictor.set_image(image_np)

# 例: 2点 (x, y)
points = np.array([[400.0, 300.0], [420.0, 320.0]], dtype=np.float32)
labels = np.array([1, 1], dtype=np.int32)  # 1: FG, 0: BG

masks, ious, low_res = point_predictor.predict(
    point_coords=points[None, :, :],
    point_labels=labels[None, :],
    multimask_output=True,
    normalize_coords=False,
)

# 戻りの代表 shape:
# masks: [C, H, W], ious: [C], low_res: [C, 256, 256]
```

## 5. 実装時の設計ポイント

- 動画の分割単位:
  - 時系列整合性を保つ単位（例: カメラ単位）で動画を分けて推論する。
- 出力の頑健化:
  - `out_binary_masks` が空配列のケースを想定し、ゼロマスク補完やスキップ方針を定義する。
- セッション管理:
  - 例外が発生しても `close_session` を呼ぶ。
- 再現性:
  - `sam3` / `torch` のバージョン固定と、`checkpoint_path` / `bpe_path` の固定をセットで運用する。

## 6. 典型エラーと対処

- `ModuleNotFoundError: torch.nn.attention`
  - `torch` が古い。SAM3 互換バージョンへ更新する。
- `FileNotFoundError: ... bpe_simple_vocab_16e6.txt.gz`
  - BPE ファイルを配置し、`bpe_path` を明示する。
- チェックポイント取得失敗（HF 認証/権限）
  - `hf auth login` を再実行し、`facebook/sam3` へのアクセス権を確認する。
- 推論出力が空
  - マスク保存側で空出力ハンドリングを実装する（ゼロ埋め/再試行/スキップ）。
