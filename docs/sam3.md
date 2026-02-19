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

- `/path/to/sam3.pt`（Hugging Face: `facebook/sam3`）
- `/path/to/config.json`（Hugging Face: `facebook/sam3`）
- `/path/to/bpe_simple_vocab_16e6.txt.gz`（CLIP BPE）

注意:
- 実運用では `checkpoint_path` と `bpe_path` を明示する。
- 環境差異や自動ダウンロード失敗を避けるため、ローカル固定パス運用を推奨する。

## 4. コードでの基本利用フロー（動画）

SAM3 の動画推論は、以下の 4 ステップで扱う。

1. `build_sam3_video_predictor(...)` で予測器を構築  
2. `start_session` で対象動画を登録  
3. `add_prompt` + `propagate_in_video` で時系列推論  
4. `close_session` で必ず後処理

```python
from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor(
    checkpoint_path="/path/to/sam3.pt",
    bpe_path="/path/to/bpe_simple_vocab_16e6.txt.gz",
    gpus_to_use=[0],
)

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
