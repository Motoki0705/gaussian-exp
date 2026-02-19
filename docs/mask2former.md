# Mask2Former 運用メモ

このドキュメントは、`transformers` の `Mask2Former` を使って panoptic segmentation を行う最小手順をまとめる。

## 1. 依存関係

```bash
pip install "transformers==4.49.0" "huggingface-hub==0.36.2" safetensors
```

## 2. Build（processor / model / id2label）

```python
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

model_name = "facebook/mask2former-swin-tiny-coco-panoptic"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained(model_name)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(device).eval()
id2label = dict(model.config.id2label)
```

- `processor`: 前処理と後処理（`post_process_panoptic_segmentation`）を担当
- `model`: 生の logits を出力
- `id2label`: `label_id` からクラス名へ変換

## 3. 推論（panoptic）

```python
import numpy as np
from PIL import Image

image_np = ...  # HxWx3 RGB
h, w = image_np.shape[:2]
image_pil = Image.fromarray(image_np, mode="RGB")

inputs = processor(images=image_pil, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

processed = processor.post_process_panoptic_segmentation(
    outputs,
    target_sizes=[(h, w)],
)[0]

segmentation = processed["segmentation"].cpu().numpy().astype(np.int32)  # [H,W]
segments_info = processed["segments_info"]  # list[dict]
```

- `segmentation`: ピクセルごとの segment id マップ
- `segments_info`: 各 segment のメタ情報
  - 代表キー: `id`, `label_id`, `score`, `was_fused`

## 4. `label_id` の解決

```python
for s in segments_info:
    label_id = int(s["label_id"])
    label_name = id2label.get(label_id, f"class_{label_id}")
```

これで panoptic の segment ごとにクラス名を復元できる。
