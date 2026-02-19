# Depth Anything V2 運用メモ

このドキュメントは、`Depth Anything V2` を `transformers` 経由で使って単眼深度を推論する最小手順をまとめる。

## 1. 依存関係

```bash
pip install "transformers==4.49.0" "huggingface-hub==0.36.2" safetensors
```

## 2. Build（pipeline）

```python
import torch
from transformers import pipeline

model_name = "depth-anything/Depth-Anything-V2-Small-hf"
device = 0 if torch.cuda.is_available() else -1

pipe = pipeline(
    task="depth-estimation",
    model=model_name,
    device=device,
)
```

## 3. 推論

```python
import numpy as np
from PIL import Image

image_np = ...  # HxWx3 RGB
image_pil = Image.fromarray(image_np, mode="RGB")

out = pipe(image_pil)
# out["depth"] は PIL Image

depth = np.asarray(out["depth"]).astype(np.float32)  # [H,W]
```

補足:
- この `depth` は相対深度として扱う。
- 可視化時は `min-max` 正規化して色付けする。

## 4. モデルサイズの選択例

- `depth-anything/Depth-Anything-V2-Small-hf`
- `depth-anything/Depth-Anything-V2-Base-hf`
- `depth-anything/Depth-Anything-V2-Large-hf`

大きいモデルほど精度は上がりやすいが、VRAM/推論時間コストも増える。

## 5. このリポジトリでのデモ

スクリプト:
- `exp/depth_anything_v2_demo.py`

実行:

```bash
python exp/depth_anything_v2_demo.py
```

主な引数:

```bash
python exp/depth_anything_v2_demo.py \
  --image-path data/tandt_db/db/playroom/images/DSC05572.jpg \
  --output-dir exp/depth_anything_v2_demo \
  --model-name depth-anything/Depth-Anything-V2-Small-hf \
  --device cuda
```

出力:
- `original.png`
- `depth_raw.npy`
- `depth_vis.png`
- `depth_overlay.png`
- `meta.txt`
