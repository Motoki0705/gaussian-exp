# DINOv3 最小メモ

このリポジトリでは、以下で DINOv3 ViT-S/16 をロードできます。

- ローダー: `third_party/dinov3/api/dinov3_loader.py`
- チェックポイント: `checkpoints/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`

## 前提

`torch.hub.load(..., source="local")` で `third_party/dinov3/hubconf.py` を読み込むため、DINOv3 依存が必要です。

```bash
. .venv/bin/activate
pip install -r third_party/dinov3/requirements.txt
```

## モデルロードと推論

```python
import torch
from third_party.dinov3.api.dinov3_loader import get_dinov3_vits16

ckpt = "checkpoints/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
model = get_dinov3_vits16(ckpt).eval().cuda()

x = torch.randn(1, 3, 224, 224, device="cuda")
with torch.inference_mode():
    y = model(x)

print(type(y), y.shape, y.dtype)  # torch.Tensor, (1, 384), torch.float32
```

## スループット・スモークテスト例

```python
import time
import torch
from third_party.dinov3.api.dinov3_loader import get_dinov3_vits16

ckpt = "checkpoints/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
model = get_dinov3_vits16(ckpt).eval().cuda()

batch_size = 64
warmup = 20
iters = 80
x = torch.randn(batch_size, 3, 224, 224, device="cuda")

for _ in range(warmup):
    with torch.inference_mode():
        _ = model(x)

torch.cuda.synchronize()
t1 = time.perf_counter()
for _ in range(iters):
    with torch.inference_mode():
        _ = model(x)
torch.cuda.synchronize()
t2 = time.perf_counter()

elapsed = t2 - t1
imgs = batch_size * iters
print("throughput_images_per_sec=", imgs / elapsed)
print("avg_latency_ms_per_batch=", (elapsed / iters) * 1000)
print("avg_latency_ms_per_image=", (elapsed / imgs) * 1000)
```

## 実測（2026-02-18）

- 環境: `torch 2.10.0+cu128`, `NVIDIA GeForce RTX 5060 Ti`
- ロード時間: 約 `1.58s`
- 出力: `torch.Tensor` shape `(1, 384)`
- スループット: `729.65 img/s`（batch=64, warmup=20, iters=80）
