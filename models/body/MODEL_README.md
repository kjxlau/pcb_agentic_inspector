---
license: mit
tags:
- image-classification
- pcb
- aoi
- yolo26
- onnx
- computer-vision
- dataset:custom
library_name: ultralytics
pipeline_tag: image-classification
---

# PCBInspect-BodyDefect

Part of the **SentinelPCB defect-inspection router**: a region classifier dispatches each
component ROI crop to a region-specific defect classifier. Sibling repos:
[PCBInspect-Region](https://huggingface.co/JcProg/PCBInspect-Region), [PCBInspect-BodyDefect](https://huggingface.co/JcProg/PCBInspect-BodyDefect), [PCBInspect-LeadDefect](https://huggingface.co/JcProg/PCBInspect-LeadDefect), [PCBInspect-TextDefect](https://huggingface.co/JcProg/PCBInspect-TextDefect).

Companion structural-feature detector (unrelated task — detects MountingHole/ComponentBody/
SolderJoint/Lead, not defects): [PCBInspect-AI](https://huggingface.co/JcProg/PCBInspect-AI).

## Role

Defect classifier for crops routed as `Body` by the region classifier. The hardest of the three specialists — six classes with real class imbalance.

## Model

- Base: `yolo26s-cls` ([Ultralytics](https://docs.ultralytics.com/models/yolo26/)),
  classification head, fine-tuned on AOI component-ROI crops.
- Export: ONNX, opset 17, no NMS (classification only) — single input
  `images` `(1, 3, 640, 640)` RGB, normalized `/255`, NCHW; single output `output0`
  `(1, 6)` raw logits (apply softmax yourself for probabilities).
- Classes (6), index order = `labels.json`: **ForeignMaterial, Golden, MissingPart, Shift, Tombstone, WrongPart**.

## Data

Trained on a proprietary AOI dataset of SMT component-ROI crops (paired defect-free reference +
defective capture per physical site), not publicly released. Split is grouped by physical
capture site (never by raw image) so a component's reference and defect crop never straddle
train/val/test.

## Metrics

**val** (top-1 0.948, macro-F1 0.875, n=784):

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| ForeignMaterial | 0.996 | 0.975 | 0.985 | 276 |
| Golden | 1.000 | 1.000 | 1.000 | 200 |
| MissingPart | 0.706 | 0.706 | 0.706 | 17 |
| Shift | 0.929 | 0.897 | 0.913 | 146 |
| Tombstone | 0.773 | 0.739 | 0.756 | 23 |
| WrongPart | 0.851 | 0.934 | 0.891 | 122 |

**test** (top-1 0.941, macro-F1 0.871, n=780):

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| ForeignMaterial | 0.996 | 0.963 | 0.979 | 267 |
| Golden | 0.995 | 1.000 | 0.998 | 200 |
| MissingPart | 1.000 | 0.500 | 0.667 | 22 |
| Shift | 0.901 | 0.938 | 0.919 | 146 |
| Tombstone | 0.833 | 0.769 | 0.800 | 26 |
| WrongPart | 0.813 | 0.916 | 0.862 | 119 |

## Limitations

`MissingPart` and `Tombstone` are the weak classes (94 and 131 raw examples total, concentrated on 12 and 15 distinct part-numbers respectively). Test: MissingPart precision 1.00 / recall 0.50 (conservative — misses about half, never false-alarms); Tombstone F1 0.80. More examples across more board/package designs would improve both; see the companion data-exploration notebook.

## Usage

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
img = Image.open("crop.jpg").convert("RGB").resize((640, 640))
x = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
(logits,) = sess.run(None, {"images": x})
probs = np.exp(logits) / np.exp(logits).sum()
print(probs)
```
