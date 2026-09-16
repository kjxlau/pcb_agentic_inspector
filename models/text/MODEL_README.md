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

# PCBInspect-TextDefect

Part of the **SentinelPCB defect-inspection router**: a region classifier dispatches each
component ROI crop to a region-specific defect classifier. Sibling repos:
[PCBInspect-Region](https://huggingface.co/JcProg/PCBInspect-Region), [PCBInspect-BodyDefect](https://huggingface.co/JcProg/PCBInspect-BodyDefect), [PCBInspect-LeadDefect](https://huggingface.co/JcProg/PCBInspect-LeadDefect), [PCBInspect-TextDefect](https://huggingface.co/JcProg/PCBInspect-TextDefect).

Companion structural-feature detector (unrelated task — detects MountingHole/ComponentBody/
SolderJoint/Lead, not defects): [PCBInspect-AI](https://huggingface.co/JcProg/PCBInspect-AI).

## Role

Defect classifier for crops routed as `Text` (silkscreen). Binary: defect-free vs wrong part printed.

## Model

- Base: `yolo26n-cls` ([Ultralytics](https://docs.ultralytics.com/models/yolo26/)),
  classification head, fine-tuned on AOI component-ROI crops.
- Export: ONNX, opset 17, no NMS (classification only) — single input
  `images` `(1, 3, 480, 480)` RGB, normalized `/255`, NCHW; single output `output0`
  `(1, 2)` raw logits (apply softmax yourself for probabilities).
- Classes (2), index order = `labels.json`: **Golden, WrongPart**.

## Data

Trained on a proprietary AOI dataset of SMT component-ROI crops (paired defect-free reference +
defective capture per physical site), not publicly released. Split is grouped by physical
capture site (never by raw image) so a component's reference and defect crop never straddle
train/val/test.

## Metrics

**val** (top-1 1.000, macro-F1 1.000, n=94):

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Golden | 1.000 | 1.000 | 1.000 | 53 |
| WrongPart | 1.000 | 1.000 | 1.000 | 41 |

**test** (top-1 1.000, macro-F1 1.000, n=97):

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Golden | 1.000 | 1.000 | 1.000 | 53 |
| WrongPart | 1.000 | 1.000 | 1.000 | 44 |

## Limitations

Smallest training set of the four (476 images total). Test top-1/macro-F1 both 1.000 (n=97), but the small, low-diversity test set means this number should be treated cautiously rather than as a tight confidence interval.

## Usage

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
img = Image.open("crop.jpg").convert("RGB").resize((480, 480))
x = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
(logits,) = sess.run(None, {"images": x})
probs = np.exp(logits) / np.exp(logits).sum()
print(probs)
```
