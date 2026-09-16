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

# PCBInspect-Region

Part of the **SentinelPCB defect-inspection router**: a region classifier dispatches each
component ROI crop to a region-specific defect classifier. Sibling repos:
[PCBInspect-Region](https://huggingface.co/JcProg/PCBInspect-Region), [PCBInspect-BodyDefect](https://huggingface.co/JcProg/PCBInspect-BodyDefect), [PCBInspect-LeadDefect](https://huggingface.co/JcProg/PCBInspect-LeadDefect), [PCBInspect-TextDefect](https://huggingface.co/JcProg/PCBInspect-TextDefect).

Companion structural-feature detector (unrelated task — detects MountingHole/ComponentBody/
SolderJoint/Lead, not defects): [PCBInspect-AI](https://huggingface.co/JcProg/PCBInspect-AI).

## Role

First stage. Given a component ROI crop, predicts which physical region it is (`Body`, `Lead`, `Text`) so the pipeline can dispatch to the matching defect classifier. Near-trivial task — the three regions look visually distinct.

## Model

- Base: `yolo26n-cls` ([Ultralytics](https://docs.ultralytics.com/models/yolo26/)),
  classification head, fine-tuned on AOI component-ROI crops.
- Export: ONNX, opset 17, no NMS (classification only) — single input
  `images` `(1, 3, 224, 224)` RGB, normalized `/255`, NCHW; single output `output0`
  `(1, 3)` raw logits (apply softmax yourself for probabilities).
- Classes (3), index order = `labels.json`: **Body, Lead, Text**.

## Data

Trained on a proprietary AOI dataset of SMT component-ROI crops (paired defect-free reference +
defective capture per physical site), not publicly released. Split is grouped by physical
capture site (never by raw image) so a component's reference and defect crop never straddle
train/val/test.

## Metrics

**val** (top-1 1.000, macro-F1 1.000, n=2004):

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Body | 0.999 | 1.000 | 1.000 | 1174 |
| Lead | 1.000 | 0.999 | 0.999 | 738 |
| Text | 1.000 | 1.000 | 1.000 | 92 |

**test** (top-1 1.000, macro-F1 1.000, n=2002):

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Body | 0.999 | 1.000 | 1.000 | 1164 |
| Lead | 1.000 | 0.999 | 0.999 | 736 |
| Text | 1.000 | 1.000 | 1.000 | 102 |

## Limitations

None observed; effectively solved on held-out data (n=2,002 test).

## Usage

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
img = Image.open("crop.jpg").convert("RGB").resize((224, 224))
x = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
(logits,) = sess.run(None, {"images": x})
probs = np.exp(logits) / np.exp(logits).sum()
print(probs)
```
