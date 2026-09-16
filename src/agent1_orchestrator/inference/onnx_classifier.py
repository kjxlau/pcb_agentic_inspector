import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

class ONNXImageClassifier:
    def __init__(self, model_path, labels_path, input_size):
        self.model_path = str(model_path)
        self.input_size = tuple(input_size)
        with open(labels_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.labels = {int(k): v for k, v in raw.items()}
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, image_path):
        # Matches the preprocessing documented with the supplied models:
        # RGB -> direct resize -> /255 -> NCHW.
        image = Image.open(image_path).convert("RGB").resize(self.input_size)
        x = np.asarray(image, dtype=np.float32) / 255.0
        return x.transpose(2, 0, 1)[None, ...]

    def predict(self, image_path):
        x = self._preprocess(image_path)
        logits = np.asarray(self.session.run(None, {self.input_name: x})[0]).reshape(-1)
        z = logits - logits.max()
        probs = np.exp(z) / np.exp(z).sum()
        idx = int(np.argmax(probs))
        return {
            "prediction": self.labels[idx],
            "confidence": float(probs[idx]),
            "probabilities": {self.labels[i]: float(probs[i]) for i in range(len(probs))}
        }
