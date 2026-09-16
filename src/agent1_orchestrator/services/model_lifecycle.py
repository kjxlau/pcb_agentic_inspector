from pathlib import Path
import yaml
from services.common import ServiceResult
from inference.onnx_classifier import ONNXImageClassifier

class ModelLifecycleService:
    """Resolve and load the four approved ONNX models used by the two-stage pipeline."""

    def __init__(self, project_root, config_path="config/models.yaml"):
        self.project_root = Path(project_root)
        with open(self.project_root/config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)["models"]
        self._cache = {}

    def get_model(self, key):
        key = key.lower()
        if key not in self.config:
            return ServiceResult(False, "MODEL_NOT_CONFIGURED", errors=[key])
        if key not in self._cache:
            cfg = self.config[key]
            model = self.project_root/cfg["path"]
            labels = self.project_root/cfg["labels"]
            if not model.exists() or not labels.exists():
                return ServiceResult(False, "MODEL_FILES_MISSING",
                                     errors=[str(model), str(labels)], recoverable=True)
            self._cache[key] = ONNXImageClassifier(model, labels, cfg["input_size"])
        return ServiceResult(True, "MODEL_AVAILABLE", data={"model": self._cache[key], "model_key": key})
