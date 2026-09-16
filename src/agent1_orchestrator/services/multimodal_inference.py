from services.common import ServiceResult
from inference.router import route_feature

class TwoStageInferenceService:
    """
    Executes the supplied image-only ONNX models.
    Golden image + measurements are retained as context/validation inputs;
    the current supplied classifiers themselves accept one RGB image tensor.
    """

    def __init__(self, model_lifecycle, feature_confidence_threshold=0.70):
        self.models = model_lifecycle
        self.feature_threshold = feature_confidence_threshold

    def infer_sample(self, sample):
        feature_model_result = self.models.get_model("feature")
        if not feature_model_result.success:
            return feature_model_result

        # Stage 1 uses the defect ROI crop, consistent with supplied model documentation.
        stage1 = feature_model_result.data["model"].predict(sample["defect_image"])
        predicted_feature = stage1["prediction"]

        if stage1["confidence"] < self.feature_threshold:
            return ServiceResult(
                False, "FEATURE_CLASSIFICATION_UNCERTAIN",
                data={"sample_id": sample["sample_id"], "feature_classification": stage1},
                recoverable=True, next_action="review"
            )

        route = route_feature(predicted_feature)
        if route is None:
            return ServiceResult(False, "UNSUPPORTED_FEATURE",
                                 data={"feature_classification": stage1}, recoverable=True)

        defect_model_result = self.models.get_model(route)
        if not defect_model_result.success:
            return defect_model_result
        stage2 = defect_model_result.data["model"].predict(sample["defect_image"])

        source_feature = sample.get("source_feature")
        machine_defect = sample.get("machine_defect", "")
        normalized_machine = machine_defect.split("_")[0].replace("Insuffcient", "Insufficient")
        return ServiceResult(
            True, "INFERENCE_COMPLETED",
            data={
                "sample_id": sample["sample_id"],
                "source_feature": source_feature,
                "machine_defect": machine_defect,
                "feature_classification": stage1,
                "routing": {"selected_model": route},
                "defect_classification": stage2,
                "comparison": {
                    "feature_agreement": source_feature.lower() == predicted_feature.lower(),
                    "defect_agreement": normalized_machine.lower() == stage2["prediction"].lower(),
                },
                "failed_inspections": sample.get("failed_inspections", {}),
            }
        )
