from types import SimpleNamespace

import pytest
from PIL import Image

from jitpi05.jitrl.vlm import (
    CycleFailurePrediction,
    cycle_failure_predictor_prompt,
    predict_cycle_failure,
    validate_cycle_prediction_target,
)


def prediction(next_subtask: str = "grasp the mug") -> CycleFailurePrediction:
    return CycleFailurePrediction.model_validate(
        {
            "next_subtask": next_subtask,
            "type": "backtrack",
            "reason": "The gripper is not aligned with the target.",
            "front_view_evidence": ["The gripper is offset from the mug."],
            "wrist_view_evidence": ["No stable contact is visible."],
            "assessment": {
                "success_likelihood": "low",
                "key_risks": "misalignment",
                "view_agreement": "agree",
                "view_dominance": "wrist confirms contact risk",
                "decision_basis": "low likelihood with matching evidence",
            },
        }
    )


def test_cycle_prompt_is_transit_default_and_two_view_explicit() -> None:
    prompt = cycle_failure_predictor_prompt(
        "put the mug in the basket",
        ["approach the mug", "grasp the mug", "place the mug at the basket"],
        "place the mug at the basket",
    )

    assert "approximately 75%" in prompt
    assert "Default to transit" in prompt
    assert "FRONT view" in prompt
    assert "WRIST view" in prompt
    assert "Never terminate the episode" in prompt
    assert "copied exactly" in prompt
    assert "next_subtask" in prompt


def test_cycle_target_validation_requires_exact_recorded_anchor() -> None:
    valid = prediction()
    assert validate_cycle_prediction_target(valid, ["approach the mug", "grasp the mug"]) is valid

    with pytest.raises(ValueError, match="exact recorded anchor"):
        validate_cycle_prediction_target(
            prediction("grasp the mug with extra reasoning"),
            ["approach the mug", "grasp the mug"],
        )


def test_cycle_prediction_sends_exactly_two_views_and_serializes_result() -> None:
    class FakePredictor:
        def __init__(self) -> None:
            self.arguments = None

        def run_sync(self, arguments):
            self.arguments = arguments
            return SimpleNamespace(output=prediction())

    predictor = FakePredictor()
    images = [Image.new("RGB", (8, 8), "black"), Image.new("RGB", (8, 8), "white")]
    result = predict_cycle_failure(
        predictor,
        images,
        "put the mug in the basket",
        ["approach the mug", "grasp the mug"],
        "grasp the mug",
    )

    assert len(predictor.arguments) == 3
    assert result["next_subtask"] == "grasp the mug"
    assert result["type"] == "backtrack"
    assert result["backend"] == "pydantic_ai_google"
    assert "front_view_evidence" in result["raw_output"]


def test_cycle_prediction_rejects_wrong_image_count() -> None:
    class UnusedPredictor:
        def run_sync(self, _arguments):
            raise AssertionError("must not call predictor")

    with pytest.raises(ValueError, match="exactly front and wrist"):
        predict_cycle_failure(
            UnusedPredictor(),
            [Image.new("RGB", (8, 8))],
            "task",
            ["approach the mug"],
            "approach the mug",
        )


def test_cycle_schema_rejects_empty_evidence() -> None:
    value = prediction().model_dump()
    value["front_view_evidence"] = []
    with pytest.raises(ValueError):
        CycleFailurePrediction.model_validate(value)
