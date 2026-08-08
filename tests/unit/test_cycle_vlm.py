from types import SimpleNamespace

import pytest
from PIL import Image

from jitpi05.jitrl.cycle import (
    build_cycle_subtask_program,
    format_cycle_subtask_program,
)
from jitpi05.jitrl.vlm import (
    CycleFailurePrediction,
    cycle_failure_predictor_prompt,
    predict_cycle_failure,
    validate_cycle_prediction_target,
)


def prediction(next_subtask: str = "subtask-01") -> CycleFailurePrediction:
    return CycleFailurePrediction.model_validate(
        {
            "type": "backtrack",
            "success_likelihood": "low",
            "next_subtask": next_subtask,
            "reason": "The gripper is not aligned with the target.",
            "front_evidence": "The target is not centered in FRONT.",
            "wrist_evidence": "WRIST shows unstable contact.",
        }
    )


def test_cycle_prompt_is_transit_default_and_minimal_schema() -> None:
    program = format_cycle_subtask_program(
        build_cycle_subtask_program("pick up the mug and place it in the basket")
    )
    prompt = cycle_failure_predictor_prompt(
        "pick up the mug and place it in the basket",
        program,
        "subtask-02: Move above the basket while holding the mug",
    )

    assert "Default to type=transit" in prompt
    assert "Never terminate the episode" in prompt
    assert "FRONT" in prompt and "WRIST" in prompt
    assert "next_subtask" in prompt
    assert '"success_likelihood"' in prompt
    assert "six fields" in prompt
    assert "subtask-00" in prompt or "exact IDs" in prompt
    assert '"type": "transit"' in prompt
    assert '"next_subtask": ""' in prompt
    assert "complete valid output example" in prompt
    assert "do not emit Markdown fences" in prompt


def test_cycle_target_validation_requires_exact_stable_program_target() -> None:
    valid = prediction()
    assert validate_cycle_prediction_target(valid, ["subtask-00: approach the mug", "subtask-01: grasp the mug"]) is valid

    with pytest.raises(ValueError, match="exact stable program target"):
        validate_cycle_prediction_target(
            prediction("subtask-01: grasp the mug with extra reasoning"),
            ["subtask-00: approach the mug", "subtask-01: grasp the mug"],
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
        ["subtask-00: approach the mug", "subtask-01: grasp the mug"],
        "subtask-01: grasp the mug",
    )

    assert len(predictor.arguments) == 3
    assert result["next_subtask"] == "subtask-01"
    assert result["type"] == "backtrack"
    assert result["backend"] == "pydantic_ai_google"
    assert result["schema_valid"] is True
    assert result["front_evidence"]
    assert result["wrist_evidence"]
    assert "success_likelihood" in result["raw_output"]


def test_cycle_prediction_rejects_wrong_image_count() -> None:
    class UnusedPredictor:
        def run_sync(self, _arguments):
            raise AssertionError("must not call predictor")

    with pytest.raises(ValueError, match="exactly front and wrist"):
        predict_cycle_failure(
            UnusedPredictor(),
            [Image.new("RGB", (8, 8))],
            "task",
            ["subtask-00: approach the mug"],
            "subtask-00: approach the mug",
        )


def test_cycle_schema_accepts_partial_model_output() -> None:
    value = CycleFailurePrediction.model_validate({"type": "transit"})
    assert value.success_likelihood == "medium"
    assert value.next_subtask == ""
    assert value.reason == ""
    assert value.front_evidence == ""
    assert value.wrist_evidence == ""
