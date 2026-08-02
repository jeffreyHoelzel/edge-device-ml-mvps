import torch
import pytest

from contract import CHECKPOINT_FORMAT_VERSION, CLASS_NAMES, IMPACT_NAMES, INPUT_SHAPE, NORMALIZATION_NAME
from infer import build_result, confidence_threshold, load_model, validate_checkpoint, validate_frequency_span
from model import NUM_CLASSES, NUM_IMPACT_LEVELS, RFInterferenceCNN


def _checkpoint() -> dict[str, object]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state": RFInterferenceCNN().state_dict(),
        "input_shape": list(INPUT_SHAPE),
        "normalization": NORMALIZATION_NAME,
        "model_config": {"num_classes": NUM_CLASSES, "num_impact_levels": NUM_IMPACT_LEVELS},
        "class_names": list(CLASS_NAMES),
        "impact_names": list(IMPACT_NAMES),
        "training_seed": 42,
        "best_epoch": 1,
        "validation_metrics": {"class_accuracy": 0.5, "impact_accuracy": 0.5, "bounds_mae": 0.2},
    }


def test_checkpoint_round_trip_loads_a_cpu_model(tmp_path) -> None:
    path = tmp_path / "model.pt"
    torch.save(_checkpoint(), path)
    model = load_model(path)
    assert not model.training


def test_checkpoint_validation_reports_incompatible_metadata() -> None:
    checkpoint = _checkpoint()
    checkpoint["format_version"] = CHECKPOINT_FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="format_version"):
        validate_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("probabilities", "expected_status", "expected_bounds", "expected_impact"),
    [
        (torch.tensor([0.1, 0.7, 0.1, 0.05, 0.05]), "detected", (0.2, 0.6), "high"),
        (torch.tensor([0.8, 0.1, 0.05, 0.03, 0.02]), "no_interference", (0.0, 0.0), "low"),
        (torch.tensor([0.1, 0.2, 0.5, 0.1, 0.1]), "abstained", (0.0, 0.0), "low"),
    ],
)
def test_result_status_handles_detected_empty_and_abstained_predictions(
    probabilities, expected_status, expected_bounds, expected_impact
) -> None:
    result = build_result(probabilities, torch.tensor([0.0, 0.0, 2.0]), torch.tensor([0.2, 0.6]), 0.6)
    assert result["detection_status"] == expected_status
    assert (result["frequency_start_normalized"], result["frequency_stop_normalized"]) == expected_bounds
    assert result["estimated_service_impact"] == expected_impact


def test_result_detects_at_the_exact_confidence_threshold() -> None:
    result = build_result(torch.tensor([0.2, 0.6, 0.2, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.2, 0.6]), 0.6)
    assert result["detection_status"] == "detected"


def test_result_maps_detected_normalized_bounds_to_hz() -> None:
    result = build_result(
        torch.tensor([0.1, 0.7, 0.1, 0.05, 0.05]),
        torch.tensor([0.0, 0.0, 2.0]),
        torch.tensor([0.25, 0.75]),
        0.6,
        (100.0, 200.0),
    )
    assert result["frequency_start_hz"] == 125.0
    assert result["frequency_stop_hz"] == 175.0


def test_abstained_result_has_no_physical_frequency_range() -> None:
    result = build_result(
        torch.tensor([0.1, 0.2, 0.5, 0.1, 0.1]),
        torch.tensor([0.0, 0.0, 2.0]),
        torch.tensor([0.25, 0.75]),
        0.6,
        (100.0, 200.0),
    )
    assert result["frequency_start_hz"] is None
    assert result["frequency_stop_hz"] is None


@pytest.mark.parametrize(
    ("start_hz", "stop_hz"),
    [(None, 100.0), (100.0, None), (-1.0, 100.0), (100.0, 100.0), (float("nan"), 100.0)],
)
def test_frequency_span_requires_a_valid_paired_range(start_hz, stop_hz) -> None:
    with pytest.raises(ValueError):
        validate_frequency_span(start_hz, stop_hz)


def test_frequency_span_is_optional() -> None:
    assert validate_frequency_span(None, None) is None


@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan"])
def test_confidence_threshold_rejects_invalid_values(value: str) -> None:
    with pytest.raises(Exception):
        confidence_threshold(value)
