import torch
import pytest

from contract import CHECKPOINT_FORMAT_VERSION, CLASS_NAMES, IMPACT_NAMES, INPUT_SHAPE, NORMALIZATION_NAME
from infer import load_model, validate_checkpoint
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
