import pytest
import torch

from checkpoint import CHECKPOINT_FORMAT_VERSION, create_checkpoint, validate_checkpoint
from infer import load_model
from model import DASRiskModel


def test_checkpoint_round_trip_loads_model(tmp_path) -> None:
    path = tmp_path / "model.pt"
    torch.save(create_checkpoint(DASRiskModel(), seed=9, horizon_seconds=180), path)
    assert isinstance(load_model(path), DASRiskModel)


def test_legacy_checkpoint_is_rejected_with_retraining_guidance(tmp_path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"model_state": {}, "model_config": {}}, path)
    with pytest.raises(ValueError, match="retrain"):
        load_model(path)


def test_checkpoint_metadata_mismatch_is_rejected() -> None:
    checkpoint = create_checkpoint(DASRiskModel(), seed=9)
    metadata = checkpoint["metadata"]
    assert isinstance(metadata, dict)
    metadata["fiber_length_m"] = 1.0
    with pytest.raises(ValueError, match="fiber_length_m"):
        validate_checkpoint(checkpoint)


def test_checkpoint_format_is_explicit() -> None:
    checkpoint = create_checkpoint(DASRiskModel(), seed=9)
    assert checkpoint["checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION
