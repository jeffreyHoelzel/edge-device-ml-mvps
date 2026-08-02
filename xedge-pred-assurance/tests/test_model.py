from pathlib import Path

import pytest
import torch

from model import ModelConfig, PredictiveAssuranceModel, TrainingContract, load_model, save_model


def test_model_output_shapes() -> None:
    model = PredictiveAssuranceModel(ModelConfig(conv_channels=8, gru_hidden=10))
    output = model(torch.randn(4, 30, 7))

    assert set(output) == {"incident_logits", "cause_logits", "severity_logits"}
    assert output["incident_logits"].shape == (4,)
    assert output["cause_logits"].shape == (4, 4)
    assert output["severity_logits"].shape == (4, 3)


def test_model_checkpoint_round_trip_preserves_contract(tmp_path: Path) -> None:
    model = PredictiveAssuranceModel(ModelConfig(conv_channels=8, gru_hidden=10))
    contract = TrainingContract(
        feature_names=("a",) * 7,
        feature_mean=(0.0,) * 7,
        feature_std=(1.0,) * 7,
        cause_names=("a", "b", "c", "d"),
        severity_names=("a", "b", "c"),
        sequence_length=30,
        sample_interval_seconds=5,
        forecast_horizon_seconds=60,
    )
    path = tmp_path / "model.pt"

    save_model(model, path, contract)
    loaded = load_model(path)

    assert loaded.contract == contract
    assert loaded.model.config == model.config


def test_legacy_checkpoint_requires_retraining(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"config": {}, "state_dict": {}}, path)

    with pytest.raises(ValueError, match="retrain"):
        load_model(path)
