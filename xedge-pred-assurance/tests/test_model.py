from dataclasses import replace
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
        feature_names=("a", "b", "c", "d", "e", "f", "g"),
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


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"feature_names": ("a",) * 6}, "feature names"),
        ({"feature_std": (1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0)}, "standard deviations"),
        ({"cause_names": ("a", "b", "c")}, "cause labels"),
        ({"sample_interval_seconds": 0}, "sample_interval_seconds"),
    ),
)
def test_save_model_rejects_invalid_contract(tmp_path: Path, change: dict[str, object], message: str) -> None:
    model = PredictiveAssuranceModel()
    contract = TrainingContract(
        ("a", "b", "c", "d", "e", "f", "g"), (0.0,) * 7, (1.0,) * 7,
        ("a", "b", "c", "d"), ("a", "b", "c"), 30, 5, 60,
    )

    with pytest.raises(ValueError, match=message):
        save_model(model, tmp_path / "model.pt", replace(contract, **change))


def test_load_model_rejects_invalid_persisted_contract(tmp_path: Path) -> None:
    model = PredictiveAssuranceModel()
    contract = TrainingContract(
        ("a", "b", "c", "d", "e", "f", "g"), (0.0,) * 7, (1.0,) * 7,
        ("a", "b", "c", "d"), ("a", "b", "c"), 30, 5, 60,
    )
    path = tmp_path / "model.pt"
    save_model(model, path, contract)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["contract"]["cause_names"] = ["a", "b", "c"]
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="invalid predictive-assurance checkpoint contract"):
        load_model(path)


def test_legacy_checkpoint_requires_retraining(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"config": {}, "state_dict": {}}, path)

    with pytest.raises(ValueError, match="retrain"):
        load_model(path)
