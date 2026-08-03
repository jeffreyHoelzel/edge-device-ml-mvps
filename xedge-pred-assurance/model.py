"""Temporal convolution + GRU multi-task model for KPI forecast windows."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


CHECKPOINT_FORMAT_VERSION = 3


@dataclass(frozen=True)
class ModelConfig:
    input_features: int = 7
    conv_channels: int = 24
    gru_hidden: int = 32
    cause_classes: int = 4
    severity_classes: int = 3


@dataclass(frozen=True)
class TrainingContract:
    """Preprocessing and forecast semantics required to use a checkpoint."""

    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    cause_names: tuple[str, ...]
    severity_names: tuple[str, ...]
    sequence_length: int
    sample_interval_seconds: int
    forecast_horizon_seconds: int
    incident_threshold: float = 0.70
    model_version: str = "predictive-assurance-v3"


@dataclass(frozen=True)
class LoadedModel:
    model: "PredictiveAssuranceModel"
    contract: TrainingContract
    evaluation_metrics: dict[str, float]


def validate_training_contract(contract: TrainingContract, config: ModelConfig) -> None:
    """Reject checkpoint metadata that cannot safely drive model inference."""
    for name, value in (
        ("input_features", config.input_features),
        ("conv_channels", config.conv_channels),
        ("gru_hidden", config.gru_hidden),
        ("cause_classes", config.cause_classes),
        ("severity_classes", config.severity_classes),
        ("sequence_length", contract.sequence_length),
        ("sample_interval_seconds", contract.sample_interval_seconds),
        ("forecast_horizon_seconds", contract.forecast_horizon_seconds),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"checkpoint {name} must be a positive integer")
    if len(contract.feature_names) != config.input_features:
        raise ValueError("checkpoint feature names do not match model input features")
    if any(not isinstance(name, str) or not name for name in contract.feature_names):
        raise ValueError("checkpoint feature names must be non-empty strings")
    if len(set(contract.feature_names)) != len(contract.feature_names):
        raise ValueError("checkpoint feature names must be unique")
    if len(contract.feature_mean) != config.input_features or len(contract.feature_std) != config.input_features:
        raise ValueError("checkpoint feature scaling does not match model input features")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in contract.feature_mean):
        raise ValueError("checkpoint feature means must be finite numbers")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0 for value in contract.feature_std):
        raise ValueError("checkpoint feature standard deviations must be positive finite numbers")
    for name, labels, classes in (
        ("cause", contract.cause_names, config.cause_classes),
        ("severity", contract.severity_names, config.severity_classes),
    ):
        if len(labels) != classes:
            raise ValueError(f"checkpoint {name} labels do not match model output classes")
        if any(not isinstance(label, str) or not label for label in labels) or len(set(labels)) != len(labels):
            raise ValueError(f"checkpoint {name} labels must be unique non-empty strings")
    if not isinstance(contract.incident_threshold, (int, float)) or isinstance(contract.incident_threshold, bool) or not math.isfinite(contract.incident_threshold) or not 0.0 <= contract.incident_threshold <= 1.0:
        raise ValueError("checkpoint incident threshold must be a finite value between 0 and 1")
    if not isinstance(contract.model_version, str) or not contract.model_version:
        raise ValueError("checkpoint model version must be a non-empty string")


class PredictiveAssuranceModel(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.temporal = nn.Sequential(
            nn.Conv1d(config.input_features, config.conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.conv_channels, config.conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gru = nn.GRU(config.conv_channels, config.gru_hidden, batch_first=True)
        self.incident_head = nn.Linear(config.gru_hidden, 1)
        self.cause_head = nn.Linear(config.gru_hidden, config.cause_classes)
        self.severity_head = nn.Linear(config.gru_hidden, config.severity_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.config.input_features:
            raise ValueError(f"expected [batch, time, {self.config.input_features}], received {tuple(x.shape)}")
        if x.shape[1] < 1:
            raise ValueError("expected at least one KPI measurement")
        if not torch.isfinite(x).all():
            raise ValueError("KPI values must be finite")
        encoded = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        _, hidden = self.gru(encoded)
        summary = hidden[-1]
        return {
            "incident_logits": self.incident_head(summary).squeeze(-1),
            "cause_logits": self.cause_head(summary),
            "severity_logits": self.severity_head(summary),
        }


def save_model(
    model: PredictiveAssuranceModel,
    path: Path,
    contract: TrainingContract,
    evaluation_metrics: dict[str, float] | None = None,
) -> None:
    validate_training_contract(contract, model.config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "config": asdict(model.config),
            "contract": asdict(contract),
            "evaluation_metrics": evaluation_metrics or {},
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_model(path: Path, device: str = "cpu") -> LoadedModel:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"{path} is not a predictive-assurance v{CHECKPOINT_FORMAT_VERSION} checkpoint; retrain the model"
        )
    try:
        config = ModelConfig(**checkpoint["config"])
        contract_values = checkpoint["contract"]
        contract = TrainingContract(
            feature_names=tuple(contract_values["feature_names"]),
            feature_mean=tuple(contract_values["feature_mean"]),
            feature_std=tuple(contract_values["feature_std"]),
            cause_names=tuple(contract_values["cause_names"]),
            severity_names=tuple(contract_values["severity_names"]),
            sequence_length=contract_values["sequence_length"],
            sample_interval_seconds=contract_values["sample_interval_seconds"],
            forecast_horizon_seconds=contract_values["forecast_horizon_seconds"],
            incident_threshold=contract_values["incident_threshold"],
            model_version=contract_values["model_version"],
        )
        validate_training_contract(contract, config)
        model = PredictiveAssuranceModel(config)
        model.load_state_dict(checkpoint["state_dict"])
    except (KeyError, TypeError, RuntimeError, ValueError) as error:
        raise ValueError(f"{path} has an invalid predictive-assurance checkpoint contract") from error
    model.to(device).eval()
    metrics = checkpoint.get("evaluation_metrics", {})
    if not isinstance(metrics, dict) or not all(isinstance(value, (int, float)) for value in metrics.values()):
        raise ValueError(f"{path} has invalid evaluation metrics")
    return LoadedModel(model=model, contract=contract, evaluation_metrics={key: float(value) for key, value in metrics.items()})
