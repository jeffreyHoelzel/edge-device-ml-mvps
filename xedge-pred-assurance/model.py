"""Temporal convolution + GRU multi-task model for KPI forecast windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    input_features: int = 7
    conv_channels: int = 24
    gru_hidden: int = 32
    cause_classes: int = 4
    severity_classes: int = 3


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
        encoded = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        _, hidden = self.gru(encoded)
        summary = hidden[-1]
        return {
            "incident_logits": self.incident_head(summary).squeeze(-1),
            "cause_logits": self.cause_head(summary),
            "severity_logits": self.severity_head(summary),
        }


def save_model(model: PredictiveAssuranceModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, path)


def load_model(path: Path, device: str = "cpu") -> PredictiveAssuranceModel:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = PredictiveAssuranceModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model
