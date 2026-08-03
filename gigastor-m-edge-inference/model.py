"""Small, CPU-friendly temporal model and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from generate_data import CAUSES, FEATURE_NAMES, SEVERITIES


class RootCauseGRU(nn.Module):
    """Feature encoder + GRU with incident, severity, and root-cause heads."""

    def __init__(self, input_size: int, hidden_size: int = 48, cause_count: int = 5, severity_count: int = 3) -> None:
        super().__init__()
        self.config = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "cause_count": cause_count,
            "severity_count": severity_count,
        }
        self.encoder = nn.Sequential(nn.Linear(input_size, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.gru = nn.GRU(32, hidden_size, batch_first=True)
        self.cause_head = nn.Linear(hidden_size, cause_count)
        self.severity_head = nn.Linear(hidden_size, severity_count)
        self.incident_head = nn.Linear(hidden_size, 2)
        self.register_buffer("feature_mean", torch.zeros(input_size))
        self.register_buffer("feature_std", torch.ones(input_size))
        self.metadata: dict[str, Any] = {
            "schema_version": "1.1",
            "feature_names": list(FEATURE_NAMES),
            "cause_names": list(CAUSES),
            "severity_names": list(SEVERITIES),
        }

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean.copy_(mean.float())
        self.feature_std.copy_(std.float().clamp_min(1e-6))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = (features - self.feature_mean) / self.feature_std
        encoded = self.encoder(normalized)
        _, hidden = self.gru(encoded)
        representation = hidden[-1]
        return self.cause_head(representation), self.severity_head(representation), self.incident_head(representation)


def save_checkpoint(model: RootCauseGRU, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = dict(model.metadata)
    if metadata:
        checkpoint_metadata.update(metadata)
    torch.save({"checkpoint_version": 2, "config": model.config, "state_dict": model.state_dict(), "metadata": checkpoint_metadata}, target)


def load_checkpoint(path: str | Path) -> RootCauseGRU:
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_version") != 2 or "metadata" not in checkpoint:
        raise ValueError("checkpoint is incompatible; regenerate data and retrain the model")
    model = RootCauseGRU(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    metadata = checkpoint["metadata"]
    for field, expected in (("feature_names", model.config["input_size"]), ("cause_names", model.config["cause_count"]), ("severity_names", model.config["severity_count"])):
        if len(metadata.get(field, [])) != expected:
            raise ValueError(f"checkpoint metadata has an invalid {field} value")
    model.metadata = metadata
    model.eval()
    return model
