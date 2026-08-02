"""A deliberately small spatial CNN + temporal GRU multi-task model."""

from __future__ import annotations

import torch
from torch import nn

from generate_data import CLASS_LABELS, DIRECTION_LABELS, MAX_SPEED_M_PER_MIN


class DASRiskModel(nn.Module):
    def __init__(self, spatial_channels: int = 16, hidden_size: int = 32) -> None:
        super().__init__()
        self.spatial_cnn = nn.Sequential(
            nn.Conv1d(1, spatial_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(spatial_channels, spatial_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.gru = nn.GRU(spatial_channels * 8, hidden_size, batch_first=True)
        self.event_head = nn.Linear(hidden_size, len(CLASS_LABELS))
        self.direction_head = nn.Linear(hidden_size, len(DIRECTION_LABELS))
        self.speed_head = nn.Linear(hidden_size, 1)
        self.location_head = nn.Linear(hidden_size, 1)
        self.escalation_head = nn.Linear(hidden_size, 1)

    def forward(self, signal: torch.Tensor) -> dict[str, torch.Tensor]:
        """Accept [batch, time, distance] normalized intensity data."""
        batch, steps, distance = signal.shape
        spatial = self.spatial_cnn(signal.reshape(batch * steps, 1, distance))
        sequence = spatial.flatten(1).reshape(batch, steps, -1)
        _, hidden = self.gru(sequence)
        features = hidden[-1]
        return {
            "event_logits": self.event_head(features),
            "direction_logits": self.direction_head(features),
            "speed_m_per_min": torch.sigmoid(self.speed_head(features)).squeeze(-1) * MAX_SPEED_M_PER_MIN,
            "future_location": torch.sigmoid(self.location_head(features)).squeeze(-1),
            "escalation_probability": torch.sigmoid(self.escalation_head(features)).squeeze(-1),
        }
