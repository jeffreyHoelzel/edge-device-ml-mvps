"""Small CPU-friendly CNN used by the RF interference MVP."""

from __future__ import annotations

import torch
from torch import nn

from contract import CLASS_NAMES, IMPACT_NAMES

NUM_CLASSES = len(CLASS_NAMES)
NUM_IMPACT_LEVELS = len(IMPACT_NAMES)


class RFInterferenceCNN(nn.Module):
    """Predict RF class, normalized frequency bounds, and service impact."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.shared = nn.Sequential(nn.Flatten(), nn.Linear(32, 48), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(48, num_classes)
        self.bounds = nn.Linear(48, 2)
        self.impact = nn.Linear(48, NUM_IMPACT_LEVELS)

    def forward(self, spectrogram: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return logits plus ordered, [0, 1]-normalized frequency bounds."""
        embedding = self.shared(self.features(spectrogram))
        raw_bounds = torch.sigmoid(self.bounds(embedding))
        frequency_bounds, _ = torch.sort(raw_bounds, dim=1)
        return {
            "class_logits": self.classifier(embedding),
            "frequency_bounds": frequency_bounds,
            "impact_logits": self.impact(embedding),
        }
