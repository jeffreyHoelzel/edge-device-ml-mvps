"""Shared data and inference contract for the RF interference MVP."""

from __future__ import annotations

import torch

CLASS_NAMES = (
    "no_interference",
    "narrowband_continuous",
    "wideband_intermittent",
    "periodic_impulsive",
    "adjacent_channel_leakage",
)
IMPACT_NAMES = ("low", "moderate", "high")
TIME_BINS = 64
FREQUENCY_BINS = 128
INPUT_SHAPE = (1, TIME_BINS, FREQUENCY_BINS)
NORMALIZATION_NAME = "per_window_zscore"
CHECKPOINT_FORMAT_VERSION = 1


def normalize_per_window(spectrograms: torch.Tensor) -> torch.Tensor:
    """Normalize each spectrogram independently for training and inference."""
    mean = spectrograms.mean(dim=(2, 3), keepdim=True)
    std = spectrograms.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
    return (spectrograms - mean) / std
