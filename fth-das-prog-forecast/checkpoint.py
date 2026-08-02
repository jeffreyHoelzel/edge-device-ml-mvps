"""Versioned checkpoint creation and validation for the DAS MVP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from generate_data import (
    CLASS_LABELS,
    DEFAULT_HORIZON_SECONDS,
    DIRECTION_LABELS,
    FIBER_LENGTH_M,
    PROTECTED_ZONE_CENTER_M,
    PROTECTED_ZONE_HALF_WIDTH_M,
    TIME_STEP_SECONDS,
)

if TYPE_CHECKING:
    from model import DASRiskModel


CHECKPOINT_FORMAT_VERSION = 2
RISK_TIER_POLICY = {"medium_threshold": 0.35, "high_threshold": 0.67}


def create_checkpoint(model: DASRiskModel, seed: int, horizon_seconds: int = DEFAULT_HORIZON_SECONDS) -> dict[str, object]:
    """Create a portable, self-describing CPU checkpoint."""
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state": model.state_dict(),
        "model_config": {"spatial_channels": 16, "hidden_size": 32},
        "metadata": {
            "class_labels": list(CLASS_LABELS),
            "direction_labels": list(DIRECTION_LABELS),
            "fiber_length_m": FIBER_LENGTH_M,
            "protected_zone_center_m": PROTECTED_ZONE_CENTER_M,
            "protected_zone_half_width_m": PROTECTED_ZONE_HALF_WIDTH_M,
            "time_step_seconds": TIME_STEP_SECONDS,
            "horizon_seconds": horizon_seconds,
            "seed": seed,
            "risk_tier_policy": RISK_TIER_POLICY,
        },
    }


def validate_checkpoint(checkpoint: object) -> dict[str, object]:
    """Validate checkpoint compatibility before model construction."""
    if not isinstance(checkpoint, dict):
        raise ValueError("invalid DAS checkpoint; retrain with the current train.py")
    if checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "unsupported DAS checkpoint format; retrain with the current train.py "
            f"to produce format {CHECKPOINT_FORMAT_VERSION}"
        )
    model_config = checkpoint.get("model_config")
    model_state = checkpoint.get("model_state")
    metadata = checkpoint.get("metadata")
    if not isinstance(model_config, dict) or not isinstance(model_state, dict) or not isinstance(metadata, dict):
        raise ValueError("invalid DAS checkpoint contents; retrain with the current train.py")
    expected_metadata = {
        "class_labels": list(CLASS_LABELS),
        "direction_labels": list(DIRECTION_LABELS),
        "fiber_length_m": FIBER_LENGTH_M,
        "protected_zone_center_m": PROTECTED_ZONE_CENTER_M,
        "protected_zone_half_width_m": PROTECTED_ZONE_HALF_WIDTH_M,
        "time_step_seconds": TIME_STEP_SECONDS,
        "risk_tier_policy": RISK_TIER_POLICY,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"incompatible DAS checkpoint metadata: {key}; retrain the model")
    if not isinstance(metadata.get("horizon_seconds"), int) or metadata["horizon_seconds"] <= 0:
        raise ValueError("invalid DAS checkpoint metadata: horizon_seconds")
    if not isinstance(metadata.get("seed"), int):
        raise ValueError("invalid DAS checkpoint metadata: seed")
    return checkpoint
