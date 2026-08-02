"""Validation and loading for the synthetic flow dataset format."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from generate_data import CAUSES, FEATURE_NAMES, SEVERITIES

REQUIRED_FIELDS = (
    "features", "baselines", "causes", "severity", "feature_names", "cause_names", "severity_names",
)


def validate_dataset(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return a validated copy-compatible dataset or raise a descriptive error."""
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        raise ValueError(f"dataset is missing required fields: {', '.join(missing)}")
    result = {field: np.asarray(data[field]) for field in REQUIRED_FIELDS}
    features, baselines = result["features"], result["baselines"]
    samples = len(features)
    if features.ndim != 3 or features.shape[0] < len(CAUSES) or features.shape[1] < 3 or features.shape[2] != len(FEATURE_NAMES):
        raise ValueError(f"features must have shape (samples >= {len(CAUSES)}, windows >= 3, {len(FEATURE_NAMES)})")
    if baselines.shape != (samples, len(FEATURE_NAMES)):
        raise ValueError(f"baselines must have shape ({samples}, {len(FEATURE_NAMES)})")
    if result["causes"].shape != (samples,) or result["severity"].shape != (samples,):
        raise ValueError("causes and severity must each have one value per sample")
    if not np.isfinite(features).all() or not np.isfinite(baselines).all():
        raise ValueError("features and baselines must contain only finite values")
    if tuple(result["feature_names"].tolist()) != FEATURE_NAMES:
        raise ValueError("dataset feature_names do not match the supported feature order")
    if tuple(result["cause_names"].tolist()) != CAUSES or tuple(result["severity_names"].tolist()) != SEVERITIES:
        raise ValueError("dataset label names do not match the supported taxonomy")
    if not np.issubdtype(result["causes"].dtype, np.integer) or not np.all((0 <= result["causes"]) & (result["causes"] < len(CAUSES))):
        raise ValueError("causes must be integer values in the supported range")
    if not np.issubdtype(result["severity"].dtype, np.integer) or not np.all((0 <= result["severity"]) & (result["severity"] < len(SEVERITIES))):
        raise ValueError("severity must be integer values in the supported range")
    return result


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    """Load and validate an NPZ dataset without retaining an open file handle."""
    with np.load(path, allow_pickle=False) as data:
        return validate_dataset(data)
