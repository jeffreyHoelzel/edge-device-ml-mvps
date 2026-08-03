"""Validation and loading for the synthetic flow dataset format."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from generate_data import CAUSES, FEATURE_NAMES, SEVERITIES

REQUIRED_FIELDS = (
    "features", "baselines", "causes", "severity", "incident", "feature_names", "cause_names", "severity_names",
    "flow_ids", "device_ids", "interfaces", "window_ends",
)


def validate_dataset(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return a validated copy-compatible dataset or raise a descriptive error."""
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        raise ValueError(f"dataset is missing required fields: {', '.join(missing)}")
    result = {field: np.asarray(data[field]) for field in REQUIRED_FIELDS}
    features, baselines = result["features"], result["baselines"]
    samples = len(features)
    if features.ndim != 3 or features.shape[0] < len(CAUSES) + 1 or features.shape[1] < 3 or features.shape[2] != len(FEATURE_NAMES):
        raise ValueError(f"features must have shape (samples >= {len(CAUSES) + 1}, windows >= 3, {len(FEATURE_NAMES)})")
    if baselines.shape != (samples, len(FEATURE_NAMES)):
        raise ValueError(f"baselines must have shape ({samples}, {len(FEATURE_NAMES)})")
    if any(result[field].shape != (samples,) for field in ("causes", "severity", "incident")):
        raise ValueError("causes, severity, and incident must each have one value per sample")
    if any(result[field].shape != (samples,) for field in ("flow_ids", "device_ids", "interfaces", "window_ends")):
        raise ValueError("context fields must each have one value per sample")
    if not np.issubdtype(features.dtype, np.number) or not np.issubdtype(baselines.dtype, np.number):
        raise ValueError("features and baselines must contain numeric values")
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
    if not np.issubdtype(result["incident"].dtype, np.integer) or not np.all((0 <= result["incident"]) & (result["incident"] <= 1)):
        raise ValueError("incident must contain only zero or one")
    return result


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    """Load and validate an NPZ dataset without retaining an open file handle."""
    with np.load(path, allow_pickle=False) as data:
        return validate_dataset(data)
