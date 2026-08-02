"""Simulate KPI arrivals and print SLA-violation forecast JSON events."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from generate_data import CAUSE_NAMES, SEVERITY_NAMES, generate_sequence
from model import load_model


def forecast_event(
    probability: float,
    cause_index: int,
    severity_index: int,
    horizon_seconds: int = 60,
    incident_threshold: float = 0.70,
    cause_confidence: float = 1.0,
    severity_confidence: float = 1.0,
    model_version: str = "predictive-assurance-v3",
) -> dict[str, object]:
    """Format one stable, JSON-serializable forecast event."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if not 0 <= cause_index < len(CAUSE_NAMES):
        raise ValueError("cause_index is out of range")
    if not 0 <= severity_index < len(SEVERITY_NAMES):
        raise ValueError("severity_index is out of range")
    if horizon_seconds < 1:
        raise ValueError("horizon_seconds must be positive")
    if not 0.0 < incident_threshold < 1.0:
        raise ValueError("incident_threshold must be between 0 and 1")
    if not 0.0 <= cause_confidence <= 1.0 or not 0.0 <= severity_confidence <= 1.0:
        raise ValueError("diagnosis confidences must be between 0 and 1")
    if not model_version:
        raise ValueError("model_version must not be empty")
    return {
        "event_type": "sla_violation_forecast",
        "probability": round(float(probability), 4),
        "forecast_horizon_seconds": int(horizon_seconds),
        "likely_cause": CAUSE_NAMES[cause_index],
        "severity": SEVERITY_NAMES[severity_index],
        "incident_threshold": round(float(incident_threshold), 4),
        "meets_alert_threshold": bool(probability >= incident_threshold),
        "cause_confidence": round(float(cause_confidence), 4),
        "severity_confidence": round(float(severity_confidence), 4),
        "model_version": model_version,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.pt"))
    parser.add_argument("--cause", choices=CAUSE_NAMES, default="rf_interference")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"{args.model} was not found; run train.py first")

    loaded = load_model(args.model)
    model = loaded.model
    contract = loaded.contract
    if contract.cause_names != CAUSE_NAMES:
        raise ValueError("checkpoint cause mapping does not match the streaming demo")
    raw_measurements = generate_sequence(
        contract.sequence_length + 12,
        CAUSE_NAMES.index(args.cause),
        severity=1,
        rng=np.random.default_rng(args.seed),
    )
    buffer: deque[np.ndarray] = deque(maxlen=contract.sequence_length)
    for measurement in raw_measurements:
        buffer.append(measurement)
        if len(buffer) < contract.sequence_length:
            continue
        raw_window = np.asarray(buffer, dtype=np.float32)
        window = ((raw_window - contract.feature_mean) / contract.feature_std).astype(np.float32)[None, ...]
        with torch.inference_mode():
            output = model(torch.from_numpy(window))
            probability = torch.sigmoid(output["incident_logits"])[0].item()
            cause = int(output["cause_logits"].argmax(dim=1)[0])
            severity = int(output["severity_logits"].argmax(dim=1)[0])
            cause_confidence = torch.softmax(output["cause_logits"], dim=1).max(dim=1).values[0].item()
            severity_confidence = torch.softmax(output["severity_logits"], dim=1).max(dim=1).values[0].item()
        print(
            json.dumps(
                forecast_event(
                    probability,
                    cause,
                    severity,
                    contract.forecast_horizon_seconds,
                    contract.incident_threshold,
                    cause_confidence,
                    severity_confidence,
                    contract.model_version,
                ),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
