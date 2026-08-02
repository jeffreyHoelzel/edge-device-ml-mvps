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


def forecast_event(probability: float, cause_index: int, severity_index: int, horizon_seconds: int = 60) -> dict[str, object]:
    """Format one stable, JSON-serializable forecast event."""
    return {
        "event_type": "sla_violation_forecast",
        "probability": round(float(probability), 4),
        "forecast_horizon_seconds": int(horizon_seconds),
        "likely_cause": CAUSE_NAMES[cause_index],
        "severity": SEVERITY_NAMES[severity_index],
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
        print(json.dumps(forecast_event(probability, cause, severity, contract.forecast_horizon_seconds), sort_keys=True))


if __name__ == "__main__":
    main()
