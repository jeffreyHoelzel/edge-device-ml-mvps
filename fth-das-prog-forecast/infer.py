"""Load a saved model and emit a concise risk forecast as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from generate_data import (
    CLASS_LABELS,
    DEFAULT_HORIZON_SECONDS,
    DIRECTION_LABELS,
    FIBER_LENGTH_M,
    SCENARIOS,
    generate_sample,
)
from model import DASRiskModel


def load_model(model_path: Path) -> DASRiskModel:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = DASRiskModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def forecast(model: DASRiskModel, signal: np.ndarray, current_location_m: float, horizon_seconds: int) -> dict[str, object]:
    with torch.inference_mode():
        outputs = model(torch.from_numpy(signal).unsqueeze(0))
    event_type = CLASS_LABELS[int(outputs["event_logits"].argmax(dim=1).item())]
    direction = DIRECTION_LABELS[int(outputs["direction_logits"].argmax(dim=1).item())]
    probability = float(outputs["escalation_probability"].item())
    if probability >= 0.67:
        risk = "high"
    elif probability >= 0.35:
        risk = "medium"
    else:
        risk = "low"
    trajectory = {
        "toward_asset": "approaching_protected_asset",
        "away_from_asset": "moving_away_from_protected_asset",
        "stationary": "stationary",
    }[direction]
    return {
        "event_type": event_type,
        "current_location_m": round(current_location_m),
        "trajectory": trajectory,
        "estimated_speed_m_per_min": round(float(outputs["speed_m_per_min"].item()), 2),
        "predicted_future_location_m": round(float(outputs["future_location"].item()) * FIBER_LENGTH_M, 1),
        "risk_horizon_seconds": horizon_seconds,
        "escalation_probability": round(probability, 3),
        "predicted_risk": risk,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/das_risk_model.pt"))
    parser.add_argument("--scenario", choices=SCENARIOS, default="excavation_approaching")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    signal, metadata = generate_sample(np.random.default_rng(args.seed), scenario=args.scenario)
    result = forecast(load_model(args.model_path), signal, metadata.current_location_m, metadata.horizon_seconds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
