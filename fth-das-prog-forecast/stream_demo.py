"""A short synthetic streaming demo that prints a forecast for rolling DAS windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_data import SCENARIOS, TIME_STEP_SECONDS, generate_sample, project_future_location
from infer import forecast, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/das_risk_model.pt"))
    parser.add_argument("--scenario", choices=SCENARIOS, default="excavation_approaching")
    parser.add_argument("--updates", type=int, default=5)
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error("--updates must be positive")

    rng = np.random.default_rng(123)
    signal, metadata = generate_sample(rng, scenario=args.scenario, time_steps=20 + args.updates)
    model = load_model(args.model_path)
    for offset in range(args.updates):
        window = signal[offset : offset + 20]
        current_location = project_future_location(
            metadata.current_location_m,
            metadata.direction,
            metadata.speed_m_per_min,
            (offset + 19) * TIME_STEP_SECONDS,
        )
        print(json.dumps(forecast(model, window, current_location, metadata.horizon_seconds)))


if __name__ == "__main__":
    main()
