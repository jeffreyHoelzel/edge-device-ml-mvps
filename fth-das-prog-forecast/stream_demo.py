"""A short synthetic streaming demo that prints a forecast for rolling DAS windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from generate_data import EventMetadata, SCENARIOS, TIME_STEP_SECONDS, generate_sample, project_future_location
from infer import forecast, load_model, resolve_horizon_seconds


def stream_forecasts(
    model: object, signal: np.ndarray, metadata: EventMetadata, updates: int
) -> Iterator[dict[str, object]]:
    """Yield forecasts whose current locations are aligned to each rolling window's final frame."""
    if updates <= 0:
        raise ValueError("updates must be positive")
    if signal.shape[0] < 20 + updates - 1:
        raise ValueError("signal does not contain enough frames for the requested updates")
    for offset in range(updates):
        window = signal[offset : offset + 20]
        current_location = project_future_location(
            metadata.current_location_m,
            metadata.direction,
            metadata.speed_m_per_min,
            (offset + 19) * TIME_STEP_SECONDS,
        )
        result = forecast(model, window, current_location, metadata.horizon_seconds)
        result["window_end_offset_seconds"] = (offset + 19) * TIME_STEP_SECONDS
        yield result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/das_risk_model.pt"))
    parser.add_argument("--scenario", choices=SCENARIOS, default="excavation_approaching")
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--horizon-seconds", type=int)
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error("--updates must be positive")

    model = load_model(args.model_path)
    try:
        horizon_seconds = resolve_horizon_seconds(model, args.horizon_seconds)
    except ValueError as error:
        parser.error(str(error))
    rng = np.random.default_rng(123)
    signal, metadata = generate_sample(
        rng,
        scenario=args.scenario,
        time_steps=20 + args.updates,
        horizon_seconds=horizon_seconds,
    )
    for result in stream_forecasts(model, signal, metadata, args.updates):
        print(json.dumps(result))


if __name__ == "__main__":
    main()
