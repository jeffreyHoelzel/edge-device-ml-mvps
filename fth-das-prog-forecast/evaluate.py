"""Deterministic held-out evaluation for the synthetic DAS risk model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from generate_data import (
    DEFAULT_HORIZON_SECONDS,
    FIBER_LENGTH_M,
    MAX_SPEED_M_PER_MIN,
    PROTECTED_ZONE_CENTER_M,
    SyntheticDASDataset,
)
from infer import load_model
from model import DASRiskModel


QUALITY_GATES = {
    "event_accuracy": 0.65,
    "direction_accuracy": 0.50,
    "speed_mae_m_per_min": 3.0,
    "future_location_mae_m": 60.0,
    "risk_mae": 0.20,
    "risk_brier_score": 0.08,
}


def _derived_future_locations(outputs: dict[str, torch.Tensor], current_locations: torch.Tensor, horizon_seconds: int) -> torch.Tensor:
    directions = outputs["direction_logits"].argmax(dim=1)
    toward_sign = torch.where(current_locations < PROTECTED_ZONE_CENTER_M, 1.0, -1.0)
    direction_sign = torch.where(
        directions == 0,
        0.0,
        torch.where(directions == 1, toward_sign, -toward_sign),
    )
    speed = outputs["speed_m_per_min"]
    return torch.clamp(current_locations + direction_sign * speed * horizon_seconds / 60.0, 0.0, FIBER_LENGTH_M)


def evaluate_model(model: DASRiskModel, dataset: SyntheticDASDataset, batch_size: int = 32) -> dict[str, object]:
    """Return stable held-out metrics and five calibration bins."""
    loader = DataLoader(dataset, batch_size=batch_size)
    totals = {
        "event_correct": 0,
        "direction_correct": 0,
        "speed_absolute_error": 0.0,
        "future_location_absolute_error": 0.0,
        "risk_absolute_error": 0.0,
        "risk_squared_error": 0.0,
        "count": 0,
    }
    bins = [{"count": 0, "predicted_sum": 0.0, "observed_sum": 0.0} for _ in range(5)]
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            outputs = model(batch["signal"])
            probabilities = outputs["escalation_probability"]
            future_locations = _derived_future_locations(outputs, batch["window_end_location"], dataset.samples[0][1].horizon_seconds)
            count = batch["signal"].shape[0]
            totals["count"] += count
            totals["event_correct"] += int((outputs["event_logits"].argmax(dim=1) == batch["class_index"]).sum().item())
            totals["direction_correct"] += int((outputs["direction_logits"].argmax(dim=1) == batch["direction_index"]).sum().item())
            totals["speed_absolute_error"] += float(
                (outputs["speed_m_per_min"] - batch["speed"] * MAX_SPEED_M_PER_MIN).abs().sum().item()
            )
            totals["future_location_absolute_error"] += float((future_locations - batch["future_location"]).abs().sum().item())
            totals["risk_absolute_error"] += float((probabilities - batch["escalation"]).abs().sum().item())
            totals["risk_squared_error"] += float(((probabilities - batch["escalation"]) ** 2).sum().item())
            for probability, observed in zip(probabilities.tolist(), batch["escalation"].tolist(), strict=True):
                bucket = min(int(probability * 5), 4)
                bins[bucket]["count"] += 1
                bins[bucket]["predicted_sum"] += probability
                bins[bucket]["observed_sum"] += observed
    count = totals["count"]
    calibration = []
    for index, bucket in enumerate(bins):
        bin_count = bucket["count"]
        calibration.append(
            {
                "lower_bound": index / 5,
                "upper_bound": (index + 1) / 5,
                "count": bin_count,
                "mean_predicted_probability": round(bucket["predicted_sum"] / bin_count, 6) if bin_count else None,
                "mean_observed_probability": round(bucket["observed_sum"] / bin_count, 6) if bin_count else None,
            }
        )
    return {
        "samples": count,
        "event_accuracy": round(totals["event_correct"] / count, 6),
        "direction_accuracy": round(totals["direction_correct"] / count, 6),
        "speed_mae_m_per_min": round(totals["speed_absolute_error"] / count, 6),
        "future_location_mae_m": round(totals["future_location_absolute_error"] / count, 6),
        "risk_mae": round(totals["risk_absolute_error"] / count, 6),
        "risk_brier_score": round(totals["risk_squared_error"] / count, 6),
        "calibration_bins": calibration,
    }


def quality_gate_failures(metrics: dict[str, object]) -> list[str]:
    failures = []
    for name, threshold in QUALITY_GATES.items():
        value = metrics[name]
        assert isinstance(value, float)
        if name.endswith("accuracy"):
            if value < threshold:
                failures.append(f"{name}={value:.3f} is below {threshold:.3f}")
        elif value > threshold:
            failures.append(f"{name}={value:.3f} exceeds {threshold:.3f}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/das_risk_model.pt"))
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--horizon-seconds", type=int, default=DEFAULT_HORIZON_SECONDS)
    parser.add_argument("--enforce-quality", action="store_true")
    args = parser.parse_args()
    if args.samples <= 0 or args.batch_size <= 0 or args.horizon_seconds <= 0:
        parser.error("--samples, --batch-size, and --horizon-seconds must be positive")
    metrics = evaluate_model(
        load_model(args.model_path),
        SyntheticDASDataset(args.samples, seed=args.seed, horizon_seconds=args.horizon_seconds),
        args.batch_size,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    failures = quality_gate_failures(metrics)
    if args.enforce_quality and failures:
        parser.error("quality gates failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
