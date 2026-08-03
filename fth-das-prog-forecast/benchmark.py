"""Measure CPU inference characteristics for the DAS MVP."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from generate_data import SCENARIOS, generate_sample
from infer import load_model
from model import DASRiskModel


def _process_max_rss_mb() -> float:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return max_rss / (1024 * 1024)
    return max_rss / 1024


def benchmark_model(model: DASRiskModel, signal: np.ndarray, warmup_runs: int, measured_runs: int) -> dict[str, object]:
    """Benchmark model-forward latency on one CPU window."""
    if warmup_runs < 0:
        raise ValueError("warmup_runs must not be negative")
    if measured_runs <= 0:
        raise ValueError("measured_runs must be positive")
    tensor = torch.from_numpy(signal).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(tensor)
        latencies_ms = []
        for _ in range(measured_runs):
            started = time.perf_counter()
            model(tensor)
            latencies_ms.append((time.perf_counter() - started) * 1_000)
    ordered = sorted(latencies_ms)
    percentile_index = max(0, int(np.ceil(0.95 * len(ordered))) - 1)
    return {
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "input_shape": list(tensor.shape),
        "input_size_bytes": tensor.numel() * tensor.element_size(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "latency_ms": {
            "minimum": round(ordered[0], 6),
            "mean": round(float(np.mean(latencies_ms)), 6),
            "p50": round(float(np.percentile(latencies_ms, 50)), 6),
            "p95": round(ordered[percentile_index], 6),
            "maximum": round(ordered[-1], 6),
        },
        "process_max_rss_mb": round(_process_max_rss_mb(), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/das_risk_model.pt"))
    parser.add_argument("--scenario", choices=SCENARIOS, default="excavation_approaching")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--measured-runs", type=int, default=50)
    args = parser.parse_args()
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must not be negative")
    if args.measured_runs <= 0:
        parser.error("--measured-runs must be positive")
    torch.set_num_threads(1)
    signal, _ = generate_sample(np.random.default_rng(args.seed), scenario=args.scenario)
    results = benchmark_model(load_model(args.model_path), signal, args.warmup_runs, args.measured_runs)
    results["model_path"] = os.fspath(args.model_path)
    results["checkpoint_size_bytes"] = args.model_path.stat().st_size
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
