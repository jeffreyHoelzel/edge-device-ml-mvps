"""Create reproducible, synthetic rolling flow summaries for the MVP."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FEATURE_NAMES = (
    "network_rtt",
    "server_response_time",
    "dns_time",
    "tcp_connection_time",
    "retransmission_rate",
    "packet_loss",
    "byte_volume",
    "flow_count",
    "interface_utilization",
)
CAUSES = ("server_delay", "wan_congestion", "packet_loss", "dns_delay", "client_side_delay")
SEVERITIES = ("minor", "major", "critical")

# Typical baseline measurements. Time values are milliseconds; rates/utilization are fractions.
BASELINE_MEAN = np.array([35.0, 80.0, 15.0, 25.0, 0.005, 0.001, 2_000_000.0, 45.0, 0.35])
BASELINE_STD = np.array([6.0, 15.0, 3.0, 5.0, 0.002, 0.0005, 500_000.0, 10.0, 0.08])


def _apply_incident(sequence: np.ndarray, cause: int, intensity: float) -> None:
    """Modify the last third of a rolling sequence according to one cause profile."""
    incident = sequence[len(sequence) * 2 // 3 :]
    if cause == 0:  # server_delay
        incident[:, 1] *= 1.0 + 2.5 * intensity
    elif cause == 1:  # wan_congestion
        incident[:, 0] *= 1.0 + 1.8 * intensity
        incident[:, 4] += 0.012 * intensity
        incident[:, 6] *= 1.0 + 0.7 * intensity
        incident[:, 7] *= 1.0 + 0.5 * intensity
        incident[:, 8] = np.minimum(0.99, incident[:, 8] + 0.45 * intensity)
    elif cause == 2:  # packet_loss
        incident[:, 0] *= 1.0 + 1.1 * intensity
        incident[:, 4] += 0.035 * intensity
        incident[:, 5] += 0.012 * intensity
    elif cause == 3:  # dns_delay
        incident[:, 2] *= 1.0 + 4.0 * intensity
    elif cause == 4:  # client_side_delay
        incident[:, 3] *= 1.0 + 3.0 * intensity


def generate_dataset(samples: int = 2_000, sequence_length: int = 12, seed: int = 7) -> dict[str, np.ndarray]:
    """Return labelled incident sequences and each sequence's pre-incident baseline."""
    if samples < len(CAUSES):
        raise ValueError(f"samples must be at least {len(CAUSES)}")
    rng = np.random.default_rng(seed)
    features = np.empty((samples, sequence_length, len(FEATURE_NAMES)), dtype=np.float32)
    baselines = np.empty((samples, len(FEATURE_NAMES)), dtype=np.float32)
    causes = np.arange(samples, dtype=np.int64) % len(CAUSES)
    rng.shuffle(causes)
    severity = rng.integers(0, len(SEVERITIES), size=samples, dtype=np.int64)

    for index in range(samples):
        baseline = BASELINE_MEAN * rng.normal(1.0, 0.12, len(FEATURE_NAMES))
        baseline = np.maximum(baseline, 1e-5)
        noise = rng.normal(0.0, BASELINE_STD * 0.30, (sequence_length, len(FEATURE_NAMES)))
        sequence = baseline + noise
        intensity = (0.45, 0.75, 1.05)[severity[index]]
        _apply_incident(sequence, int(causes[index]), intensity)
        # Rates, counts, and volume cannot be negative.
        sequence = np.maximum(sequence, 1e-6)
        features[index] = sequence
        baselines[index] = baseline

    return {
        "features": features,
        "baselines": baselines,
        "causes": causes,
        "severity": severity,
        "feature_names": np.asarray(FEATURE_NAMES),
        "cause_names": np.asarray(CAUSES),
        "severity_names": np.asarray(SEVERITIES),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_flows.npz"))
    parser.add_argument("--samples", type=int, default=2_000)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **generate_dataset(args.samples, args.sequence_length, args.seed))
    print(f"Wrote {args.samples} labelled flow sequences to {args.output}")


if __name__ == "__main__":
    main()
