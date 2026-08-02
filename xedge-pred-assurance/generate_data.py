"""Create deterministic synthetic KPI windows for the assurance MVP."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FEATURE_NAMES = (
    "latency_ms",
    "jitter_ms",
    "packet_loss_pct",
    "uplink_mbps",
    "downlink_mbps",
    "signal_quality_pct",
    "retransmission_count",
)
CAUSE_NAMES = ("rf_interference", "congestion", "backhaul_degradation", "device_fault")
SEVERITY_NAMES = ("minor", "major", "critical")
SAMPLE_INTERVAL_SECONDS = 5
FORECAST_HORIZON_SECONDS = 60
FORECAST_STEPS = FORECAST_HORIZON_SECONDS // SAMPLE_INTERVAL_SECONDS

# Stable scaling makes training and streaming use exactly the same representation.
FEATURE_MEAN = np.array([35.0, 4.0, 0.1, 35.0, 80.0, 82.0, 1.0], dtype=np.float32)
FEATURE_STD = np.array([35.0, 8.0, 1.0, 20.0, 45.0, 18.0, 4.0], dtype=np.float32)


def normalize_features(values: np.ndarray) -> np.ndarray:
    """Normalize raw KPI measurements; accepts (..., 7) arrays."""
    return ((values - FEATURE_MEAN) / FEATURE_STD).astype(np.float32)


def _severity_strength(severity: int) -> float:
    return (0.48, 0.75, 1.0)[severity]


def _generate_trajectory(
    length: int,
    cause_id: int | None,
    severity: int,
    rng: np.random.Generator,
    degradation_start: int | None = None,
) -> np.ndarray:
    """Return one raw KPI trajectory with an optional degradation ramp."""
    time = np.arange(length, dtype=np.float32)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    daily_wobble = np.sin(time / 5.0 + phase)[:, None]
    baseline = np.array([35.0, 4.0, 0.1, 35.0, 80.0, 82.0, 1.0], dtype=np.float32)
    profile_offset = rng.normal(
        0.0,
        np.array([5.0, 1.2, 0.03, 5.0, 10.0, 5.0, 0.8], dtype=np.float32),
    ).astype(np.float32)
    noise = np.array([3.0, 0.8, 0.04, 2.0, 4.0, 2.0, 0.5], dtype=np.float32)
    values = baseline + profile_offset + daily_wobble * np.array(
        [1.5, 0.4, 0.01, 1.0, 1.5, 1.0, 0.1], dtype=np.float32
    )
    values += rng.normal(0.0, noise, size=(length, len(FEATURE_NAMES))).astype(np.float32)

    # Benign bursts make normal traffic less separable from incident windows.
    benign_bursts = rng.random(length) < 0.04
    values[benign_bursts, 0] += rng.uniform(5.0, 14.0, size=benign_bursts.sum())
    values[benign_bursts, 1] += rng.uniform(1.0, 4.0, size=benign_bursts.sum())
    values[benign_bursts, 6] += rng.uniform(1.0, 3.0, size=benign_bursts.sum())

    if cause_id is not None:
        start = degradation_start
        if start is None:
            start = int(rng.integers(max(2, length // 3), max(3, length * 2 // 3)))
        ramp = np.clip((time - start) / max(1, length - start - 1), 0.0, 1.0)
        strength = _severity_strength(severity)
        effects = (
            np.array([42, 13, 2.2, -18, -38, -40, 8], dtype=np.float32),  # RF
            np.array([85, 24, 1.2, -23, -52, -2, 7], dtype=np.float32),   # congestion
            np.array([110, 18, 1.7, -28, -58, 0, 5], dtype=np.float32),   # backhaul
            np.array([55, 35, 2.8, -30, -45, -10, 15], dtype=np.float32), # device fault
        )[cause_id]
        values += ramp[:, None] * strength * effects
        if cause_id == 3:
            # Device faults are characteristically bursty as well as degraded.
            spikes = rng.random(length) < (0.10 + 0.12 * ramp)
            values[spikes, 1] += 18.0 * strength
            values[spikes, 6] += 10.0 * strength

    values[:, 0:3] = np.maximum(values[:, 0:3], 0.0)
    values[:, 3:5] = np.maximum(values[:, 3:5], 0.2)
    values[:, 5] = np.clip(values[:, 5], 0.0, 100.0)
    values[:, 6] = np.maximum(values[:, 6], 0.0)
    return values.astype(np.float32)


def generate_sequence(
    sequence_length: int,
    cause_id: int | None,
    severity: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return one raw KPI window, optionally ending in a degradation pattern."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if cause_id is not None and not 0 <= cause_id < len(CAUSE_NAMES):
        raise ValueError(f"cause_id must be between 0 and {len(CAUSE_NAMES) - 1}")
    if not 0 <= severity < len(SEVERITY_NAMES):
        raise ValueError(f"severity must be between 0 and {len(SEVERITY_NAMES) - 1}")
    return _generate_trajectory(sequence_length, cause_id, severity, rng or np.random.default_rng())


def generate_forecast_window(
    sequence_length: int,
    cause_id: int | None,
    severity: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return observations whose optional incident reaches target state 60 seconds later.

    Incidents begin as a weak precursor inside the observed window and reach the
    labelled state at the fixed future target. The target sample is not returned.
    """
    if sequence_length < 8:
        raise ValueError("sequence_length must be at least 8")
    if cause_id is not None and not 0 <= cause_id < len(CAUSE_NAMES):
        raise ValueError(f"cause_id must be between 0 and {len(CAUSE_NAMES) - 1}")
    if not 0 <= severity < len(SEVERITY_NAMES):
        raise ValueError(f"severity must be between 0 and {len(SEVERITY_NAMES) - 1}")
    rng = rng or np.random.default_rng()
    total_length = sequence_length + FORECAST_STEPS
    start = None
    if cause_id is not None:
        start = int(rng.integers(sequence_length // 2, sequence_length - 1))
    trajectory = _generate_trajectory(total_length, cause_id, severity, rng, start)
    return trajectory[:sequence_length]


def make_dataset(samples: int, sequence_length: int, seed: int = 7) -> dict[str, np.ndarray]:
    """Build windows labelled by an SLA-degradation state 60 seconds ahead."""
    if samples < 1:
        raise ValueError("samples must be positive")
    if sequence_length < 8:
        raise ValueError("sequence_length must be at least 8")
    rng = np.random.default_rng(seed)
    raw = np.empty((samples, sequence_length, len(FEATURE_NAMES)), dtype=np.float32)
    incident = np.empty(samples, dtype=np.float32)
    cause = np.zeros(samples, dtype=np.int64)
    severity = np.zeros(samples, dtype=np.int64)

    for row in range(samples):
        is_incident = rng.random() < 0.58
        incident[row] = float(is_incident)
        if is_incident:
            cause[row] = int(rng.integers(len(CAUSE_NAMES)))
            severity[row] = int(rng.choice(3, p=(0.28, 0.48, 0.24)))
            raw[row] = generate_forecast_window(sequence_length, int(cause[row]), int(severity[row]), rng)
        else:
            raw[row] = generate_forecast_window(sequence_length, None, rng=rng)

    return {
        "x": normalize_features(raw),
        "incident": incident,
        "cause": cause,
        "severity": severity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_train.npz"))
    parser.add_argument("--samples", type=int, default=2400)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.samples < 8 or args.sequence_length < 8:
        raise ValueError("--samples and --sequence-length must both be at least 8")

    data = make_dataset(args.samples, args.sequence_length, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    positives = int(data["incident"].sum())
    print(f"Wrote {args.output} ({args.samples} windows; {positives} degradation windows).")


if __name__ == "__main__":
    main()
