"""Render a synthetic DAS intensity panel and its physical event track."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_data import FIBER_LENGTH_M, PROTECTED_ZONE_M, SCENARIOS, TIME_STEP_SECONDS, generate_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, default="excavation_approaching")
    parser.add_argument("--output", type=Path, default=Path("artifacts/event_track.png"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    signal, meta = generate_sample(np.random.default_rng(args.seed), scenario=args.scenario)
    times = np.arange(signal.shape[0])
    distances = np.linspace(0, FIBER_LENGTH_M, signal.shape[1])
    track = meta.current_location_m + meta.direction * meta.speed_m_per_min * times * TIME_STEP_SECONDS / 60.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    figure, (heatmap_ax, track_ax) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    image = heatmap_ax.imshow(
        signal, extent=[0, FIBER_LENGTH_M, signal.shape[0] - 1, 0], aspect="auto", cmap="magma"
    )
    heatmap_ax.axvspan(PROTECTED_ZONE_M - 50, PROTECTED_ZONE_M + 50, color="cyan", alpha=0.25, label="protected zone")
    heatmap_ax.plot(track, times, "c--", linewidth=2, label="synthetic event track")
    heatmap_ax.set_ylabel("time step")
    heatmap_ax.set_title(f"Synthetic DAS-like intensity: {meta.scenario}")
    heatmap_ax.legend(loc="upper right")
    figure.colorbar(image, ax=heatmap_ax, label="signal intensity")

    track_ax.plot(times, track, marker="o", color="tab:orange", label="event location")
    track_ax.axhspan(PROTECTED_ZONE_M - 50, PROTECTED_ZONE_M + 50, color="tab:red", alpha=0.18, label="protected zone")
    track_ax.set(xlabel="time step", ylabel="location along fiber (m)")
    track_ax.legend(loc="best")
    figure.savefig(args.output, dpi=150)
    print(f"saved visualization to {args.output}")


if __name__ == "__main__":
    main()
