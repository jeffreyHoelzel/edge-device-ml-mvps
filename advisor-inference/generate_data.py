"""Generate labeled synthetic RF spectrogram windows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = [
    "no_interference",
    "narrowband_continuous",
    "wideband_intermittent",
    "periodic_impulsive",
    "adjacent_channel_leakage",
]
IMPACT_NAMES = ["low", "moderate", "high"]
TIME_BINS = 64
FREQUENCY_BINS = 128


@dataclass(frozen=True)
class Sample:
    spectrogram: np.ndarray
    class_id: int
    frequency_bounds: np.ndarray
    impact_id: int


def _choose_band(rng: np.random.Generator, min_width: float, max_width: float) -> tuple[int, int]:
    width = int(rng.integers(max(2, round(min_width * FREQUENCY_BINS)), round(max_width * FREQUENCY_BINS) + 1))
    start = int(rng.integers(0, FREQUENCY_BINS - width + 1))
    return start, start + width


def _impact(amplitude: float, bandwidth: float, duty_cycle: float) -> int:
    score = amplitude * (0.35 + bandwidth) * (0.35 + duty_cycle)
    return 0 if score < 0.75 else 1 if score < 1.7 else 2


def generate_sample(rng: np.random.Generator, class_id: int | None = None) -> Sample:
    """Create one window and labels with randomized physical characteristics."""
    if class_id is None:
        class_id = int(rng.integers(len(CLASS_NAMES)))
    elif not isinstance(class_id, (int, np.integer)) or isinstance(class_id, bool):
        raise ValueError(f"class_id must be an integer from 0 to {len(CLASS_NAMES) - 1}")
    elif not 0 <= class_id < len(CLASS_NAMES):
        raise ValueError(f"class_id must be from 0 to {len(CLASS_NAMES) - 1}, got {class_id}")
    noise_floor = float(rng.uniform(-1.25, -0.45))
    window = rng.normal(noise_floor, rng.uniform(0.08, 0.20), (TIME_BINS, FREQUENCY_BINS)).astype(np.float32)

    if class_id == 0:
        return Sample(window, 0, np.array([0.0, 0.0], dtype=np.float32), 0)

    amplitude = float(rng.uniform(0.8, 3.2))
    if class_id == 1:  # Narrow horizontal line, nearly continuous in time.
        start, stop = _choose_band(rng, 0.015, 0.07)
        duration = int(rng.integers(int(TIME_BINS * 0.72), TIME_BINS + 1))
        time_start = int(rng.integers(0, TIME_BINS - duration + 1))
        window[time_start : time_start + duration, start:stop] += amplitude
        duty = duration / TIME_BINS
    elif class_id == 2:  # Wide band present in intermittent blocks.
        start, stop = _choose_band(rng, 0.16, 0.48)
        duty = float(rng.uniform(0.18, 0.72))
        mask = rng.random(TIME_BINS) < duty
        # Smooth the mask into short, realistic active intervals.
        for time_index in np.flatnonzero(mask):
            window[time_index : min(TIME_BINS, time_index + int(rng.integers(1, 5))), start:stop] += amplitude
        duty = float(np.mean(window[:, start:stop].mean(axis=1) > noise_floor + amplitude * 0.25))
    elif class_id == 3:  # Repeated short, high-energy impulse columns.
        start, stop = _choose_band(rng, 0.12, 0.62)
        period = int(rng.integers(5, 15))
        duration = int(rng.integers(1, 4))
        phase = int(rng.integers(0, period))
        positions = list(range(phase, TIME_BINS, period))
        for position in positions:
            window[position : position + duration, start:stop] += amplitude
        duty = min(1.0, len(positions) * duration / TIME_BINS)
    else:  # Leakage concentrated beside one channel edge with a decaying skirt.
        width = float(rng.uniform(0.06, 0.19))
        edge_on_left = bool(rng.integers(0, 2))
        if edge_on_left:
            stop = int(rng.integers(int(0.35 * FREQUENCY_BINS), int(0.72 * FREQUENCY_BINS)))
            start = max(0, stop - int(width * FREQUENCY_BINS))
            frequencies = np.arange(start, stop)
            profile = amplitude * np.exp(-2.7 * (stop - 1 - frequencies) / max(1, stop - start))
        else:
            start = int(rng.integers(int(0.28 * FREQUENCY_BINS), int(0.65 * FREQUENCY_BINS)))
            stop = min(FREQUENCY_BINS, start + int(width * FREQUENCY_BINS))
            frequencies = np.arange(start, stop)
            profile = amplitude * np.exp(-2.7 * (frequencies - start) / max(1, stop - start))
        duration = int(rng.integers(int(TIME_BINS * 0.45), TIME_BINS + 1))
        time_start = int(rng.integers(0, TIME_BINS - duration + 1))
        window[time_start : time_start + duration, start:stop] += profile
        duty = duration / TIME_BINS

    bounds = np.array([start / FREQUENCY_BINS, stop / FREQUENCY_BINS], dtype=np.float32)
    return Sample(window, class_id, bounds, _impact(amplitude, (stop - start) / FREQUENCY_BINS, duty))


def generate_dataset(samples: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if samples < 1:
        raise ValueError("samples must be at least 1")
    rng = np.random.default_rng(seed)
    generated = [generate_sample(rng, index % len(CLASS_NAMES)) for index in range(samples)]
    rng.shuffle(generated)
    return (
        np.stack([sample.spectrogram for sample in generated])[:, None, :, :],
        np.asarray([sample.class_id for sample in generated], dtype=np.int64),
        np.stack([sample.frequency_bounds for sample in generated]),
        np.asarray([sample.impact_id for sample in generated], dtype=np.int64),
    )


def save_preview(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    figure, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(16, 3), constrained_layout=True)
    for class_id, axis in enumerate(axes):
        sample = generate_sample(rng, class_id)
        axis.imshow(sample.spectrogram, aspect="auto", origin="lower", cmap="magma")
        axis.set_title(CLASS_NAMES[class_id].replace("_", "\n"), fontsize=9)
        axis.set_xlabel("Frequency bin")
    axes[0].set_ylabel("Time bin")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic RF spectrogram data.")
    parser.add_argument("--samples", type=int, default=1_000)
    parser.add_argument("--output", type=Path, default=Path("data/train.npz"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-output", type=Path, default=Path("data/example_spectrogram.npy"))
    parser.add_argument("--preview", type=Path, default=Path("artifacts/generated_examples.png"))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    spectrograms, classes, bounds, impacts = generate_dataset(args.samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, spectrograms=spectrograms, classes=classes, bounds=bounds, impacts=impacts)
    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.sample_output, spectrograms[0, 0])
    save_preview(args.preview, args.seed + 1)
    print(f"Saved {args.samples} samples to {args.output}; inference sample: {args.sample_output}")


if __name__ == "__main__":
    main()
