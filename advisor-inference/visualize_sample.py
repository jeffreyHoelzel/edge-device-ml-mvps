"""Save a readable image of an individual generated spectrogram."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from contract import FREQUENCY_BINS, TIME_BINS, validate_frequency_span


def load_visualization_array(path: Path, index: int) -> np.ndarray:
    """Load one validated spectrogram from .npy or the public dataset .npz format."""
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "spectrograms" not in loaded.files:
                raise ValueError("Dataset input is missing the spectrograms array")
            spectrograms = loaded["spectrograms"]
            if not 0 <= index < len(spectrograms):
                raise ValueError(f"Dataset index must be from 0 to {len(spectrograms) - 1}, got {index}")
            array = spectrograms[index, 0]
        finally:
            loaded.close()
    else:
        array = np.squeeze(loaded)
    if array.shape != (TIME_BINS, FREQUENCY_BINS):
        raise ValueError(f"Expected a ({TIME_BINS}, {FREQUENCY_BINS}) spectrogram, got {array.shape}")
    return array

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a .npy spectrogram or a dataset sample.")
    parser.add_argument("--input", type=Path, default=Path("data/example_spectrogram.npy"))
    parser.add_argument("--index", type=int, default=0, help="Index when input is a generated .npz dataset.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sample_visualization.png"))
    parser.add_argument("--frequency-start-hz", type=float)
    parser.add_argument("--frequency-stop-hz", type=float)
    args = parser.parse_args()
    try:
        frequency_span_hz = validate_frequency_span(args.frequency_start_hz, args.frequency_stop_hz)
    except ValueError as error:
        parser.error(str(error))
    array = load_visualization_array(args.input, args.index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    image_args = {"aspect": "auto", "origin": "lower", "cmap": "magma"}
    if frequency_span_hz is None:
        image = axis.imshow(array, **image_args)
        axis.set(title="RF spectrogram window", xlabel="Frequency bin", ylabel="Time bin")
    else:
        image = axis.imshow(array, extent=(*frequency_span_hz, 0, TIME_BINS), **image_args)
        axis.set(title="RF spectrogram window", xlabel="Frequency (Hz)", ylabel="Time bin")
    figure.colorbar(image, ax=axis, label="Relative power")
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
