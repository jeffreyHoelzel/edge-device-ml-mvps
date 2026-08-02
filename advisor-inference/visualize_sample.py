"""Save a readable image of an individual generated spectrogram."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from contract import FREQUENCY_BINS, TIME_BINS

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a .npy spectrogram or a dataset sample.")
    parser.add_argument("--input", type=Path, default=Path("data/example_spectrogram.npy"))
    parser.add_argument("--index", type=int, default=0, help="Index when input is a generated .npz dataset.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sample_visualization.png"))
    args = parser.parse_args()
    loaded = np.load(args.input)
    array = loaded["spectrograms"][args.index, 0] if isinstance(loaded, np.lib.npyio.NpzFile) else np.squeeze(loaded)
    if array.shape != (TIME_BINS, FREQUENCY_BINS):
        raise ValueError(f"Expected a ({TIME_BINS}, {FREQUENCY_BINS}) spectrogram, got {array.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    image = axis.imshow(array, aspect="auto", origin="lower", cmap="magma")
    axis.set(title="RF spectrogram window", xlabel="Normalized frequency", ylabel="Time bin")
    figure.colorbar(image, ax=axis, label="Relative power")
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
