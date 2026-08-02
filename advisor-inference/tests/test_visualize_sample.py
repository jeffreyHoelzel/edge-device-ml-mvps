import numpy as np
import pytest

from contract import FREQUENCY_BINS, TIME_BINS
from visualize_sample import load_visualization_array


def test_load_visualization_array_reads_npy_and_dataset_sample(tmp_path) -> None:
    sample = np.zeros((TIME_BINS, FREQUENCY_BINS), dtype=np.float32)
    npy_path = tmp_path / "sample.npy"
    np.save(npy_path, sample)
    assert np.array_equal(load_visualization_array(npy_path, 0), sample)

    dataset_path = tmp_path / "dataset.npz"
    np.savez(dataset_path, spectrograms=np.stack([sample, sample + 1])[:, None])
    assert np.array_equal(load_visualization_array(dataset_path, 1), sample + 1)


def test_load_visualization_array_rejects_invalid_dataset_index(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.npz"
    np.savez(dataset_path, spectrograms=np.zeros((1, 1, TIME_BINS, FREQUENCY_BINS), dtype=np.float32))
    with pytest.raises(ValueError, match="Dataset index"):
        load_visualization_array(dataset_path, -1)
