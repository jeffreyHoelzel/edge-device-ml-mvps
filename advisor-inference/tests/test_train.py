import numpy as np
import pytest
import torch

from contract import FREQUENCY_BINS, TIME_BINS, normalize_per_window
from train import load_training_data, positive_float, positive_int, stratified_split_indices


def _write_dataset(path, **overrides: np.ndarray) -> None:
    arrays = {
        "spectrograms": np.zeros((2, 1, TIME_BINS, FREQUENCY_BINS), dtype=np.float32),
        "classes": np.array([0, 1], dtype=np.int64),
        "bounds": np.array([[0.0, 0.0], [0.2, 0.4]], dtype=np.float32),
        "impacts": np.array([0, 1], dtype=np.int64),
    }
    arrays.update(overrides)
    np.savez(path, **arrays)


def test_load_training_data_accepts_valid_contract(tmp_path) -> None:
    path = tmp_path / "valid.npz"
    _write_dataset(path)
    features, classes, bounds, impacts = load_training_data(path)
    assert features.shape == (2, 1, TIME_BINS, FREQUENCY_BINS)
    assert classes.tolist() == [0, 1]
    assert bounds.shape == (2, 2)
    assert impacts.tolist() == [0, 1]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"classes": np.array([0])}, "align"),
        ({"spectrograms": np.zeros((2, TIME_BINS, FREQUENCY_BINS), dtype=np.float32)}, "shape"),
        ({"classes": np.array([0, 5])}, "classes"),
        ({"impacts": np.array([0, 3])}, "impacts"),
        ({"bounds": np.array([[0.0, 0.0], [0.7, 0.2]])}, "bounds"),
    ],
)
def test_load_training_data_rejects_invalid_contract(tmp_path, overrides, message: str) -> None:
    path = tmp_path / "invalid.npz"
    _write_dataset(path, **overrides)
    with pytest.raises(ValueError, match=message):
        load_training_data(path)


def test_positive_argument_types_reject_invalid_values() -> None:
    with pytest.raises(Exception):
        positive_int("0")
    with pytest.raises(Exception):
        positive_float("0")
    with pytest.raises(Exception):
        positive_float("nan")


def test_normalization_is_shared_between_training_and_inference() -> None:
    normalized = normalize_per_window(torch.zeros((1, 1, TIME_BINS, FREQUENCY_BINS)))
    assert normalized.shape == (1, 1, TIME_BINS, FREQUENCY_BINS)


def test_stratified_split_is_seeded_non_overlapping_and_preserves_singletons() -> None:
    classes = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2])
    train_indices, validation_indices = stratified_split_indices(classes, seed=11)
    repeated_train, repeated_validation = stratified_split_indices(classes, seed=11)
    assert torch.equal(train_indices, repeated_train)
    assert torch.equal(validation_indices, repeated_validation)
    assert not set(train_indices.tolist()).intersection(validation_indices.tolist())
    assert sorted(train_indices.tolist() + validation_indices.tolist()) == list(range(len(classes)))
    assert 8 in train_indices.tolist()
    assert 8 not in validation_indices.tolist()
    assert set(classes[validation_indices].tolist()) == {0, 1}
