import numpy as np
import pytest

from train import load_training_data, positive_float, positive_int


def _write_dataset(path, **overrides: np.ndarray) -> None:
    arrays = {
        "spectrograms": np.zeros((2, 1, 64, 128), dtype=np.float32),
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
    assert features.shape == (2, 1, 64, 128)
    assert classes.tolist() == [0, 1]
    assert bounds.shape == (2, 2)
    assert impacts.tolist() == [0, 1]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"classes": np.array([0])}, "align"),
        ({"spectrograms": np.zeros((2, 64, 128), dtype=np.float32)}, "shape"),
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
