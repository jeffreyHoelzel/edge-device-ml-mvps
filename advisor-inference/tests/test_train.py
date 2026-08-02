import numpy as np
import pytest
import torch

from contract import FREQUENCY_BINS, TIME_BINS, normalize_per_window
from train import evaluate, is_better_metrics, load_training_data, positive_float, positive_int, stratified_split_indices


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


class _FixedOutputModel(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "class_logits": torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]),
            "impact_logits": torch.tensor([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 3.0]]),
            "frequency_bounds": torch.tensor([[0.7, 0.8], [0.1, 0.4], [0.5, 0.7]]),
        }


def test_evaluate_reports_all_prediction_head_metrics() -> None:
    features = torch.zeros((3, 1, TIME_BINS, FREQUENCY_BINS))
    classes = torch.tensor([0, 1, 2])
    bounds = torch.tensor([[0.0, 0.0], [0.2, 0.4], [0.4, 0.8]])
    impacts = torch.tensor([0, 1, 2])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features, classes, bounds, impacts), batch_size=3)
    metrics = evaluate(_FixedOutputModel(), loader, torch.device("cpu"))
    assert metrics["class_accuracy"] == 1.0
    assert metrics["impact_accuracy"] == pytest.approx(2 / 3)
    assert metrics["bounds_mae"] == pytest.approx(0.075)


def test_metric_selection_prefers_class_accuracy_then_bound_error() -> None:
    first = {"class_accuracy": 0.8, "impact_accuracy": 0.5, "bounds_mae": 0.2}
    assert is_better_metrics(first, None)
    assert is_better_metrics({"class_accuracy": 0.9, "impact_accuracy": 0.1, "bounds_mae": 0.9}, first)
    assert is_better_metrics({"class_accuracy": 0.8, "impact_accuracy": 0.1, "bounds_mae": 0.1}, first)
    assert not is_better_metrics({"class_accuracy": 0.8, "impact_accuracy": 0.9, "bounds_mae": 0.2}, first)
