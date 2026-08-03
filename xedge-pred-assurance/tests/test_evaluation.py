from dataclasses import replace

import numpy as np
import pytest

from generate_data import make_dataset
from train import EvaluationMetrics, enforce_quality_gates, expected_calibration_error, split_indices


def test_split_is_deterministic_and_has_no_overlap() -> None:
    data = make_dataset(240, 12, seed=3)
    first_train, first_evaluation = split_indices(data, seed=8)
    second_train, second_evaluation = split_indices(data, seed=8)

    assert np.array_equal(first_train, second_train)
    assert np.array_equal(first_evaluation, second_evaluation)
    assert not np.intersect1d(first_train, first_evaluation).size


def test_expected_calibration_error_weights_bin_error_by_bin_size() -> None:
    probabilities = np.array([0.2, 0.2, 0.8, 0.8])
    labels = np.array([0, 0, 1, 1])

    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.2)


def test_quality_gate_rejects_weak_metrics() -> None:
    metrics = EvaluationMetrics(0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.01, 0.01)

    enforce_quality_gates(metrics)
    with pytest.raises(ValueError, match="incident_auroc"):
        enforce_quality_gates(replace(metrics, incident_auroc=0.50))
