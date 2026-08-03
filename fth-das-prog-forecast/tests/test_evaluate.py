import torch

from evaluate import QUALITY_GATES, evaluate_model, quality_gate_failures
from generate_data import SyntheticDASDataset
from model import DASRiskModel


def test_evaluation_returns_expected_metrics_and_calibration_bins() -> None:
    metrics = evaluate_model(DASRiskModel(), SyntheticDASDataset(10, seed=12), batch_size=4)
    assert metrics["samples"] == 10
    assert set(QUALITY_GATES).issubset(metrics)
    assert len(metrics["calibration_bins"]) == 5
    assert sum(bin_["count"] for bin_ in metrics["calibration_bins"]) == 10


def test_quality_gate_failures_report_both_metric_directions() -> None:
    metrics = {name: threshold for name, threshold in QUALITY_GATES.items()}
    metrics["event_accuracy"] = 0.0
    metrics["speed_mae_m_per_min"] = 99.0
    failures = quality_gate_failures(metrics)
    assert any("event_accuracy" in failure for failure in failures)
    assert any("speed_mae_m_per_min" in failure for failure in failures)
