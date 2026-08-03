import numpy as np
import pytest

from benchmark import benchmark_model
from model import DASRiskModel


def test_benchmark_reports_a_stable_measurement_schema() -> None:
    result = benchmark_model(DASRiskModel(), np.zeros((20, 96), dtype=np.float32), warmup_runs=1, measured_runs=3)
    assert result["warmup_runs"] == 1
    assert result["measured_runs"] == 3
    assert result["input_shape"] == [1, 20, 96]
    assert result["parameter_count"] > 0
    assert set(result["latency_ms"]) == {"minimum", "mean", "p50", "p95", "maximum"}
    assert result["latency_ms"]["minimum"] >= 0


@pytest.mark.parametrize(
    ("warmup_runs", "measured_runs", "message"),
    [(-1, 1, "warmup_runs"), (0, 0, "measured_runs")],
)
def test_benchmark_validates_run_counts(warmup_runs: int, measured_runs: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark_model(DASRiskModel(), np.zeros((20, 96), dtype=np.float32), warmup_runs, measured_runs)
