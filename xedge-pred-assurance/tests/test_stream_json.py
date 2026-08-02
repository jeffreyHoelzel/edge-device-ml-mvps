import json

import pytest

from model import TrainingContract
from streaming import forecast_event


CONTRACT = TrainingContract(("a",) * 7, (0.0,) * 7, (1.0,) * 7, ("rf_interference", "congestion", "backhaul_degradation", "device_fault"), ("minor", "major", "critical"), 2, 5, 60)


def test_forecast_event_json_schema() -> None:
    event = json.loads(json.dumps(forecast_event(0.88, 0, 1, CONTRACT, cause_confidence=1.0, severity_confidence=1.0)))

    assert set(event) == {
        "event_type",
        "probability",
        "forecast_horizon_seconds",
        "likely_cause",
        "severity",
        "incident_threshold",
        "meets_alert_threshold",
        "cause_confidence",
        "severity_confidence",
        "model_version",
    }
    assert event["event_type"] == "sla_violation_forecast"
    assert event["probability"] == 0.88
    assert event["forecast_horizon_seconds"] == 60
    assert event["likely_cause"] == "rf_interference"
    assert event["severity"] == "major"
    assert event["incident_threshold"] == 0.7
    assert event["meets_alert_threshold"] is True
    assert event["cause_confidence"] == 1.0
    assert event["severity_confidence"] == 1.0


def test_forecast_event_marks_low_probability_as_not_alerting() -> None:
    event = forecast_event(0.69, 0, 1, CONTRACT, cause_confidence=1.0, severity_confidence=1.0)

    assert event["meets_alert_threshold"] is False


@pytest.mark.parametrize("probability", (-0.1, 1.1))
def test_forecast_event_rejects_invalid_probability(probability: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        forecast_event(probability, 0, 1, CONTRACT, cause_confidence=1.0, severity_confidence=1.0)
