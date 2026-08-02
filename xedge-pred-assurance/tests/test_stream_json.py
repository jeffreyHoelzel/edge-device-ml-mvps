import json

from stream_demo import forecast_event


def test_forecast_event_json_schema() -> None:
    event = json.loads(json.dumps(forecast_event(0.88, 0, 1)))

    assert set(event) == {
        "event_type",
        "probability",
        "forecast_horizon_seconds",
        "likely_cause",
        "severity",
    }
    assert event["event_type"] == "sla_violation_forecast"
    assert event["probability"] == 0.88
    assert event["forecast_horizon_seconds"] == 60
    assert event["likely_cause"] == "rf_interference"
    assert event["severity"] == "major"
