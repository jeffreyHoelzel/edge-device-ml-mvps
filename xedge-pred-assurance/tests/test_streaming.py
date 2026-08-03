import io
import json

import pytest
import torch

from model import LoadedModel, ModelConfig, PredictiveAssuranceModel, TrainingContract
from streaming import KpiStreamProcessor, run_jsonl


def loaded_model() -> LoadedModel:
    model = PredictiveAssuranceModel(ModelConfig(conv_channels=2, gru_hidden=2))
    return LoadedModel(model.eval(), TrainingContract(("latency_ms", "jitter_ms", "packet_loss_pct", "uplink_mbps", "downlink_mbps", "signal_quality_pct", "retransmission_count"), (0.0,) * 7, (1.0,) * 7, ("rf", "congestion", "backhaul", "device"), ("minor", "major", "critical"), 2, 5, 60), {})


def record(timestamp: str, service_id: str = "edge-1") -> dict[str, object]:
    return {"timestamp": timestamp, "service_id": service_id, "latency_ms": 1, "jitter_ms": 1, "packet_loss_pct": 0, "uplink_mbps": 1, "downlink_mbps": 1, "signal_quality_pct": 90, "retransmission_count": 0}


def test_processor_emits_timestamped_event_after_complete_window() -> None:
    processor = KpiStreamProcessor(loaded_model())
    assert processor.process(record("2026-01-01T00:00:00Z")) is None
    event = processor.process(record("2026-01-01T00:00:05Z"))
    assert event is not None
    assert event["service_id"] == "edge-1"
    assert event["forecast_target_timestamp"] == "2026-01-01T00:01:05Z"


def test_processor_resets_on_cadence_gap() -> None:
    processor = KpiStreamProcessor(loaded_model())
    processor.process(record("2026-01-01T00:00:00Z"))
    with pytest.raises(ValueError, match="cadence gap"):
        processor.process(record("2026-01-01T00:00:10Z"))


def test_jsonl_runner_reports_bad_records_and_continues() -> None:
    source = io.StringIO("not-json\n" + json.dumps(record("2026-01-01T00:00:00Z")) + "\n" + json.dumps(record("2026-01-01T00:00:05Z")) + "\n")
    output = io.StringIO()
    errors = io.StringIO()
    run_jsonl(loaded_model(), source, output, errors)
    assert "line 1" in errors.getvalue()
    assert len(output.getvalue().splitlines()) == 1


@pytest.mark.parametrize("value", ("1", True))
def test_processor_rejects_non_numeric_kpis(value: object) -> None:
    item = record("2026-01-01T00:00:00Z")
    item["latency_ms"] = value

    with pytest.raises(ValueError, match="JSON numbers"):
        KpiStreamProcessor(loaded_model()).process(item)


def test_jsonl_runner_continues_after_overflowing_kpi_value() -> None:
    invalid = record("2026-01-01T00:00:00Z")
    invalid["latency_ms"] = 10**400
    source = io.StringIO(
        json.dumps(invalid)
        + "\n"
        + json.dumps(record("2026-01-01T00:00:00Z"))
        + "\n"
        + json.dumps(record("2026-01-01T00:00:05Z"))
        + "\n"
    )
    output = io.StringIO()
    errors = io.StringIO()

    run_jsonl(loaded_model(), source, output, errors)

    assert "line 1" in errors.getvalue()
    assert len(output.getvalue().splitlines()) == 1
