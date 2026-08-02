"""JSON Lines KPI ingestion and forecast-event formatting."""

from __future__ import annotations

import json
import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import TextIO

import numpy as np
import torch

from model import LoadedModel


def forecast_event(
    probability: float, cause_index: int, severity_index: int, contract, **extra: object
) -> dict[str, object]:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if not 0 <= cause_index < len(contract.cause_names):
        raise ValueError("cause_index is out of range")
    if not 0 <= severity_index < len(contract.severity_names):
        raise ValueError("severity_index is out of range")
    event = {
        "event_type": "sla_violation_forecast",
        "probability": round(float(probability), 4),
        "forecast_horizon_seconds": contract.forecast_horizon_seconds,
        "likely_cause": contract.cause_names[cause_index],
        "severity": contract.severity_names[severity_index],
        "incident_threshold": round(float(contract.incident_threshold), 4),
        "meets_alert_threshold": bool(probability >= contract.incident_threshold),
        "cause_confidence": round(float(extra["cause_confidence"]), 4),
        "severity_confidence": round(float(extra["severity_confidence"]), 4),
        "model_version": contract.model_version,
    }
    event.update({key: value for key, value in extra.items() if key not in {"cause_confidence", "severity_confidence"}})
    return event


class KpiStreamProcessor:
    def __init__(self, loaded: LoadedModel) -> None:
        self.loaded = loaded
        self.buffers: dict[str, deque[tuple[datetime, np.ndarray]]] = {}
        self.last_timestamp: dict[str, datetime] = {}

    def process(self, record: dict[str, object]) -> dict[str, object] | None:
        contract = self.loaded.contract
        required = {"timestamp", "service_id", *contract.feature_names}
        if set(record) != required:
            raise ValueError(f"KPI record must contain exactly: {', '.join(sorted(required))}")
        service_id = record["service_id"]
        if not isinstance(service_id, str) or not service_id:
            raise ValueError("service_id must be a non-empty string")
        timestamp = _parse_timestamp(record["timestamp"])
        values = np.asarray([record[name] for name in contract.feature_names], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("KPI values must be finite numbers")
        previous = self.last_timestamp.get(service_id)
        if previous is not None and timestamp - previous != timedelta(seconds=contract.sample_interval_seconds):
            self.buffers.pop(service_id, None)
            self.last_timestamp.pop(service_id, None)
            raise ValueError(f"service {service_id} cadence gap; rolling window reset")
        self.last_timestamp[service_id] = timestamp
        buffer = self.buffers.setdefault(service_id, deque(maxlen=contract.sequence_length))
        buffer.append((timestamp, values))
        if len(buffer) < contract.sequence_length:
            return None
        raw = np.stack([item[1] for item in buffer])
        x = ((raw - contract.feature_mean) / contract.feature_std).astype(np.float32)[None, ...]
        with torch.inference_mode():
            output = self.loaded.model(torch.from_numpy(x))
            probability = torch.sigmoid(output["incident_logits"])[0].item()
            cause = int(output["cause_logits"].argmax(dim=1)[0])
            severity = int(output["severity_logits"].argmax(dim=1)[0])
            cause_confidence = torch.softmax(output["cause_logits"], dim=1).max(dim=1).values[0].item()
            severity_confidence = torch.softmax(output["severity_logits"], dim=1).max(dim=1).values[0].item()
        return forecast_event(
            probability, cause, severity, contract,
            cause_confidence=cause_confidence,
            severity_confidence=severity_confidence,
            service_id=service_id,
            window_end_timestamp=_format_timestamp(timestamp),
            forecast_target_timestamp=_format_timestamp(timestamp + timedelta(seconds=contract.forecast_horizon_seconds)),
        )


def run_jsonl(loaded: LoadedModel, source: TextIO, output: TextIO, errors: TextIO = sys.stderr) -> None:
    processor = KpiStreamProcessor(loaded)
    for line_number, line in enumerate(source, start=1):
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("KPI record must be a JSON object")
            event = processor.process(record)
            if event is not None:
                print(json.dumps(event, sort_keys=True), file=output)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            print(f"line {line_number}: {error}", file=errors)


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an RFC 3339 string")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("timestamp must be an RFC 3339 string") from error
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return timestamp.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
