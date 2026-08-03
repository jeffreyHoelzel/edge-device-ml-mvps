# XEdge Predictive Service Assurance MVP

A small, deterministic, CPU-only Python proof of concept that forecasts an impending network SLA degradation from a rolling KPI window. It uses synthetic data, temporal 1D convolutions, a GRU, and three output heads:

- Incident probability
- Likely cause: RF interference, congestion, backhaul degradation, or device fault
- Severity: minor, major, or critical

The seven input KPIs are latency, jitter, packet loss, uplink throughput, downlink throughput, signal quality, and retransmission count. Synthetic incident windows expose early degradation precursors in the observed window and reach their labelled incident state exactly 60 seconds later. KPI samples have a fixed five-second cadence, so the target is 12 samples ahead and is never included in the model input.

## Requirements

- Python 3.14 (managed by `uv` if it is not already installed)
- `uv`

All dependencies, including the CPU PyTorch wheel and development-only pytest, are declared in the monorepo root `pyproject.toml`.

## Setup

From the monorepo root, create the Python 3.14 environment and install dependencies:

```bash
uv sync --dev --python 3.14
```

## Generate synthetic data

```bash
uv run --directory xedge-pred-assurance python generate_data.py --output data/synthetic_train.npz
```

This produces normal windows plus RF interference, congestion, backhaul degradation, and device-fault degradation patterns.

## Train and save a model

```bash
uv run --directory xedge-pred-assurance python train.py --data data/synthetic_train.npz --output artifacts/model.pt
```

`train.py` is deterministic for a given seed and trains on CPU only. If the data path does not exist, it generates the synthetic data automatically. Training uses a deterministic, stratified holdout and only saves a model that meets fixed synthetic-data quality gates for forecast discrimination, alert precision/recall, diagnosis accuracy, and calibration. The saved checkpoint contains the model plus the complete preprocessing, cadence, horizon, label, threshold, and evaluation contract.

Checkpoints and datasets are versioned contracts. Retrain models after changing this MVP; old unversioned artifacts are rejected rather than interpreted with changed preprocessing.

## Test

```bash
PYTHONPATH=xedge-pred-assurance uv run pytest xedge-pred-assurance/tests
```

The tests cover generation determinism, data and artifact validation, forecast-quality gates, event schema, JSON Lines ingestion, and a generated-data-to-streaming integration path.

## Run the streaming demo

```bash
uv run --directory xedge-pred-assurance python stream_demo.py --model artifacts/model.pt --cause rf_interference
```

Simulation is the default; `--simulate` remains available for explicit invocation. For live-style operation, use `--stdin` and provide one JSON object per line. Each record must contain an RFC 3339 timestamp with timezone, a non-empty `service_id`, and all seven KPI fields. Each service has its own rolling window and must arrive exactly five seconds after its prior sample; malformed records and cadence gaps are reported to stderr and reset only the affected service.

```bash
printf '%s\n' '{"timestamp":"2026-01-01T00:00:00Z","service_id":"edge-17","latency_ms":35,"jitter_ms":4,"packet_loss_pct":0.1,"uplink_mbps":35,"downlink_mbps":80,"signal_quality_pct":82,"retransmission_count":1}' | uv run --directory xedge-pred-assurance python stream_demo.py --model artifacts/model.pt --stdin
```

The model emits one JSON event per complete window. The prediction is for the timestamp exactly 60 seconds after the final sample in that window. `likely_cause` and `severity` remain present for compatibility, but consumers must use `meets_alert_threshold` before treating them as an alert. Example format:

```json
{
  "event_type": "sla_violation_forecast",
  "probability": 0.88,
  "forecast_horizon_seconds": 60,
  "likely_cause": "rf_interference",
  "severity": "major",
  "incident_threshold": 0.7,
  "meets_alert_threshold": true,
  "cause_confidence": 0.91,
  "severity_confidence": 0.83,
  "model_version": "predictive-assurance-v3"
}
```

Choose another simulated pattern with `--cause congestion`, `--cause backhaul_degradation`, or `--cause device_fault`.
