# XEdge Predictive Service Assurance MVP

A small, deterministic, CPU-only Python proof of concept that forecasts an impending network SLA degradation from a rolling KPI window. It uses synthetic data, temporal 1D convolutions, a GRU, and three output heads:

- Incident probability
- Likely cause: RF interference, congestion, backhaul degradation, or device fault
- Severity: minor, major, or critical

The seven input KPIs are latency, jitter, packet loss, uplink throughput, downlink throughput, signal quality, and retransmission count. Synthetic incident windows gradually degrade near the end, making them suitable for an advance forecast rather than only post-incident classification.

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

`train.py` is deterministic for a given seed and trains on CPU only. If the data path does not exist, it generates the synthetic data automatically. The saved checkpoint contains both model weights and model configuration.

## Test

```bash
PYTHONPATH=xedge-pred-assurance uv run pytest xedge-pred-assurance/tests
```

The basic tests verify model output tensor shapes and the emitted forecast JSON schema.

## Run the streaming demo

```bash
uv run --directory xedge-pred-assurance python stream_demo.py --model artifacts/model.pt --cause rf_interference
```

The demo simulates individual incoming KPI measurements, keeps a rolling 30-measurement window, reloads the saved model, and emits one JSON event per complete window. Example format:

```json
{
  "event_type": "sla_violation_forecast",
  "probability": 0.88,
  "forecast_horizon_seconds": 60,
  "likely_cause": "rf_interference",
  "severity": "major"
}
```

Choose another simulated pattern with `--cause congestion`, `--cause backhaul_degradation`, or `--cause device_fault`.
