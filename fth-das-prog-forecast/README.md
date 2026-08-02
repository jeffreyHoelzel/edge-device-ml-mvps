# FTH-DAS Risk Forecasting MVP

This is a small, CPU-only research MVP that forecasts whether a synthetic physical event may escalate toward a protected fiber location. It uses generated spatiotemporal intensity arrays shaped as **time × distance along a fiber**. It is not real DAS data processing and is not an implementation of any proprietary VIAVI pipeline.

The model is a small spatial 1D CNN followed by a GRU. It predicts an event class, trajectory direction, speed, future location, and escalation probability. Synthetic cases cover background noise, passing vehicles, stationary vibration, excavation approaching the asset, and excavation moving away.

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/). All dependencies, including pytest, are declared in the monorepo root `pyproject.toml`.

```bash
uv sync --dev
```

## Train

Training is CPU-only and writes a reloadable PyTorch checkpoint.

```bash
uv run --directory fth-das-prog-forecast python train.py
```

For a quicker smoke run:

```bash
uv run --directory fth-das-prog-forecast python train.py --epochs 2 --samples 160
```

The default checkpoint is `artifacts/das_risk_model.pt`.

## Test

```bash
PYTHONPATH=fth-das-prog-forecast uv run pytest fth-das-prog-forecast/tests
```

The test suite checks model output shapes, output probability bounds, and future-location projection/clipping.

## Visualize a track

```bash
uv run --directory fth-das-prog-forecast python visualize_track.py --scenario excavation_approaching
```

This saves `artifacts/event_track.png`, containing a DAS-like intensity heatmap, physical event track, and highlighted protected zone. Other scenarios can be selected with `--scenario`, such as `excavation_away` or `passing_vehicle`.

## Run inference

Train first, then run:

```bash
uv run --directory fth-das-prog-forecast python infer.py --scenario excavation_approaching
```

The script reloads the saved model on CPU and emits JSON in this form:

```json
{
  "event_type": "excavation_activity",
  "current_location_m": 1842,
  "trajectory": "approaching_protected_asset",
  "estimated_speed_m_per_min": 4.2,
  "risk_horizon_seconds": 300,
  "escalation_probability": 0.81,
  "predicted_risk": "high"
}
```

`predicted_future_location_m` is also included to expose the future-location head.

## Stream demonstration

```bash
uv run --directory fth-das-prog-forecast python stream_demo.py --scenario excavation_approaching --updates 5
```

This feeds rolling windows from one synthetic trace into the reloaded model and prints one JSON forecast per update.
