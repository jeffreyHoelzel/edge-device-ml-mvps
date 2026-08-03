# FTH-DAS Risk Forecasting MVP

This is a small, CPU-only research MVP that forecasts whether a synthetic physical event may escalate toward a protected fiber location. It uses generated spatiotemporal intensity arrays shaped as **time × distance along a fiber**. It is not real DAS data processing and is not an implementation of any proprietary VIAVI pipeline.

The model is a small spatial 1D CNN followed by a GRU. It predicts an event class, trajectory direction, speed, and escalation probability. The forecast location is deterministically projected from the final frame's supplied physical location, predicted trajectory and speed, and the requested horizon. Synthetic cases cover background noise, passing vehicles, stationary vibration, excavation approaching the asset, and excavation moving away.

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

For a quicker smoke run without enforcing the default quality gates:

```bash
uv run --directory fth-das-prog-forecast python train.py --epochs 2 --samples 160 --no-enforce-quality
```

The default checkpoint is `artifacts/das_risk_model.pt`. Checkpoints are format version 2 and include model architecture, label mappings, fiber and protected-zone geometry, timing, forecast horizon, seed, probability-tier policy, and held-out validation metrics. Earlier checkpoints are intentionally rejected; retrain with the current `train.py`.

Training uses a separate fixed-seed synthetic validation dataset, retains the best validation checkpoint, and enforces modest deterministic quality gates by default. Use `--seed`, `--horizon-seconds`, `--validation-samples`, and `--validation-seed` to configure reproducible runs. `--no-enforce-quality` is intended only for smoke tests or exploration. Inference, streaming, and evaluation use the checkpoint's recorded horizon by default; their optional `--horizon-seconds` argument explicitly overrides it.

## Test

```bash
PYTHONPATH=fth-das-prog-forecast uv run pytest fth-das-prog-forecast/tests
```

The test suite covers model/data validation, protected-zone geometry, checkpoint compatibility, forecast JSON, rolling-stream timing, evaluation metrics, visualization output, and CPU benchmark schema.

## Evaluate a checkpoint

```bash
uv run --directory fth-das-prog-forecast python evaluate.py --model-path artifacts/das_risk_model.pt --enforce-quality
```

Evaluation reports held-out event and direction accuracy, speed and derived future-location MAE, risk MAE/Brier score, and five probability calibration bins. The synthetic probability tiers are `low` below 0.35, `medium` from 0.35 to below 0.67, and `high` at or above 0.67. They are probability bands for this demonstration, not operational alert guarantees.

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

`predicted_future_location_m` is included as the physically derived location forecast.

`current_location_m` is the physical event location at the input window's final frame. `predicted_future_location_m` is derived from that value, `trajectory`, `estimated_speed_m_per_min`, and `risk_horizon_seconds`, and is clipped to the modeled 0–2,500 m fiber. A stationary trajectory reports zero speed and retains the current location.

## Stream demonstration

```bash
uv run --directory fth-das-prog-forecast python stream_demo.py --scenario excavation_approaching --updates 5
```

This feeds rolling windows from one synthetic trace into the reloaded model and prints one JSON forecast per update.

Each stream record also includes `window_end_offset_seconds`, which is the timestamp offset of that rolling window's final frame. The demo requires a positive update count and enough frames for each full 20-frame window.

## Benchmark CPU inference

```bash
uv run --directory fth-das-prog-forecast python benchmark.py --model-path artifacts/das_risk_model.pt
```

The benchmark reports model/checkpoint size, parameter count, window shape, forward-pass latency summary, and process maximum RSS. It is a local CPU measurement only; it does not establish a deployment SLA.

## Input contract and MVP limits

The model accepts finite floating-point tensors shaped `[batch, time, distance]`, with at least one time step and two distance bins. CLI inference and streaming generate their own synthetic inputs; callers of `forecast()` must supply the physical location at the input window's final frame.

This MVP does not ingest real DAS traces, perform sensor calibration or coordinate mapping, manage multi-channel streams, connect to live transport, deduplicate alerts, export a deployment format, or provide production observability. The synthetic generators, probability calibration, and CPU benchmark should not be treated as field validation or operational readiness evidence.
