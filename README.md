# Edge Device ML MVPs

> **Disclaimer:** These MVPs were created using only publicly available VIAVI product information. All datasets, measurements, telemetry, and examples are synthetic; no real, customer, confidential, proprietary, or VIAVI operational data was used. These materials are illustrative MVPs only and are not affiliated with or endorsed by VIAVI.

This monorepo contains CPU-friendly machine-learning MVPs for testing edge-device workflows:

- `advisor-inference`: RF-interference classification and localization.
- `fth-das-prog-forecast`: distributed-acoustic-sensing risk forecasting.
- `gigastor-m-edge-inference`: application-performance root-cause ranking.
- `xedge-pred-assurance`: predictive network service assurance.

## Setup

The root `pyproject.toml` and `uv.lock` are the sole dependency definition for every MVP. From the repository root:

```bash
uv sync --dev
```

Run an MVP command from the root environment with `--directory`:

```bash
uv run --directory advisor-infra python train.py
uv run --directory fth-das-prog-forecast python train.py --epochs 2 --samples 160 --no-enforce-quality
uv run --directory gigastor-m-edge-infra python stream_demo.py --events 5
uv run --directory xedge-pred-assurance python stream_demo.py --cause rf_interference
```

Run each MVP's tests independently so its local, script-style imports resolve correctly:

```bash
PYTHONPATH=advisor-inference uv run pytest advisor-inference/tests
PYTHONPATH=fth-das-prog-forecast uv run pytest fth-das-prog-forecast/tests
PYTHONPATH=gigastor-m-edge-inference uv run pytest gigastor-m-edge-inference/tests
PYTHONPATH=xedge-pred-assurance uv run pytest xedge-pred-assurance/tests
```
