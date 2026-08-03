# Edge Device ML MVPs

> **Disclaimer:** These MVPs were created using only publicly available VIAVI product information. All datasets, measurements, telemetry, and examples are synthetic; no real, customer, confidential, proprietary, or VIAVI operational data was used. These materials are illustrative MVPs only and are not affiliated with or endorsed by VIAVI.

## Executive Summary

This monorepo demonstrates compact, CPU-only machine-learning workflows that turn high-rate edge measurements into structured predictions, confidence, and supporting evidence. Every MVP uses synthetic data and is intended to explore a narrow edge-intelligence task—not to represent a validated production deployment or a proprietary product pipeline.

- [`advisor-inference`](advisor-inference/README.md) classifies RF-interference patterns in spectrogram windows, localizes the affected frequency range, and estimates service impact. Its intended VIAVI targets are OneAdvisor 800 and CellAdvisor 5G spectrum-analysis workflows.
- [`fth-das-prog-forecast`](fth-das-prog-forecast/README.md) forecasts whether a detected physical event may escalate toward a protected fiber location. Its intended VIAVI target is the FTH-DAS distributed acoustic sensing interrogator.
- [`gigastor-m-edge-inference`](gigastor-m-edge-inference/README.md) ranks likely causes of application-performance degradation from rolling packet-derived flow summaries. Its intended VIAVI target is Observer GigaStor M.
- [`xedge-pred-assurance`](xedge-pred-assurance/README.md) forecasts an impending network service degradation and identifies its likely cause and severity. Its intended VIAVI target is XEdge monitoring sensors.

The shared design pattern is deterministic measurement processing followed by a task-specific model that emits a machine-readable insight for a dashboard, alarm, workflow, or downstream analytics system.

Included MVPs:

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
uv run --directory advisor-inference python train.py
uv run --directory fth-das-prog-forecast python train.py --epochs 2 --samples 160 --no-enforce-quality
uv run --directory gigastor-m-edge-inference python stream_demo.py --events 5
uv run --directory xedge-pred-assurance python stream_demo.py --cause rf_interference
```

Run each MVP's tests independently so its local, script-style imports resolve correctly:

```bash
PYTHONPATH=advisor-inference uv run pytest advisor-inference/tests
PYTHONPATH=fth-das-prog-forecast uv run pytest fth-das-prog-forecast/tests
PYTHONPATH=gigastor-m-edge-inference uv run pytest gigastor-m-edge-inference/tests
PYTHONPATH=xedge-pred-assurance uv run pytest xedge-pred-assurance/tests
```

## Architecture Overview

Each MVP is intentionally self-contained and imports sibling modules directly. Run its scripts with `uv run --directory <mvp>` so those script-style imports resolve correctly.

```text
edge-device-ml-mvps/
├── pyproject.toml, uv.lock              # Shared Python environment and lockfile
├── viavi_edge_ai_model_approaches.md    # Public-information model approach reference
├── advisor-inference/                   # RF interference classification and localization
│   ├── generate_data.py, train.py, infer.py
│   ├── model.py, contract.py, visualize_sample.py
│   └── tests/
├── fth-das-prog-forecast/               # DAS event progression and risk forecasting
│   ├── generate_data.py, train.py, infer.py, stream_demo.py
│   ├── model.py, checkpoint.py, evaluate.py, benchmark.py, visualize_track.py
│   └── tests/
├── gigastor-m-edge-inference/           # Application-performance root-cause ranking
│   ├── generate_data.py, train.py, infer.py, stream_demo.py
│   ├── model.py, data_contract.py
│   └── tests/
└── xedge-pred-assurance/                # Predictive network service assurance
    ├── generate_data.py, train.py, stream_demo.py
    ├── model.py, streaming.py
    └── tests/
```
