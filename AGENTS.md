# Repository Guide

## Overview and setup

This monorepo contains small, CPU-only machine-learning MVPs for edge-device
workflows. The projects are intentionally script-oriented rather than packaged
Python applications. Each MVP imports sibling modules directly, so run commands
from its directory via `uv run --directory <mvp>` and test it with a matching
`PYTHONPATH`.

The root `pyproject.toml` and `uv.lock` define the shared environment. The
project requires Python 3.14 and uses `uv`, PyTorch CPU wheels, NumPy,
scikit-learn, pandas, Pydantic, ONNX/ONNX Runtime, matplotlib, and pytest.

```bash
uv sync --dev
```

Use `python <script>.py --help` with `uv run --directory <mvp>` to inspect a
script's options. Do not add CUDA requirements: models, checkpoints, and
inference are expected to work on CPU.

## MVP directories and commands

### `advisor-inference`

RF-interference classification and frequency-range localization from synthetic
spectrograms. `generate_data.py` creates training inputs, `train.py` saves the
CNN checkpoint, `infer.py` emits one JSON result, and `visualize_sample.py`
renders a spectrogram image.

```bash
uv run --directory advisor-inference python generate_data.py --samples 1000 --output data/train.npz
uv run --directory advisor-inference python train.py --data data/train.npz --epochs 18 --model-output artifacts/rf_interference_cnn.pt
uv run --directory advisor-inference python infer.py data/example_spectrogram.npy --model artifacts/rf_interference_cnn.pt
uv run --directory advisor-inference python visualize_sample.py --input data/example_spectrogram.npy --output artifacts/sample_visualization.png
PYTHONPATH=advisor-inference uv run pytest advisor-inference/tests
```

### `fth-das-prog-forecast`

Distributed-acoustic-sensing risk forecasting from synthetic time-by-distance
traces. `train.py` generates its in-memory data and writes a checkpoint;
`infer.py` produces one forecast; `stream_demo.py` emits rolling forecasts; and
`visualize_track.py` saves an event-track plot. `generate_data.py` and
`model.py` are support modules, not standalone CLIs.

```bash
uv run --directory fth-das-prog-forecast python train.py --epochs 2 --samples 160
uv run --directory fth-das-prog-forecast python infer.py --scenario excavation_approaching
uv run --directory fth-das-prog-forecast python stream_demo.py --scenario excavation_approaching --updates 5
uv run --directory fth-das-prog-forecast python visualize_track.py --scenario excavation_approaching
PYTHONPATH=fth-das-prog-forecast uv run pytest fth-das-prog-forecast/tests
```

### `gigastor-m-edge-inference`

Application-performance root-cause ranking from synthetic rolling flow
summaries. `generate_data.py` creates the dataset, `train.py` saves a GRU
checkpoint, `infer.py` emits an incident JSON record, and `stream_demo.py`
prints one record per flow window.

```bash
uv run --directory gigastor-m-edge-inference python generate_data.py --output data/synthetic_flows.npz --samples 2000 --seed 7
uv run --directory gigastor-m-edge-inference python train.py --data data/synthetic_flows.npz --model artifacts/observer_gru.pt --epochs 30
uv run --directory gigastor-m-edge-inference python infer.py --model artifacts/observer_gru.pt --data data/synthetic_flows.npz --index 0 --top-k 3
uv run --directory gigastor-m-edge-inference python stream_demo.py --model artifacts/observer_gru.pt --data data/synthetic_flows.npz --events 5
PYTHONPATH=gigastor-m-edge-inference uv run pytest gigastor-m-edge-inference/tests
```

### `xedge-pred-assurance`

Predictive network-service assurance from synthetic rolling KPI windows.
`generate_data.py` creates the dataset, `train.py` saves the multi-head model,
and `stream_demo.py` simulates incoming KPI measurements and emits forecast
JSON. `model.py` is a support module, not a standalone CLI.

```bash
uv run --directory xedge-pred-assurance python generate_data.py --output data/synthetic_train.npz
uv run --directory xedge-pred-assurance python train.py --data data/synthetic_train.npz --output artifacts/model.pt
uv run --directory xedge-pred-assurance python stream_demo.py --model artifacts/model.pt --cause rf_interference
PYTHONPATH=xedge-pred-assurance uv run pytest xedge-pred-assurance/tests
```

## Contribution guidelines

- Always develop changes in a Git worktree under `worktrees/`; keep the main
  checkout clean and use it only for coordination and inspection. Create a
  branch worktree before editing, for example:

  ```bash
  git worktree add worktrees/<topic> -b <branch-name>
  ```

  Run development commands and tests from that worktree. Do not make code or
  documentation changes directly in the main checkout.
- Keep Python source files below 1,000 lines; split cohesive functionality into
  focused modules before reaching that limit.
- Preserve CPU-only execution. Load checkpoints with CPU-compatible settings and
  avoid device-specific assumptions.
- Keep synthetic data generation and seeded training deterministic when a seed
  is provided. Document intentional changes to generated-data behavior.
- Treat inference and streaming JSON as public demo interfaces. Preserve field
  names and value meanings, or update tests and documentation with a compatible
  migration plan.
- Add or update focused pytest coverage for behavioral changes, then run the
  affected MVP's isolated test command. Run all four suites when changing shared
  dependencies or cross-project conventions.
- Do not commit generated `data/`, `artifacts/`, virtual environments, caches,
  coverage output, or local editor files. These paths are ignored already.
- Follow the prevailing simple Python style. No formatter, linter, or type
  checker is configured, so do not introduce one incidentally.
- Keep changes scoped to one MVP unless a shared dependency, documentation, or
  repo-wide convention genuinely needs to change.

## Commits

Use a concise, imperative subject prefixed with a Conventional Commit-style
type:

- `feat:` for a user-visible capability
- `fix:` for a bug correction
- `refactor:` for behavior-preserving restructuring
- `test:` for test-only changes
- `docs:` for documentation
- `chore:` for maintenance, dependencies, or tooling
- `perf:` for measured performance improvements

Examples: `feat: add confidence threshold to RF inference` and
`test: cover clipped DAS future locations`.
