# RF Interference Intelligence MVP

A compact, CPU-only demonstration of a complete RF-interference data-to-insight pipeline. It produces randomized synthetic spectrogram windows, trains a small 2D PyTorch CNN, localizes the affected frequency range, and emits an inference JSON record.

The five supported classes are `no_interference`, `narrowband_continuous`, `wideband_intermittent`, `periodic_impulsive`, and `adjacent_channel_leakage`. Every non-empty class randomizes frequency position, bandwidth, amplitude, duration, duty cycle, and noise floor. The CNN has three output heads: class, normalized start/stop frequency, and low/moderate/high estimated service impact.

## Setup

This project targets Python 3.14 and uses only CPU PyTorch. From the monorepo root:

```bash
uv sync --dev
```

## Generate data and inspect it

```bash
uv run --directory advisor-infra python generate_data.py --samples 1000 --output data/train.npz
uv run --directory advisor-infra python visualize_sample.py --input data/example_spectrogram.npy --output artifacts/sample_visualization.png
```

`generate_data.py` also saves a five-class overview to `artifacts/generated_examples.png` and a single `.npy` inference input to `data/example_spectrogram.npy`. These PNGs are intended for quick visual inspection.

## Train and save a model

```bash
uv run --directory advisor-infra python train.py --data data/train.npz --epochs 18 --model-output artifacts/rf_interference_cnn.pt
```

The checkpoint contains the model state and is saved as a CPU-compatible PyTorch `.pt` file.

## Run inference

```bash
uv run --directory advisor-infra python infer.py data/example_spectrogram.npy --model artifacts/rf_interference_cnn.pt
```

Example output shape:

```json
{
  "event_type": "rf_interference",
  "class": "wideband_intermittent",
  "confidence": 0.93,
  "frequency_start_normalized": 0.42,
  "frequency_stop_normalized": 0.58,
  "estimated_service_impact": "moderate"
}
```

Frequency values are normalized to the `[0, 1]` span of the input window. For `no_interference`, both bounds are emitted as `0.0` and impact is `low`.

## Test

```bash
PYTHONPATH=advisor-infra uv run pytest advisor-infra/tests
```

The tests check all output-head shapes and enforce normalized, ordered frequency bounds both in the CNN and in generated interference labels.
