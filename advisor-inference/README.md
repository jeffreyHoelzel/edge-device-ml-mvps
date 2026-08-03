# RF Interference Intelligence MVP

> **Disclaimer:** These MVPs were created using only publicly available VIAVI product information. All datasets, measurements, telemetry, and examples are synthetic; no real, customer, confidential, proprietary, or VIAVI operational data was used. These materials are illustrative MVPs only and are not affiliated with or endorsed by VIAVI.

## Executive Summary

This MVP demonstrates a compact edge-inference workflow for RF-interference intelligence: it takes a spectrogram window, classifies the interference pattern, estimates the affected frequency range, and reports an estimated service impact in structured JSON. It mirrors the task shape described for spectrum-analysis workflows—detecting what is present, where it occurs, and how it may affect service—using only randomized synthetic inputs and a small CPU-only CNN.

Its intended VIAVI targets are OneAdvisor 800 and CellAdvisor 5G spectrum-analysis workflows. This is a conceptual integration target, not a claim of deployment, compatibility, or validation on either product.

A compact, CPU-only demonstration of a complete RF-interference data-to-insight pipeline. It produces randomized synthetic spectrogram windows, trains a small 2D PyTorch CNN, localizes the affected frequency range, and emits an inference JSON record. It is a synthetic-data MVP, not a validated production RF detector.

The five supported classes are `no_interference`, `narrowband_continuous`, `wideband_intermittent`, `periodic_impulsive`, and `adjacent_channel_leakage`. Every non-empty class randomizes frequency position, bandwidth, amplitude, duration, duty cycle, and noise floor. The CNN has three output heads: class, normalized start/stop frequency, and low/moderate/high estimated service impact.

## Setup

This project targets Python 3.14 and uses only CPU PyTorch. From the monorepo root:

```bash
uv sync --dev
```

## Generate data and inspect it

```bash
uv run --directory advisor-inference python generate_data.py --samples 1000 --output data/train.npz
uv run --directory advisor-inference python visualize_sample.py --input data/example_spectrogram.npy --output artifacts/sample_visualization.png
```

`generate_data.py` also saves a five-class overview to `artifacts/generated_examples.png` and a single `.npy` inference input to `data/example_spectrogram.npy`. These PNGs are intended for quick visual inspection.

## Train and save a model

```bash
uv run --directory advisor-inference python train.py --data data/train.npz --epochs 18 --model-output artifacts/rf_interference_cnn.pt
```

The checkpoint contains CPU-compatible model weights plus versioned input, label, normalization, seed, and validation-metric metadata. Training reports class accuracy, impact accuracy, and positive-window frequency-bound MAE, and saves the best validation epoch. Training requires a non-empty held-out validation split; use at least 10 samples with the balanced synthetic generator so every class has a repeated example.

## Run inference

```bash
uv run --directory advisor-inference python infer.py data/example_spectrogram.npy --model artifacts/rf_interference_cnn.pt

# Supply the span of the input window to add physical Hz bounds.
uv run --directory advisor-inference python infer.py data/example_spectrogram.npy --model artifacts/rf_interference_cnn.pt --frequency-start-hz 3500000000 --frequency-stop-hz 3600000000
```

Example output shape:

```json
{
  "event_type": "rf_interference",
  "class": "wideband_intermittent",
  "confidence": 0.93,
  "frequency_start_normalized": 0.42,
  "frequency_stop_normalized": 0.58,
  "estimated_service_impact": "moderate",
  "detection_status": "detected",
  "confidence_threshold": 0.6,
  "frequency_start_hz": null,
  "frequency_stop_hz": null
}
```

Frequency values are normalized to the `[0, 1]` span of the input window. Supplying both frequency-span options adds physical Hz bounds. The default confidence threshold is `0.60`; a lower-confidence non-empty class is emitted with `detection_status: "abstained"`, empty bounds, and `low` impact. For `no_interference`, status is `no_interference`, both bounds are `0.0`, and impact is `low`.

## MVP boundaries and real-data requirements

The generator is intentionally limited to isolated synthetic patterns. It does not model mixed or overlapping interference, receiver artifacts, changing gain, frequency drift, propagation effects, or out-of-distribution signals. The reported validation metrics measure only a held-out split of this synthetic distribution.

Before using this workflow beyond a demo, collect representative RF captures with the frequency span, sample rate and FFT configuration, receiver/gain characteristics, and ground-truth interference class, frequency bounds, and service impact. Evaluate separately on held-out real captures—including mixed and unseen conditions—and define deployment acceptance thresholds for class accuracy, localization error, impact accuracy, calibration, and abstention behavior.

## Test

```bash
PYTHONPATH=advisor-inference uv run pytest advisor-inference/tests
```

The tests check all output-head shapes and enforce normalized, ordered frequency bounds both in the CNN and in generated interference labels.
