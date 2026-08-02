# Observer Root-Cause Inference MVP

A small, CPU-only PyTorch MVP that ranks likely causes of application-performance degradation from synthetic rolling packet-derived flow summaries. It deliberately uses a compact feature encoder and GRU rather than external telemetry, infrastructure, or UI.

The five synthetic incident classes are `server_delay`, `wan_congestion`, `packet_loss`, `dns_delay`, and `client_side_delay`. Each flow sequence contains 12 rolling summaries with network RTT, server response time, DNS time, TCP connection time, retransmission rate, packet loss, byte volume, flow count, and interface utilization.

Evidence is intentionally separate from the model: the last three measurements are compared deterministically with each record's pre-incident baseline. A ratio at or above 1.25 is `elevated`, at or below 0.75 is `reduced`, and otherwise it is `normal` (or `stable` for RTT).

## Requirements and setup

Python 3.14 and [uv](https://docs.astral.sh/uv/) are required. All runtime and development dependencies are declared in the monorepo root `pyproject.toml`; the model is CPU-only and never moves tensors to CUDA.

```bash
uv sync --dev
```

## Generate training data

```bash
uv run --directory gigastor-m-edge-inference python generate_data.py --output data/synthetic_flows.npz --samples 2000 --seed 7
```

## Train and save the model

```bash
uv run --directory gigastor-m-edge-inference python train.py --data data/synthetic_flows.npz --model artifacts/observer_gru.pt --epochs 30
```

The checkpoint includes the model configuration, weights, and feature-normalization statistics. `infer.py` and `stream_demo.py` reload it from disk with `map_location="cpu"`.

## Test

```bash
PYTHONPATH=gigastor-m-edge-inference uv run pytest gigastor-m-edge-inference/tests
```

The tests verify that root-cause probabilities sum to approximately one, rankings are descending, and saved checkpoints reload without changing logits.

## Infer one rolling flow record

```bash
uv run --directory gigastor-m-edge-inference python infer.py --model artifacts/observer_gru.pt --data data/synthetic_flows.npz --index 0 --top-k 3
```

The command writes structured JSON to stdout, for example:

```json
{
  "event_type": "application_performance_incident",
  "severity": "major",
  "root_cause_ranking": [
    {"cause": "server_delay", "probability": 0.72},
    {"cause": "wan_congestion", "probability": 0.17},
    {"cause": "packet_loss", "probability": 0.07}
  ],
  "evidence": {
    "server_response_time": "elevated",
    "network_rtt": "stable",
    "tcp_retransmissions": "normal"
  }
}
```

## Stream demonstration

```bash
uv run --directory gigastor-m-edge-inference python stream_demo.py --model artifacts/observer_gru.pt --data data/synthetic_flows.npz --events 5
```

This prints one JSON incident event per synthetic rolling flow record, suitable for piping to an observer or log collector.
