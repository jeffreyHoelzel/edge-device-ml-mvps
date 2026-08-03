# Observer Root-Cause Inference MVP

> **Disclaimer:** These MVPs were created using only publicly available VIAVI product information. All datasets, measurements, telemetry, and examples are synthetic; no real, customer, confidential, proprietary, or VIAVI operational data was used. These materials are illustrative MVPs only and are not affiliated with or endorsed by VIAVI.

## Executive Summary

This MVP demonstrates an edge root-cause-ranking workflow for application-performance incidents. It converts rolling synthetic packet-derived flow summaries into an incident probability, severity, ranked likely causes, and deterministic evidence, showing how a high volume of network observations can become a compact diagnosis for an operator or downstream workflow. It does not ingest real network telemetry or establish production diagnostic accuracy.

Its intended VIAVI target is Observer GigaStor M. This is a conceptual integration target, not a claim of deployment, compatibility, or validation on the product.

A small, CPU-only PyTorch MVP that ranks likely causes of application-performance degradation from synthetic rolling packet-derived flow summaries. It deliberately uses a compact feature encoder and GRU rather than external telemetry, infrastructure, or UI.

The five synthetic incident classes are `server_delay`, `wan_congestion`, `packet_loss`, `dns_delay`, and `client_side_delay`; deterministic normal traffic is also generated. Each flow sequence contains 12 rolling summaries with network RTT, server response time, DNS time, TCP connection time, retransmission rate, packet loss, byte volume, flow count, and interface utilization.

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

The checkpoint includes the model configuration, weights, feature-normalization statistics, schema labels, seed, and validation metrics. `infer.py` and `stream_demo.py` reload it from disk with `map_location="cpu"`.

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
  "schema_version": "1.1",
  "event_type": "application_performance_incident",
  "incident_detected": true,
  "incident_probability": 0.94,
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

Incident events retain the original ranking and categorical evidence fields and add confidence, probability-mass, numeric evidence details, and flow context. Records below `--incident-threshold` emit `application_performance_status` with `incident_detected: false`, instead of a fabricated root cause. This is a synthetic operational MVP: validate the JSONL input contract and model behavior against real telemetry before production use.

## Stream demonstration

```bash
uv run --directory gigastor-m-edge-inference python stream_demo.py --model artifacts/observer_gru.pt --data data/synthetic_flows.npz --events 5
```

This prints one JSON incident event per synthetic rolling flow record, suitable for piping to an observer or log collector.

To stream externally prepared records, pass `--input records.jsonl`. Each non-empty line must contain `features` (a rolling `windows x 9` numeric array), `baseline` (nine numeric values), and an optional `context` object.
