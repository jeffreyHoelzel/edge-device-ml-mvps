"""Run a saved model and emit an interpretable incident JSON record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from generate_data import CAUSES, FEATURE_NAMES, SEVERITIES
from data_contract import load_dataset
from model import RootCauseGRU, load_checkpoint


def _comparison(recent: float, baseline: float, stable_word: str = "normal") -> str:
    """Deterministically label a metric from its ratio to its supplied baseline."""
    ratio = recent / max(baseline, 1e-6)
    if ratio >= 1.25:
        return "elevated"
    if ratio <= 0.75:
        return "reduced"
    return stable_word


def build_event(
    model: RootCauseGRU, features: np.ndarray, baseline: np.ndarray, top_k: int = 5, incident_threshold: float = 0.5,
    root_cause_threshold: float = 0.4, context: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a ranked prediction plus baseline-derived, non-model evidence."""
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"features must have shape (windows, {len(FEATURE_NAMES)})")
    if features.shape[0] < 3:
        raise ValueError("features must contain at least 3 windows for recent-window evidence")
    if baseline.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"baseline must have shape ({len(FEATURE_NAMES)},)")
    cause_names = tuple(model.metadata.get("cause_names", CAUSES))
    severity_names = tuple(model.metadata.get("severity_names", SEVERITIES))
    if not 1 <= top_k <= len(cause_names):
        raise ValueError(f"top_k must be between 1 and {len(cause_names)}")
    if not 0.0 <= incident_threshold <= 1.0:
        raise ValueError("incident_threshold must be between 0 and 1")
    if not 0.0 <= root_cause_threshold <= 1.0:
        raise ValueError("root_cause_threshold must be between 0 and 1")
    with torch.no_grad():
        cause_logits, severity_logits, incident_logits = model(torch.tensor(features[None], dtype=torch.float32))
        probabilities = torch.softmax(cause_logits[0], dim=0).cpu().numpy()
        severity = severity_names[int(severity_logits[0].argmax())]
        incident_probability = float(torch.softmax(incident_logits[0], dim=0)[1])
        severity_probability = float(torch.softmax(severity_logits[0], dim=0).max())
    ordered = np.argsort(probabilities)[::-1][:top_k]
    recent = features[-3:].mean(axis=0)  # a fixed, explicit recent-measurement window
    observed = dict(zip(FEATURE_NAMES, recent, strict=True))
    reference = dict(zip(FEATURE_NAMES, baseline, strict=True))
    evidence = {
        "server_response_time": _comparison(observed["server_response_time"], reference["server_response_time"]),
        "network_rtt": _comparison(observed["network_rtt"], reference["network_rtt"], stable_word="stable"),
        "dns_time": _comparison(observed["dns_time"], reference["dns_time"]),
        "tcp_connection_time": _comparison(observed["tcp_connection_time"], reference["tcp_connection_time"]),
        "tcp_retransmissions": _comparison(observed["retransmission_rate"], reference["retransmission_rate"]),
        "packet_loss": _comparison(observed["packet_loss"], reference["packet_loss"]),
        "interface_utilization": _comparison(observed["interface_utilization"], reference["interface_utilization"]),
    }
    evidence_details = {
        name: {
            "recent": round(float(observed[name]), 6),
            "baseline": round(float(reference[name]), 6),
            "ratio": round(float(observed[name] / max(reference[name], 1e-6)), 6),
            "state": _comparison(observed[name], reference[name], "stable" if name == "network_rtt" else "normal"),
        }
        for name in FEATURE_NAMES
    }
    detected = incident_probability >= incident_threshold
    event = {
        "schema_version": "1.1",
        "event_type": "application_performance_incident" if detected else "application_performance_status",
        "incident_detected": detected,
        "incident_probability": round(incident_probability, 6),
        "root_cause_confidence": round(float(probabilities.max()), 6),
        "root_cause_status": "confident" if probabilities.max() >= root_cause_threshold else "uncertain",
        "severity_probability": round(severity_probability, 6),
        "root_cause_probability_mass": round(float(probabilities[ordered].sum()), 6) if detected else 0.0,
        "severity": severity if detected else "none",
        "root_cause_ranking": [
            {"cause": cause_names[int(index)], "probability": round(float(probabilities[index]), 6)} for index in ordered
        ] if detected else [],
        "evidence": evidence,
        "evidence_details": evidence_details,
    }
    if context is not None:
        event["context"] = context
    return event


def load_record(data_path: Path, index: int) -> tuple[np.ndarray, np.ndarray]:
    data = load_dataset(data_path)
    if not 0 <= index < len(data["features"]):
        raise IndexError(f"index must be between 0 and {len(data['features']) - 1}")
    return data["features"][index], data["baselines"][index]


def load_record_with_context(data_path: Path, index: int) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    data = load_dataset(data_path)
    if not 0 <= index < len(data["features"]):
        raise IndexError(f"index must be between 0 and {len(data['features']) - 1}")
    context = {
        "flow_id": str(data["flow_ids"][index]),
        "device_id": str(data["device_ids"][index]),
        "interface": str(data["interfaces"][index]),
        "window_end": str(data["window_ends"][index]),
    }
    return data["features"][index], data["baselines"][index], context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/observer_gru.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_flows.npz"))
    parser.add_argument("--index", type=int, default=0, help="record index in the NPZ data set")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--incident-threshold", type=float, default=0.5)
    parser.add_argument("--root-cause-threshold", type=float, default=0.4)
    args = parser.parse_args()
    features, baseline, context = load_record_with_context(args.data, args.index)
    print(json.dumps(build_event(load_checkpoint(args.model), features, baseline, args.top_k, args.incident_threshold, args.root_cause_threshold, context), indent=2))


if __name__ == "__main__":
    main()
