"""Run either simulated or JSON Lines KPI-arrival forecasts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

from generate_data import CAUSE_NAMES, generate_sequence
from model import load_model
from streaming import forecast_event, run_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.pt"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--simulate", action="store_true")
    mode.add_argument("--stdin", action="store_true")
    parser.add_argument("--cause", choices=CAUSE_NAMES, default="rf_interference")
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"{args.model} was not found; run train.py first")
    loaded = load_model(args.model)
    if args.stdin:
        run_jsonl(loaded, sys.stdin, sys.stdout)
        return
    contract = loaded.contract
    raw_measurements = generate_sequence(contract.sequence_length + 12, CAUSE_NAMES.index(args.cause), rng=np.random.default_rng(args.seed))
    buffer: deque[np.ndarray] = deque(maxlen=contract.sequence_length)
    for measurement in raw_measurements:
        buffer.append(measurement)
        if len(buffer) < contract.sequence_length:
            continue
        raw = np.asarray(buffer, dtype=np.float32)
        x = ((raw - contract.feature_mean) / contract.feature_std).astype(np.float32)[None, ...]
        with torch.inference_mode():
            output = loaded.model(torch.from_numpy(x))
            probability = torch.sigmoid(output["incident_logits"])[0].item()
            cause = int(output["cause_logits"].argmax(dim=1)[0])
            severity = int(output["severity_logits"].argmax(dim=1)[0])
            cause_confidence = torch.softmax(output["cause_logits"], dim=1).max(dim=1).values[0].item()
            severity_confidence = torch.softmax(output["severity_logits"], dim=1).max(dim=1).values[0].item()
        print(json.dumps(forecast_event(probability, cause, severity, contract, cause_confidence=cause_confidence, severity_confidence=severity_confidence), sort_keys=True))


if __name__ == "__main__":
    main()
