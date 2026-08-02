"""Simulate a rolling observer by printing one inference event at a time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infer import build_event, load_record
from model import load_checkpoint
from data_contract import load_dataset


def validate_event_count(events: int, available: int) -> None:
    """Ensure a requested stream length is representable by the input data."""
    if events < 0:
        raise ValueError("events must be zero or greater")
    if events > available:
        raise ValueError(f"events must not exceed available records ({available})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/observer_gru.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_flows.npz"))
    parser.add_argument("--events", type=int, default=5)
    args = parser.parse_args()
    model = load_checkpoint(args.model)
    validate_event_count(args.events, len(load_dataset(args.data)["features"]))
    for index in range(args.events):
        features, baseline = load_record(args.data, index)
        print(json.dumps(build_event(model, features, baseline), separators=(",", ":")))


if __name__ == "__main__":
    main()
