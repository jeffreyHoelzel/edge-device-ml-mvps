"""Simulate a rolling observer by printing one inference event at a time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infer import build_event, load_jsonl_records, load_record_with_context
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
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--data", type=Path)
    inputs.add_argument("--input", type=Path, help="JSONL records with features, baseline, and optional context")
    parser.add_argument("--events", type=int, default=5)
    parser.add_argument("--incident-threshold", type=float, default=0.5)
    parser.add_argument("--root-cause-threshold", type=float, default=0.4)
    args = parser.parse_args()
    model = load_checkpoint(args.model)
    if args.input:
        records = load_jsonl_records(args.input)
    else:
        data_path = args.data or Path("data/synthetic_flows.npz")
        records = [load_record_with_context(data_path, index) for index in range(len(load_dataset(data_path)["features"]))]
    validate_event_count(args.events, len(records))
    for features, baseline, context in records[:args.events]:
        print(json.dumps(build_event(model, features, baseline, incident_threshold=args.incident_threshold, root_cause_threshold=args.root_cause_threshold, context=context), separators=(",", ":")))


if __name__ == "__main__":
    main()
