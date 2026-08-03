"""Train the CPU-only synthetic DAS risk model and save a portable checkpoint."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from checkpoint import create_checkpoint
from evaluate import evaluate_model, quality_gate_failures
from generate_data import DEFAULT_HORIZON_SECONDS, MAX_SPEED_M_PER_MIN, SyntheticDASDataset
from model import DASRiskModel


def loss_for(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    mse = nn.MSELoss()
    return (
        nn.functional.cross_entropy(outputs["event_logits"], batch["class_index"])
        + nn.functional.cross_entropy(outputs["direction_logits"], batch["direction_index"])
        + mse(outputs["speed_m_per_min"] / MAX_SPEED_M_PER_MIN, batch["speed"])
        + nn.functional.binary_cross_entropy(outputs["escalation_probability"], batch["escalation"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon-seconds", type=int, default=DEFAULT_HORIZON_SECONDS)
    parser.add_argument("--validation-samples", type=int, default=200)
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument("--no-enforce-quality", action="store_true")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/das_risk_model.pt"))
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.horizon_seconds <= 0:
        parser.error("--horizon-seconds must be positive")
    if args.validation_samples <= 0:
        parser.error("--validation-samples must be positive")
    validation_seed = args.validation_seed if args.validation_seed is not None else args.seed + 1

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    loader = DataLoader(
        SyntheticDASDataset(args.samples, seed=args.seed, horizon_seconds=args.horizon_seconds),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_dataset = SyntheticDASDataset(
        args.validation_samples, seed=validation_seed, horizon_seconds=args.horizon_seconds
    )
    model = DASRiskModel().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    best_state = None
    best_metrics = None
    best_validation_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(batch["signal"])
            loss = loss_for(outputs, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch["signal"].size(0)
        metrics = evaluate_model(model, validation_dataset, args.batch_size)
        validation_loss = (
            1 - metrics["event_accuracy"]
            + 1 - metrics["direction_accuracy"]
            + metrics["speed_mae_m_per_min"] / MAX_SPEED_M_PER_MIN
            + metrics["risk_brier_score"]
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
        print(
            f"epoch={epoch:02d} loss={total_loss / len(loader.dataset):.4f} "
            f"validation_event_accuracy={metrics['event_accuracy']:.3f}"
        )

    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    failures = quality_gate_failures(best_metrics)
    if failures and not args.no_enforce_quality:
        parser.error("quality gates failed: " + "; ".join(failures))
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        create_checkpoint(
            model,
            seed=args.seed,
            horizon_seconds=args.horizon_seconds,
            evaluation_metrics=best_metrics,
        ),
        args.model_path,
    )
    print(f"saved CPU checkpoint to {args.model_path}")


if __name__ == "__main__":
    main()
