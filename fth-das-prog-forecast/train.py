"""Train the CPU-only synthetic DAS risk model and save a portable checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

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

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    loader = DataLoader(
        SyntheticDASDataset(args.samples, seed=args.seed, horizon_seconds=args.horizon_seconds),
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = DASRiskModel().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(batch["signal"])
            loss = loss_for(outputs, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch["signal"].size(0)
        print(f"epoch={epoch:02d} loss={total_loss / len(loader.dataset):.4f}")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "model_config": {"spatial_channels": 16, "hidden_size": 32}}, args.model_path)
    print(f"saved CPU checkpoint to {args.model_path}")


if __name__ == "__main__":
    main()
