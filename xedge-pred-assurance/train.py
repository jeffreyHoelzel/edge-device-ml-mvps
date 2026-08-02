"""Train and save the CPU-only predictive assurance model."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from generate_data import make_dataset
from model import PredictiveAssuranceModel, save_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not torch.any(mask):
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(logits[mask], labels[mask])


def loss_for_batch(outputs: dict[str, torch.Tensor], incident: torch.Tensor, cause: torch.Tensor, severity: torch.Tensor) -> torch.Tensor:
    incident_loss = nn.functional.binary_cross_entropy_with_logits(outputs["incident_logits"], incident)
    mask = incident > 0.5
    cause_loss = masked_cross_entropy(outputs["cause_logits"], cause, mask)
    severity_loss = masked_cross_entropy(outputs["severity_logits"], severity, mask)
    return incident_loss + 0.70 * cause_loss + 0.40 * severity_loss


def load_or_generate(path: Path, samples: int, sequence_length: int, seed: int) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path) as stored:
            return {key: stored[key] for key in ("x", "incident", "cause", "severity")}
    data = make_dataset(samples, sequence_length, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_train.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.pt"))
    parser.add_argument("--samples", type=int, default=2400)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    set_seed(args.seed)
    data = load_or_generate(args.data, args.samples, args.sequence_length, args.seed)
    dataset = TensorDataset(
        torch.from_numpy(data["x"]),
        torch.from_numpy(data["incident"]),
        torch.from_numpy(data["cause"]),
        torch.from_numpy(data["severity"]),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    model = PredictiveAssuranceModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, incident, cause, severity in loader:
            optimizer.zero_grad()
            loss = loss_for_batch(model(x), incident, cause, severity)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
        print(f"epoch {epoch:02d}/{args.epochs}: loss={total_loss / len(dataset):.4f}")

    save_model(model, args.output)
    print(f"Saved CPU model to {args.output}")


if __name__ == "__main__":
    main()
