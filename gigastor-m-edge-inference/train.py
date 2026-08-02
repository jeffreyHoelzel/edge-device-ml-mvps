"""Train the root-cause and severity heads from synthetic flow summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import RootCauseGRU, save_checkpoint
from data_contract import load_dataset


def train(data_path: Path, model_path: Path, epochs: int = 30, batch_size: int = 64, seed: int = 7) -> RootCauseGRU:
    """Train on CPU and save a reloadable checkpoint; returns the trained model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    raw = load_dataset(data_path)
    features = torch.tensor(raw["features"], dtype=torch.float32)
    causes = torch.tensor(raw["causes"], dtype=torch.long)
    severity = torch.tensor(raw["severity"], dtype=torch.long)
    incident = torch.tensor(raw["incident"], dtype=torch.long)
    permutation = torch.randperm(len(features))
    split = int(len(features) * 0.85)
    train_indices, validation_indices = permutation[:split], permutation[split:]
    model = RootCauseGRU(input_size=features.shape[-1], cause_count=len(raw["cause_names"]), severity_count=len(raw["severity_names"]))
    model.set_normalization(features[train_indices].mean(dim=(0, 1)), features[train_indices].std(dim=(0, 1)))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(features[train_indices], causes[train_indices], severity[train_indices], incident[train_indices]), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_features, batch_causes, batch_severity, batch_incident in loader:
            optimizer.zero_grad()
            cause_logits, severity_logits, incident_logits = model(batch_features)
            incident_mask = batch_incident.bool()
            incident_loss = criterion(incident_logits, batch_incident)
            if incident_mask.any():
                cause_loss = criterion(cause_logits[incident_mask], batch_causes[incident_mask])
                severity_loss = criterion(severity_logits[incident_mask], batch_severity[incident_mask])
                loss = cause_loss + 0.45 * severity_loss + incident_loss
            else:
                loss = incident_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_features)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                validation_logits, _, _ = model(features[validation_indices])
                accuracy = (validation_logits.argmax(dim=1) == causes[validation_indices]).float().mean().item()
            model.train()
            print(f"epoch={epoch + 1:02d} loss={running_loss / len(train_indices):.4f} validation_cause_accuracy={accuracy:.3f}")

    model.eval()
    save_checkpoint(model, model_path, {
        "feature_names": raw["feature_names"].tolist(),
        "cause_names": raw["cause_names"].tolist(),
        "severity_names": raw["severity_names"].tolist(),
        "seed": seed,
    })
    print(f"Saved CPU checkpoint to {model_path}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_flows.npz"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/observer_gru.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    train(args.data, args.model, args.epochs, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
