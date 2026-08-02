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
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
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
    loader = DataLoader(
        TensorDataset(features[train_indices], causes[train_indices], severity[train_indices], incident[train_indices]),
        batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),
    )

    model.train()
    validation_metrics: dict[str, float] = {}
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
                validation_logits, validation_severity, validation_incident = model(features[validation_indices])
                actual_incident = incident[validation_indices].bool()
                predicted_incident = validation_incident.argmax(dim=1).bool()
                true_positive = (predicted_incident & actual_incident).sum().item()
                precision = true_positive / max(predicted_incident.sum().item(), 1)
                recall = true_positive / max(actual_incident.sum().item(), 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-12)
                if actual_incident.any():
                    cause_accuracy = (validation_logits[actual_incident].argmax(dim=1) == causes[validation_indices][actual_incident]).float().mean().item()
                    severity_accuracy = (validation_severity[actual_incident].argmax(dim=1) == severity[validation_indices][actual_incident]).float().mean().item()
                else:
                    cause_accuracy = 0.0
                    severity_accuracy = 0.0
                incident_probability = torch.softmax(validation_incident, dim=1)[:, 1]
                brier = ((incident_probability - incident[validation_indices].float()) ** 2).mean().item()
                validation_metrics = {"incident_precision": precision, "incident_recall": recall, "incident_f1": f1, "incident_brier": brier, "cause_accuracy": cause_accuracy, "severity_accuracy": severity_accuracy}
            model.train()
            print(f"epoch={epoch + 1:02d} loss={running_loss / len(train_indices):.4f} validation_incident_f1={f1:.3f} validation_cause_accuracy={cause_accuracy:.3f} validation_severity_accuracy={severity_accuracy:.3f}")

    model.eval()
    save_checkpoint(model, model_path, {
        "feature_names": raw["feature_names"].tolist(),
        "cause_names": raw["cause_names"].tolist(),
        "severity_names": raw["severity_names"].tolist(),
        "seed": seed,
        "validation_metrics": validation_metrics,
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
