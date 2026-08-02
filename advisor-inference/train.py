"""Train and save the compact RF interference CNN on generated data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import RFInterferenceCNN


def normalize_per_window(spectrograms: torch.Tensor) -> torch.Tensor:
    mean = spectrograms.mean(dim=(2, 3), keepdim=True)
    std = spectrograms.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
    return (spectrograms - mean) / std


def evaluate(model: RFInterferenceCNN, loader: DataLoader[tuple[torch.Tensor, ...]], device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for features, classes, _, _ in loader:
            output = model(normalize_per_window(features.to(device)))
            correct += int((output["class_logits"].argmax(dim=1).cpu() == classes).sum())
            total += len(classes)
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RF interference classifier on a .npz dataset.")
    parser.add_argument("--data", type=Path, default=Path("data/train.npz"))
    parser.add_argument("--model-output", type=Path, default=Path("artifacts/rf_interference_cnn.pt"))
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    data = np.load(args.data)
    features = torch.from_numpy(data["spectrograms"].astype(np.float32))
    classes = torch.from_numpy(data["classes"].astype(np.int64))
    bounds = torch.from_numpy(data["bounds"].astype(np.float32))
    impacts = torch.from_numpy(data["impacts"].astype(np.int64))
    split = max(1, int(len(features) * 0.8))
    train_set = TensorDataset(features[:split], classes[:split], bounds[:split], impacts[:split])
    validation_set = TensorDataset(features[split:], classes[split:], bounds[split:], impacts[split:])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)

    model = RFInterferenceCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    class_loss = nn.CrossEntropyLoss()
    bound_loss = nn.SmoothL1Loss(reduction="none")
    impact_loss = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_features, batch_classes, batch_bounds, batch_impacts in train_loader:
            batch_features = normalize_per_window(batch_features.to(device))
            batch_classes = batch_classes.to(device)
            batch_bounds = batch_bounds.to(device)
            batch_impacts = batch_impacts.to(device)
            output = model(batch_features)
            # Localization has no meaningful target for no-interference windows.
            positive_mask = (batch_classes != 0).float().unsqueeze(1)
            localization = (bound_loss(output["frequency_bounds"], batch_bounds) * positive_mask).sum()
            localization = localization / positive_mask.sum().clamp_min(1.0)
            loss = class_loss(output["class_logits"], batch_classes) + 0.7 * localization
            loss = loss + 0.35 * impact_loss(output["impact_logits"], batch_impacts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(batch_classes)
        validation_accuracy = evaluate(model, validation_loader, device)
        print(f"epoch {epoch:02d}/{args.epochs}: loss={running_loss / len(train_set):.4f}, val_class_accuracy={validation_accuracy:.3f}")

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "input_shape": [1, 64, 128], "normalization": "per_window_zscore"},
        args.model_output,
    )
    print(f"Saved CPU model checkpoint to {args.model_output}")


if __name__ == "__main__":
    main()
