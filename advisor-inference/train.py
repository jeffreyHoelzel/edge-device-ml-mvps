"""Train and save the compact RF interference CNN on generated data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from contract import (
    CHECKPOINT_FORMAT_VERSION,
    CLASS_NAMES,
    FREQUENCY_BINS,
    IMPACT_NAMES,
    INPUT_SHAPE,
    NORMALIZATION_NAME,
    TIME_BINS,
    normalize_per_window,
)
from model import NUM_CLASSES, NUM_IMPACT_LEVELS, RFInterferenceCNN


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite value greater than 0")
    return parsed


def load_training_data(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load and validate the fixed RF training-data contract."""
    loaded = np.load(path)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError("Training data must be a .npz file with spectrograms, classes, bounds, and impacts")
    try:
        required = {"spectrograms", "classes", "bounds", "impacts"}
        missing = required.difference(loaded.files)
        if missing:
            raise ValueError(f"Training data is missing required arrays: {', '.join(sorted(missing))}")
        spectrograms = loaded["spectrograms"]
        classes = loaded["classes"]
        bounds = loaded["bounds"]
        impacts = loaded["impacts"]
    finally:
        loaded.close()

    count = len(spectrograms)
    if count < 1 or spectrograms.shape != (count, 1, TIME_BINS, FREQUENCY_BINS):
        raise ValueError(f"spectrograms must have shape (N, 1, {TIME_BINS}, {FREQUENCY_BINS}) with N at least 1")
    if classes.shape != (count,) or bounds.shape != (count, 2) or impacts.shape != (count,):
        raise ValueError("classes, bounds, and impacts must align with the spectrogram count")
    if not np.issubdtype(spectrograms.dtype, np.number) or not np.issubdtype(bounds.dtype, np.number):
        raise ValueError("spectrograms and bounds must be numeric arrays")
    if not np.issubdtype(classes.dtype, np.integer) or not np.issubdtype(impacts.dtype, np.integer):
        raise ValueError("classes and impacts must be integer arrays")
    if not np.isfinite(spectrograms).all() or not np.isfinite(bounds).all():
        raise ValueError("spectrograms and bounds must contain only finite values")
    if np.any(classes < 0) or np.any(classes >= 5):
        raise ValueError("classes must contain IDs from 0 to 4")
    if np.any(impacts < 0) or np.any(impacts >= 3):
        raise ValueError("impacts must contain IDs from 0 to 2")
    if np.any(bounds < 0.0) or np.any(bounds > 1.0) or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("bounds must be ordered and normalized to [0, 1]")
    return (
        torch.from_numpy(spectrograms.astype(np.float32)),
        torch.from_numpy(classes.astype(np.int64)),
        torch.from_numpy(bounds.astype(np.float32)),
        torch.from_numpy(impacts.astype(np.int64)),
    )


def stratified_split_indices(
    classes: torch.Tensor, seed: int, validation_fraction: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a deterministic split while retaining singleton classes in training."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    labels = classes.detach().cpu().numpy()
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        validation_count = 0 if len(indices) == 1 else min(len(indices) - 1, max(1, int(len(indices) * validation_fraction)))
        validation_indices.extend(indices[:validation_count].tolist())
        train_indices.extend(indices[validation_count:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return torch.tensor(train_indices, dtype=torch.long), torch.tensor(validation_indices, dtype=torch.long)


def evaluate(
    model: RFInterferenceCNN, loader: DataLoader[tuple[torch.Tensor, ...]], device: torch.device
) -> dict[str, float | None]:
    model.eval()
    class_correct = impact_correct = total = positive_count = 0
    bounds_absolute_error = 0.0
    with torch.no_grad():
        for features, classes, bounds, impacts in loader:
            output = model(normalize_per_window(features.to(device)))
            class_predictions = output["class_logits"].argmax(dim=1).cpu()
            impact_predictions = output["impact_logits"].argmax(dim=1).cpu()
            class_correct += int((class_predictions == classes).sum())
            impact_correct += int((impact_predictions == impacts).sum())
            positive_mask = classes != 0
            if positive_mask.any():
                predicted_bounds = output["frequency_bounds"].cpu()[positive_mask]
                bounds_absolute_error += float(torch.abs(predicted_bounds - bounds[positive_mask]).sum())
                positive_count += int(positive_mask.sum())
            total += len(classes)
    return {
        "class_accuracy": class_correct / max(total, 1),
        "impact_accuracy": impact_correct / max(total, 1),
        "bounds_mae": bounds_absolute_error / (positive_count * 2) if positive_count else None,
    }


def is_better_metrics(candidate: dict[str, float | None], best: dict[str, float | None] | None) -> bool:
    """Prefer class accuracy, then lower localization error, retaining the first tie."""
    if best is None:
        return True
    if candidate["class_accuracy"] != best["class_accuracy"]:
        return candidate["class_accuracy"] > best["class_accuracy"]
    candidate_mae = float("inf") if candidate["bounds_mae"] is None else candidate["bounds_mae"]
    best_mae = float("inf") if best["bounds_mae"] is None else best["bounds_mae"]
    return candidate_mae < best_mae


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RF interference classifier on a .npz dataset.")
    parser.add_argument("--data", type=Path, default=Path("data/train.npz"))
    parser.add_argument("--model-output", type=Path, default=Path("artifacts/rf_interference_cnn.pt"))
    parser.add_argument("--epochs", type=positive_int, default=18)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--learning-rate", type=positive_float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    features, classes, bounds, impacts = load_training_data(args.data)
    train_indices, validation_indices = stratified_split_indices(classes, args.seed)
    singleton_ids = sorted(int(class_id) for class_id in torch.unique(classes) if int((classes == class_id).sum()) == 1)
    if singleton_ids:
        singleton_names = ", ".join(str(class_id) for class_id in singleton_ids)
        print(f"validation note: singleton class IDs remain training-only: {singleton_names}")
    train_set = TensorDataset(
        features[train_indices], classes[train_indices], bounds[train_indices], impacts[train_indices]
    )
    validation_set = TensorDataset(
        features[validation_indices], classes[validation_indices], bounds[validation_indices], impacts[validation_indices]
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=loader_generator)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)

    model = RFInterferenceCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    class_loss = nn.CrossEntropyLoss()
    bound_loss = nn.SmoothL1Loss(reduction="none")
    impact_loss = nn.CrossEntropyLoss()
    best_metrics: dict[str, float | None] | None = None
    best_epoch = 0
    best_model_state: dict[str, torch.Tensor] | None = None

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
        validation_metrics = evaluate(model, validation_loader, device)
        if is_better_metrics(validation_metrics, best_metrics):
            best_metrics = validation_metrics
            best_epoch = epoch
            best_model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        bounds_mae = validation_metrics["bounds_mae"]
        bounds_text = "n/a" if bounds_mae is None else f"{bounds_mae:.4f}"
        print(
            f"epoch {epoch:02d}/{args.epochs}: loss={running_loss / len(train_set):.4f}, "
            f"val_class_accuracy={validation_metrics['class_accuracy']:.3f}, "
            f"val_impact_accuracy={validation_metrics['impact_accuracy']:.3f}, val_bounds_mae={bounds_text}"
        )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state": best_model_state,
            "input_shape": list(INPUT_SHAPE),
            "normalization": NORMALIZATION_NAME,
            "model_config": {"num_classes": NUM_CLASSES, "num_impact_levels": NUM_IMPACT_LEVELS},
            "class_names": list(CLASS_NAMES),
            "impact_names": list(IMPACT_NAMES),
            "training_seed": args.seed,
            "best_epoch": best_epoch,
            "validation_metrics": best_metrics,
        },
        args.model_output,
    )
    print(f"Saved best CPU model checkpoint from epoch {best_epoch} to {args.model_output}")


if __name__ == "__main__":
    main()
