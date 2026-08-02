"""Train and save the CPU-only predictive assurance model."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from generate_data import (
    CAUSE_NAMES,
    DATASET_FORMAT_VERSION,
    FEATURE_MEAN,
    FEATURE_NAMES,
    FEATURE_STD,
    FORECAST_HORIZON_SECONDS,
    SAMPLE_INTERVAL_SECONDS,
    SEVERITY_NAMES,
    make_dataset,
    save_dataset,
)
from model import PredictiveAssuranceModel, TrainingContract, save_model


ALERT_THRESHOLD = 0.70
EVALUATION_FRACTION = 0.20
QUALITY_GATES = {
    "incident_auroc": 0.95,
    "incident_precision": 0.85,
    "incident_recall": 0.85,
    "cause_accuracy": 0.80,
    "severity_accuracy": 0.70,
    "expected_calibration_error": 0.10,
}


@dataclass(frozen=True)
class EvaluationMetrics:
    incident_auroc: float
    incident_average_precision: float
    incident_precision: float
    incident_recall: float
    cause_accuracy: float
    severity_accuracy: float
    brier_score: float
    expected_calibration_error: float


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


def validate_dataset(data: dict[str, np.ndarray], metadata: dict[str, object] | None = None) -> None:
    required = ("x", "incident", "cause", "severity")
    if set(data) != set(required):
        raise ValueError(f"dataset must contain exactly: {', '.join(required)}")
    x = data["x"]
    if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] < 8 or x.shape[2] != len(FEATURE_NAMES):
        raise ValueError("dataset x must have shape [samples, time>=8, 7]")
    if not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
        raise ValueError("dataset x must contain finite floating-point values")
    for name, classes in (("incident", 2), ("cause", len(CAUSE_NAMES)), ("severity", len(SEVERITY_NAMES))):
        labels = data[name]
        if labels.ndim != 1 or len(labels) != len(x) or not np.issubdtype(labels.dtype, np.number):
            raise ValueError(f"dataset {name} must be a numeric vector matching x")
        if not np.isfinite(labels).all() or not np.all(labels == labels.astype(np.int64)):
            raise ValueError(f"dataset {name} must contain finite integer labels")
        if np.any(labels < 0) or np.any(labels >= classes):
            raise ValueError(f"dataset {name} labels are out of range")
    if metadata is not None:
        if metadata.get("format_version") != DATASET_FORMAT_VERSION:
            raise ValueError("dataset format is unsupported; regenerate the dataset")
        if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("dataset feature order does not match this model")
        if metadata.get("sequence_length") != x.shape[1]:
            raise ValueError("dataset metadata sequence length does not match x")


def load_or_generate(path: Path, samples: int, sequence_length: int, seed: int) -> dict[str, np.ndarray]:
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as stored:
                data = {key: stored[key] for key in ("x", "incident", "cause", "severity")}
                metadata = json.loads(str(stored["metadata_json"].item()))
        except (KeyError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{path} is not a valid predictive-assurance dataset; regenerate it") from error
        validate_dataset(data, metadata)
        return data
    data = make_dataset(samples, sequence_length, seed)
    validate_dataset(data)
    save_dataset(path, data)
    return data


def validate_training_options(args: argparse.Namespace) -> None:
    if args.samples < 8:
        raise ValueError("--samples must be at least 8")
    if args.sequence_length < 8:
        raise ValueError("--sequence-length must be at least 8")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")


def split_indices(data: dict[str, np.ndarray], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically split each normal/incident diagnosis stratum into holdout."""
    incident = data["incident"].astype(bool)
    strata = np.where(incident, data["cause"] * len(SEVERITY_NAMES) + data["severity"], 12)
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    evaluation_parts: list[np.ndarray] = []
    for stratum in np.unique(strata):
        indices = np.flatnonzero(strata == stratum)
        indices = rng.permutation(indices)
        holdout = min(max(1, round(len(indices) * EVALUATION_FRACTION)), len(indices) - 1)
        if holdout < 1:
            raise ValueError("dataset needs at least two examples in every diagnosis stratum")
        evaluation_parts.append(indices[:holdout])
        train_parts.append(indices[holdout:])
    return np.concatenate(train_parts), np.concatenate(evaluation_parts)


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    errors = []
    for lower, upper in zip(np.linspace(0.0, 1.0, bins, endpoint=False), np.linspace(1.0 / bins, 1.0, bins)):
        in_bin = (probabilities >= lower) & ((probabilities < upper) if upper < 1.0 else (probabilities <= upper))
        if np.any(in_bin):
            errors.append(abs(probabilities[in_bin].mean() - labels[in_bin].mean()) * in_bin.mean())
    return float(sum(errors))


def evaluate_model(model: PredictiveAssuranceModel, data: dict[str, np.ndarray], indices: np.ndarray) -> EvaluationMetrics:
    model.eval()
    with torch.inference_mode():
        output = model(torch.from_numpy(data["x"][indices]))
        probabilities = torch.sigmoid(output["incident_logits"]).numpy()
        cause_predictions = output["cause_logits"].argmax(dim=1).numpy()
        severity_predictions = output["severity_logits"].argmax(dim=1).numpy()
    incident = data["incident"][indices].astype(np.int64)
    positive = incident == 1
    return EvaluationMetrics(
        incident_auroc=float(roc_auc_score(incident, probabilities)),
        incident_average_precision=float(average_precision_score(incident, probabilities)),
        incident_precision=float(precision_score(incident, probabilities >= ALERT_THRESHOLD, zero_division=0)),
        incident_recall=float(recall_score(incident, probabilities >= ALERT_THRESHOLD, zero_division=0)),
        cause_accuracy=float(accuracy_score(data["cause"][indices][positive], cause_predictions[positive])),
        severity_accuracy=float(accuracy_score(data["severity"][indices][positive], severity_predictions[positive])),
        brier_score=float(brier_score_loss(incident, probabilities)),
        expected_calibration_error=expected_calibration_error(probabilities, incident),
    )


def enforce_quality_gates(metrics: EvaluationMetrics) -> None:
    failures = [
        f"{name}={getattr(metrics, name):.3f} (minimum {minimum:.3f})"
        for name, minimum in QUALITY_GATES.items()
        if getattr(metrics, name) < minimum if name != "expected_calibration_error"
    ]
    if metrics.expected_calibration_error > QUALITY_GATES["expected_calibration_error"]:
        failures.append(
            "expected_calibration_error="
            f"{metrics.expected_calibration_error:.3f} (maximum {QUALITY_GATES['expected_calibration_error']:.3f})"
        )
    if failures:
        raise ValueError("model failed quality gates: " + "; ".join(failures))


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
    validate_training_options(args)
    set_seed(args.seed)
    data = load_or_generate(args.data, args.samples, args.sequence_length, args.seed)
    train_indices, evaluation_indices = split_indices(data, args.seed)
    dataset = TensorDataset(
        torch.from_numpy(data["x"][train_indices]),
        torch.from_numpy(data["incident"][train_indices]),
        torch.from_numpy(data["cause"][train_indices]),
        torch.from_numpy(data["severity"][train_indices]),
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

    metrics = evaluate_model(model, data, evaluation_indices)
    enforce_quality_gates(metrics)
    print("evaluation=" + json.dumps(asdict(metrics), sort_keys=True))
    contract = TrainingContract(
        feature_names=FEATURE_NAMES,
        feature_mean=tuple(float(value) for value in FEATURE_MEAN),
        feature_std=tuple(float(value) for value in FEATURE_STD),
        cause_names=CAUSE_NAMES,
        severity_names=SEVERITY_NAMES,
        sequence_length=data["x"].shape[1],
        sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
        forecast_horizon_seconds=FORECAST_HORIZON_SECONDS,
    )
    save_model(model, args.output, contract, evaluation_metrics=asdict(metrics))
    print(f"Saved CPU model to {args.output}")


if __name__ == "__main__":
    main()
