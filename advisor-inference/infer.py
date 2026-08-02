"""Run RF interference inference on a generated .npy spectrogram."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from contract import (
    CHECKPOINT_FORMAT_VERSION,
    CLASS_NAMES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    FREQUENCY_BINS,
    IMPACT_NAMES,
    INPUT_SHAPE,
    NORMALIZATION_NAME,
    TIME_BINS,
    normalize_per_window,
)
from model import NUM_CLASSES, NUM_IMPACT_LEVELS, RFInterferenceCNN


def load_spectrogram(path: Path) -> torch.Tensor:
    array = np.load(path).astype(np.float32)
    array = np.squeeze(array)
    if array.shape != (TIME_BINS, FREQUENCY_BINS):
        raise ValueError(f"Expected one spectrogram shaped ({TIME_BINS}, {FREQUENCY_BINS}), got {array.shape}")
    return torch.from_numpy(array)[None, None, :, :]


def validate_checkpoint(checkpoint: object) -> dict[str, object]:
    """Ensure a checkpoint matches the current RF input and label contract."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid RF checkpoint: expected a metadata dictionary")
    expected = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "input_shape": list(INPUT_SHAPE),
        "normalization": NORMALIZATION_NAME,
        "model_config": {"num_classes": NUM_CLASSES, "num_impact_levels": NUM_IMPACT_LEVELS},
        "class_names": list(CLASS_NAMES),
        "impact_names": list(IMPACT_NAMES),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Incompatible RF checkpoint {key}: expected {value!r}, got {checkpoint.get(key)!r}")
    if not isinstance(checkpoint.get("model_state"), dict):
        raise ValueError("Invalid RF checkpoint: missing model_state")
    return checkpoint


def load_model(path: Path) -> RFInterferenceCNN:
    checkpoint = validate_checkpoint(torch.load(path, map_location="cpu", weights_only=True))
    model = RFInterferenceCNN()
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as error:
        raise ValueError(f"Incompatible RF checkpoint model_state: {error}") from error
    model.eval()
    return model


def confidence_threshold(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite value from 0 to 1")
    return parsed


def build_result(
    probabilities: torch.Tensor, impact_logits: torch.Tensor, frequency_bounds: torch.Tensor, threshold: float
) -> dict[str, str | float]:
    """Format one prediction while making low-confidence decisions explicit."""
    class_id = int(probabilities.argmax())
    confidence = float(probabilities[class_id])
    impact_id = int(impact_logits.argmax())
    bounds = frequency_bounds.tolist()
    status = "detected"
    if class_id == 0:
        status = "no_interference"
    elif confidence < threshold:
        status = "abstained"
    if status != "detected":
        bounds = [0.0, 0.0]
        impact_id = 0
    return {
        "event_type": "rf_interference",
        "class": CLASS_NAMES[class_id],
        "confidence": round(confidence, 4),
        "frequency_start_normalized": round(float(bounds[0]), 4),
        "frequency_stop_normalized": round(float(bounds[1]), 4),
        "estimated_service_impact": IMPACT_NAMES[impact_id],
        "detection_status": status,
        "confidence_threshold": round(threshold, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify and localize one RF spectrogram .npy file.")
    parser.add_argument("spectrogram", type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/rf_interference_cnn.pt"))
    parser.add_argument("--confidence-threshold", type=confidence_threshold, default=DEFAULT_CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    model = load_model(args.model)
    with torch.no_grad():
        output = model(normalize_per_window(load_spectrogram(args.spectrogram)))
        probabilities = torch.softmax(output["class_logits"], dim=1)[0]
        result = build_result(
            probabilities, output["impact_logits"][0], output["frequency_bounds"][0], args.confidence_threshold
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
