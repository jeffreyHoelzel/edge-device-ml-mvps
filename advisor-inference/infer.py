"""Run RF interference inference on a generated .npy spectrogram."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from contract import CLASS_NAMES, FREQUENCY_BINS, IMPACT_NAMES, TIME_BINS, normalize_per_window
from model import RFInterferenceCNN


def load_spectrogram(path: Path) -> torch.Tensor:
    array = np.load(path).astype(np.float32)
    array = np.squeeze(array)
    if array.shape != (TIME_BINS, FREQUENCY_BINS):
        raise ValueError(f"Expected one spectrogram shaped ({TIME_BINS}, {FREQUENCY_BINS}), got {array.shape}")
    return torch.from_numpy(array)[None, None, :, :]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify and localize one RF spectrogram .npy file.")
    parser.add_argument("spectrogram", type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/rf_interference_cnn.pt"))
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    model = RFInterferenceCNN()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        output = model(normalize_per_window(load_spectrogram(args.spectrogram)))
        probabilities = torch.softmax(output["class_logits"], dim=1)[0]
        class_id = int(probabilities.argmax())
        impact_id = int(output["impact_logits"].argmax(dim=1)[0])
        bounds = output["frequency_bounds"][0].tolist()

    # A no-interference prediction has intentionally empty localization bounds.
    if class_id == 0:
        bounds = [0.0, 0.0]
        impact_id = 0
    result = {
        "event_type": "rf_interference",
        "class": CLASS_NAMES[class_id],
        "confidence": round(float(probabilities[class_id]), 4),
        "frequency_start_normalized": round(float(bounds[0]), 4),
        "frequency_stop_normalized": round(float(bounds[1]), 4),
        "estimated_service_impact": IMPACT_NAMES[impact_id],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
