import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from checkpoint import create_checkpoint
from generate_data import generate_sample
from infer import forecast
from model import DASRiskModel
from stream_demo import stream_forecasts


PROJECT_DIR = Path(__file__).resolve().parents[1]
FORECAST_FIELDS = {
    "event_type",
    "current_location_m",
    "trajectory",
    "estimated_speed_m_per_min",
    "predicted_future_location_m",
    "risk_horizon_seconds",
    "escalation_probability",
    "predicted_risk",
}


def test_forecast_json_contract_has_stable_fields_and_ranges() -> None:
    result = forecast(DASRiskModel(), np.zeros((20, 96), dtype=np.float32), 1_000, 300)
    assert set(result) == FORECAST_FIELDS
    assert isinstance(result["current_location_m"], int)
    assert 0 <= result["predicted_future_location_m"] <= 2_500
    assert 0 <= result["escalation_probability"] <= 1
    assert result["predicted_risk"] in {"low", "medium", "high"}


def test_stream_yields_requested_records_with_monotonic_window_offsets() -> None:
    signal, metadata = generate_sample(np.random.default_rng(5), scenario="excavation_approaching", time_steps=23)
    records = list(stream_forecasts(DASRiskModel(), signal, metadata, updates=4))
    assert len(records) == 4
    assert [record["window_end_offset_seconds"] for record in records] == [285, 300, 315, 330]
    locations = [record["current_location_m"] for record in records]
    assert locations == sorted(locations, reverse=metadata.direction < 0)


def test_stream_rejects_incomplete_traces() -> None:
    signal, metadata = generate_sample(np.random.default_rng(5), time_steps=20)
    with pytest.raises(ValueError, match="enough frames"):
        list(stream_forecasts(DASRiskModel(), signal, metadata, updates=2))


def test_inference_cli_loads_a_versioned_checkpoint(tmp_path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    torch.save(create_checkpoint(DASRiskModel(), seed=7), checkpoint_path)
    completed = subprocess.run(
        [sys.executable, "infer.py", "--model-path", str(checkpoint_path), "--scenario", "background_noise"],
        cwd=PROJECT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert set(result) == FORECAST_FIELDS


def test_visualization_cli_creates_an_image(tmp_path) -> None:
    output_path = tmp_path / "track.png"
    subprocess.run(
        [sys.executable, "visualize_track.py", "--scenario", "excavation_away", "--output", str(output_path)],
        cwd=PROJECT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    assert output_path.stat().st_size > 0
