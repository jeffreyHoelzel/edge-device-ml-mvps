from datetime import datetime, timedelta, timezone

import torch

from generate_data import (
    CAUSE_NAMES,
    FEATURE_MEAN,
    FEATURE_NAMES,
    FEATURE_STD,
    FORECAST_HORIZON_SECONDS,
    SAMPLE_INTERVAL_SECONDS,
    SEVERITY_NAMES,
    make_dataset,
)
from model import PredictiveAssuranceModel, TrainingContract, load_model, save_model
from streaming import KpiStreamProcessor
from train import loss_for_batch, set_seed


def test_generated_data_to_checkpoint_to_streaming_event(tmp_path) -> None:
    set_seed(4)
    data = make_dataset(32, 8, seed=4)
    model = PredictiveAssuranceModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    x = torch.from_numpy(data["x"][:16])
    optimizer.zero_grad()
    loss_for_batch(model(x), torch.from_numpy(data["incident"][:16]), torch.from_numpy(data["cause"][:16]), torch.from_numpy(data["severity"][:16])).backward()
    optimizer.step()
    contract = TrainingContract(
        FEATURE_NAMES, tuple(float(value) for value in FEATURE_MEAN), tuple(float(value) for value in FEATURE_STD),
        CAUSE_NAMES, SEVERITY_NAMES, 8, SAMPLE_INTERVAL_SECONDS, FORECAST_HORIZON_SECONDS,
    )
    path = tmp_path / "model.pt"
    save_model(model, path, contract)
    processor = KpiStreamProcessor(load_model(path))
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    event = None
    for step, values in enumerate(data["x"][0] * FEATURE_STD + FEATURE_MEAN):
        record = {"timestamp": (timestamp + timedelta(seconds=step * 5)).isoformat().replace("+00:00", "Z"), "service_id": "edge-1"}
        record.update({name: float(value) for name, value in zip(FEATURE_NAMES, values)})
        event = processor.process(record)
    assert event is not None
    assert event["service_id"] == "edge-1"
    assert event["forecast_horizon_seconds"] == 60
