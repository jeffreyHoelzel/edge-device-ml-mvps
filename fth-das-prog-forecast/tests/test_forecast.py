import numpy as np
import pytest
import torch

from generate_data import TIME_STEP_SECONDS, generate_sample, project_future_location
from infer import forecast


class FixedOutputModel(torch.nn.Module):
    def __init__(self, direction_index: int, speed: float) -> None:
        super().__init__()
        self.direction_index = direction_index
        self.speed = speed

    def forward(self, signal: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "event_logits": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            "direction_logits": torch.nn.functional.one_hot(
                torch.tensor([self.direction_index]), num_classes=3
            ).float(),
            "speed_m_per_min": torch.tensor([self.speed]),
            "escalation_probability": torch.tensor([0.8]),
        }


def test_future_location_is_derived_from_forecast_fields() -> None:
    result = forecast(FixedOutputModel(direction_index=1, speed=4.2), np.zeros((20, 96), dtype=np.float32), 1_800, 300)
    assert result["trajectory"] == "approaching_protected_asset"
    assert result["predicted_future_location_m"] == 1_821.0


def test_stationary_forecast_keeps_current_location() -> None:
    result = forecast(FixedOutputModel(direction_index=0, speed=9.0), np.zeros((20, 96), dtype=np.float32), 1_800, 300)
    assert result["estimated_speed_m_per_min"] == 0.0
    assert result["predicted_future_location_m"] == 1_800.0


@pytest.mark.parametrize(
    ("direction_index", "current_location", "expected_location"),
    [(1, 1_975, 1_995.0), (2, 1_975, 1_955.0), (1, 2_000, 1_980.0), (2, 2_000, 2_020.0)],
)
def test_nonstationary_forecasts_continue_moving_inside_the_protected_zone(
    direction_index: int, current_location: float, expected_location: float
) -> None:
    result = forecast(
        FixedOutputModel(direction_index=direction_index, speed=4.0),
        np.zeros((20, 96), dtype=np.float32),
        current_location,
        300,
    )
    assert result["predicted_future_location_m"] == expected_location


def test_generated_future_location_starts_at_window_end() -> None:
    _, metadata = generate_sample(np.random.default_rng(7), scenario="excavation_approaching")
    expected_end = project_future_location(
        metadata.current_location_m,
        metadata.direction,
        metadata.speed_m_per_min,
        19 * TIME_STEP_SECONDS,
    )
    expected_future = project_future_location(
        expected_end, metadata.direction, metadata.speed_m_per_min, metadata.horizon_seconds
    )
    assert metadata.window_end_location_m == expected_end
    assert metadata.future_location_m == expected_future
