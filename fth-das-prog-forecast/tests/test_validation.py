import numpy as np
import pytest
import torch

from generate_data import SyntheticDASDataset, generate_sample, validate_signal
from infer import forecast
from model import DASRiskModel


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"time_steps": 0}, "time_steps"),
        ({"distance_bins": 1}, "distance_bins"),
        ({"horizon_seconds": 0}, "horizon_seconds"),
        ({"protected_zone_half_width_m": 0}, "half width"),
        ({"protected_zone_center_m": 2_490, "protected_zone_half_width_m": 20}, "fit"),
    ],
)
def test_generate_sample_rejects_invalid_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generate_sample(np.random.default_rng(7), **kwargs)


def test_dataset_requires_a_positive_sample_count() -> None:
    with pytest.raises(ValueError, match="num_samples"):
        SyntheticDASDataset(0)


@pytest.mark.parametrize(
    ("signal", "message"),
    [
        (torch.ones(20, 96), "shape"),
        (torch.ones(1, 20, 1), "distance bins"),
        (torch.ones(1, 20, 96, dtype=torch.int64), "floating-point"),
        (torch.full((1, 20, 96), float("nan")), "finite"),
    ],
)
def test_model_input_validation(signal: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DASRiskModel()(signal)


def test_forecast_rejects_an_invalid_horizon() -> None:
    with pytest.raises(ValueError, match="horizon_seconds"):
        forecast(DASRiskModel(), np.zeros((20, 96), dtype=np.float32), 1_000, 0)


def test_validate_signal_accepts_valid_input() -> None:
    validate_signal(torch.zeros(1, 20, 96))
