import pytest

from generate_data import FORECAST_HORIZON_SECONDS, SAMPLE_INTERVAL_SECONDS, dataset_metadata, make_dataset
from train import validate_dataset


def test_dataset_validation_rejects_out_of_range_labels() -> None:
    data = make_dataset(12, 8)
    data["cause"][0] = 9

    with pytest.raises(ValueError, match="out of range"):
        validate_dataset(data)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("sample_interval_seconds", SAMPLE_INTERVAL_SECONDS + 1, "sample interval"),
        ("forecast_horizon_seconds", FORECAST_HORIZON_SECONDS + 1, "forecast horizon"),
    ),
)
def test_dataset_validation_rejects_incompatible_timing_metadata(field: str, value: int, message: str) -> None:
    data = make_dataset(32, 8)
    metadata = dataset_metadata(8)
    metadata[field] = value

    with pytest.raises(ValueError, match=message):
        validate_dataset(data, metadata)
