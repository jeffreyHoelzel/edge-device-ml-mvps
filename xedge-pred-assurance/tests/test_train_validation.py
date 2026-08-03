from argparse import Namespace

import pytest

from generate_data import FORECAST_HORIZON_SECONDS, MIN_TRAINING_SAMPLES, SAMPLE_INTERVAL_SECONDS, dataset_metadata, make_dataset, save_dataset
from train import load_or_generate, validate_dataset, validate_training_coverage, validate_training_options


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


def test_training_options_require_enough_samples_for_stratified_holdout() -> None:
    args = Namespace(samples=MIN_TRAINING_SAMPLES - 1, sequence_length=8, epochs=1, batch_size=1)

    with pytest.raises(ValueError, match=str(MIN_TRAINING_SAMPLES)):
        validate_training_options(args)


def test_loaded_dataset_requires_every_diagnosis_stratum(tmp_path) -> None:
    data = make_dataset(MIN_TRAINING_SAMPLES, 8, seed=2)
    mask = (data["cause"] == 0) & (data["severity"] == 0)
    data["incident"][mask] = 0
    path = tmp_path / "incomplete.npz"
    save_dataset(path, data)

    with pytest.raises(ValueError, match="rf_interference/minor"):
        load_or_generate(path, MIN_TRAINING_SAMPLES, 8, seed=2)


def test_training_coverage_accepts_generated_training_data() -> None:
    validate_training_coverage(make_dataset(MIN_TRAINING_SAMPLES, 8, seed=2))
