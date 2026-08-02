import pytest

from generate_data import make_dataset
from train import validate_dataset


def test_dataset_validation_rejects_out_of_range_labels() -> None:
    data = make_dataset(12, 8)
    data["cause"][0] = 9

    with pytest.raises(ValueError, match="out of range"):
        validate_dataset(data)
