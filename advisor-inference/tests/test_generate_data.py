import numpy as np
import pytest

from contract import CLASS_NAMES, FREQUENCY_BINS, TIME_BINS
from generate_data import generate_dataset, generate_sample


def test_generated_interference_has_valid_frequency_bounds() -> None:
    rng = np.random.default_rng(7)
    for class_id in range(1, 5):
        sample = generate_sample(rng, class_id)
        assert sample.spectrogram.shape == (TIME_BINS, FREQUENCY_BINS)
        assert 0.0 <= sample.frequency_bounds[0] < sample.frequency_bounds[1] <= 1.0


def test_generation_is_seeded_and_preserves_class_labels() -> None:
    first = generate_dataset(10, seed=19)
    second = generate_dataset(10, seed=19)
    for first_value, second_value in zip(first, second, strict=True):
        assert np.array_equal(first_value, second_value)
    assert set(first[1]) == set(range(len(CLASS_NAMES)))


@pytest.mark.parametrize("class_id", [-1, len(CLASS_NAMES), 1.5, True])
def test_invalid_explicit_class_id_is_rejected(class_id: int | float | bool) -> None:
    with pytest.raises(ValueError, match="class_id"):
        generate_sample(np.random.default_rng(3), class_id)  # type: ignore[arg-type]


def test_dataset_requires_at_least_one_sample() -> None:
    with pytest.raises(ValueError, match="samples"):
        generate_dataset(0)
