import numpy as np

from generate_data import FREQUENCY_BINS, TIME_BINS, generate_sample


def test_generated_interference_has_valid_frequency_bounds() -> None:
    rng = np.random.default_rng(7)
    for class_id in range(1, 5):
        sample = generate_sample(rng, class_id)
        assert sample.spectrogram.shape == (TIME_BINS, FREQUENCY_BINS)
        assert 0.0 <= sample.frequency_bounds[0] < sample.frequency_bounds[1] <= 1.0
