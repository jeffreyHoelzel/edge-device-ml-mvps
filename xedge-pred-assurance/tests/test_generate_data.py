import numpy as np

from generate_data import FORECAST_HORIZON_SECONDS, FORECAST_STEPS, generate_sequence, make_dataset


def test_dataset_is_deterministic_and_has_fixed_forecast_horizon() -> None:
    first = make_dataset(32, 12, seed=4)
    second = make_dataset(32, 12, seed=4)

    assert FORECAST_HORIZON_SECONDS == 60
    assert FORECAST_STEPS == 12
    assert first.keys() == second.keys()
    for key in first:
        assert np.array_equal(first[key], second[key])


def test_dataset_labels_are_for_a_future_target() -> None:
    data = make_dataset(64, 12, seed=9)

    assert data["x"].shape == (64, 12, 7)
    assert set(np.unique(data["incident"])).issubset({0.0, 1.0})
    assert np.all((data["cause"] >= 0) & (data["cause"] < 4))
    assert np.all((data["severity"] >= 0) & (data["severity"] < 3))


def test_raw_kpis_remain_physical_with_profile_and_benign_burst_variation() -> None:
    first = generate_sequence(64, None, rng=np.random.default_rng(5))
    second = generate_sequence(64, None, rng=np.random.default_rng(6))

    assert np.all(first[:, 0:3] >= 0.0)
    assert np.all(first[:, 3:5] >= 0.2)
    assert np.all((first[:, 5] >= 0.0) & (first[:, 5] <= 100.0))
    assert np.all(first[:, 6] >= 0.0)
    assert not np.array_equal(first, second)
