from generate_data import FIBER_LENGTH_M, project_future_location


def test_future_location_calculation() -> None:
    assert project_future_location(1_800, 1, 4.2, 300) == 1_821.0
    assert project_future_location(1_800, -1, 4.2, 300) == 1_779.0


def test_future_location_is_clipped_to_fiber() -> None:
    assert project_future_location(FIBER_LENGTH_M - 1, 1, 12, 300) == FIBER_LENGTH_M
