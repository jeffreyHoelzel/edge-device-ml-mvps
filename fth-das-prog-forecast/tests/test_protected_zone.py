from generate_data import distance_to_protected_zone, protected_zone_bounds


def test_protected_zone_distance_is_zero_inside_and_at_boundaries() -> None:
    assert protected_zone_bounds(2_000, 50) == (1_950, 2_050)
    assert distance_to_protected_zone(1_950, 2_000, 50) == 0
    assert distance_to_protected_zone(2_000, 2_000, 50) == 0
    assert distance_to_protected_zone(2_050, 2_000, 50) == 0


def test_protected_zone_distance_uses_nearest_boundary() -> None:
    assert distance_to_protected_zone(1_800, 2_000, 50) == 150
    assert distance_to_protected_zone(2_200, 2_000, 50) == 150
