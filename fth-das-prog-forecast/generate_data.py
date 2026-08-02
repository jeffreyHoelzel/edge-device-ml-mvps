"""Synthetic DAS-like spatiotemporal data; it does not process real DAS data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_LABELS = ("background_noise", "passing_vehicle", "stationary_vibration", "excavation_activity")
DIRECTION_LABELS = ("stationary", "toward_asset", "away_from_asset")
SCENARIOS = (
    "background_noise",
    "passing_vehicle",
    "stationary_vibration",
    "excavation_approaching",
    "excavation_away",
)
FIBER_LENGTH_M = 2_500.0
PROTECTED_ZONE_M = 2_000.0
MAX_SPEED_M_PER_MIN = 12.0
DEFAULT_HORIZON_SECONDS = 300
TIME_STEP_SECONDS = 15


@dataclass(frozen=True)
class EventMetadata:
    scenario: str
    event_type: str
    class_index: int
    direction_index: int
    current_location_m: float
    window_end_location_m: float
    direction: int
    speed_m_per_min: float
    intensity: float
    protected_zone_m: float
    horizon_seconds: int
    future_location_m: float
    escalation_probability: float


def project_future_location(
    location_m: float, direction: int, speed_m_per_min: float, horizon_seconds: float,
    fiber_length_m: float = FIBER_LENGTH_M,
) -> float:
    """Project location while clipping it to the modeled fiber."""
    projected = location_m + direction * speed_m_per_min * horizon_seconds / 60.0
    return float(np.clip(projected, 0.0, fiber_length_m))


def _toward_direction(location_m: float, protected_zone_m: float) -> int:
    return 1 if location_m < protected_zone_m else -1


def _scenario_labels(scenario: str, location_m: float, protected_zone_m: float) -> tuple[int, int, int, float]:
    if scenario == "background_noise":
        return 0, 0, 0, 0.0
    if scenario == "passing_vehicle":
        direction = 1 if location_m < fiber_midpoint() else -1
        return 1, 2 if direction < 0 else 1, direction, 8.0
    if scenario == "stationary_vibration":
        return 2, 0, 0, 0.0
    toward = _toward_direction(location_m, protected_zone_m)
    if scenario == "excavation_approaching":
        return 3, 1, toward, 4.5
    return 3, 2, -toward, 4.5


def fiber_midpoint() -> float:
    return FIBER_LENGTH_M / 2.0


def generate_sample(
    rng: np.random.Generator,
    scenario: str | None = None,
    time_steps: int = 20,
    distance_bins: int = 96,
    horizon_seconds: int = DEFAULT_HORIZON_SECONDS,
    protected_zone_m: float = PROTECTED_ZONE_M,
) -> tuple[np.ndarray, EventMetadata]:
    """Create one [time, distance] intensity array and its synthetic labels."""
    scenario = scenario or str(rng.choice(SCENARIOS))
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    location = float(rng.uniform(150.0, FIBER_LENGTH_M - 150.0))
    class_index, direction_index, direction, base_speed = _scenario_labels(scenario, location, protected_zone_m)
    speed = 0.0 if base_speed == 0 else float(np.clip(rng.normal(base_speed, 1.0), 1.0, MAX_SPEED_M_PER_MIN))
    intensity = float(rng.uniform(0.45, 1.0))
    time = np.arange(time_steps, dtype=np.float32)
    distance = np.linspace(0, FIBER_LENGTH_M, distance_bins, dtype=np.float32)
    # Each frame represents 15 seconds: enough motion to be visible in this tiny grid.
    frame_locations = location + direction * speed * time * TIME_STEP_SECONDS / 60.0
    window_end_location = project_future_location(
        location, direction, speed, (time_steps - 1) * TIME_STEP_SECONDS
    )
    future_location = project_future_location(window_end_location, direction, speed, horizon_seconds)
    signal = rng.normal(0.0, 0.055, size=(time_steps, distance_bins)).astype(np.float32)
    width_m = 55.0 if scenario != "passing_vehicle" else 40.0
    for frame, center in enumerate(frame_locations):
        amplitude = intensity
        if scenario == "background_noise":
            amplitude = 0.0
        elif scenario == "stationary_vibration":
            amplitude *= 0.85 + 0.15 * np.sin(frame * 0.9)
        elif scenario == "passing_vehicle":
            amplitude *= 0.7 + 0.3 * np.sin(frame * 1.2) ** 2
        signal[frame] += amplitude * np.exp(-0.5 * ((distance - center) / width_m) ** 2)
    signal += 0.018 * np.sin(distance[None, :] / 75.0 + time[:, None] * 0.6)

    distance_to_zone = abs(future_location - protected_zone_m)
    is_approaching_excavation = scenario == "excavation_approaching"
    risk = 0.03 + 0.91 * float(is_approaching_excavation) * np.exp(-distance_to_zone / 700.0)
    risk += float(rng.normal(0.0, 0.02))
    risk = float(np.clip(risk, 0.0, 1.0))
    metadata = EventMetadata(
        scenario=scenario,
        event_type=CLASS_LABELS[class_index],
        class_index=class_index,
        direction_index=direction_index,
        current_location_m=location,
        window_end_location_m=window_end_location,
        direction=direction,
        speed_m_per_min=speed,
        intensity=intensity,
        protected_zone_m=protected_zone_m,
        horizon_seconds=horizon_seconds,
        future_location_m=future_location,
        escalation_probability=risk,
    )
    return signal.astype(np.float32), metadata


class SyntheticDASDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic in-memory dataset intended for a fast CPU demonstration."""

    def __init__(self, num_samples: int, seed: int = 7, **sample_kwargs: object) -> None:
        rng = np.random.default_rng(seed)
        self.samples = [generate_sample(rng, **sample_kwargs) for _ in range(num_samples)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        signal, meta = self.samples[index]
        return {
            "signal": torch.from_numpy(signal),
            "class_index": torch.tensor(meta.class_index, dtype=torch.long),
            "direction_index": torch.tensor(meta.direction_index, dtype=torch.long),
            "speed": torch.tensor(meta.speed_m_per_min / MAX_SPEED_M_PER_MIN, dtype=torch.float32),
            "escalation": torch.tensor(meta.escalation_probability, dtype=torch.float32),
        }
