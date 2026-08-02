import torch

from generate_data import CLASS_LABELS, DIRECTION_LABELS
from model import DASRiskModel


def test_model_output_shapes() -> None:
    model = DASRiskModel()
    outputs = model(torch.randn(3, 20, 96))
    assert outputs["event_logits"].shape == (3, len(CLASS_LABELS))
    assert outputs["direction_logits"].shape == (3, len(DIRECTION_LABELS))
    assert outputs["speed_m_per_min"].shape == (3,)
    assert outputs["escalation_probability"].shape == (3,)


def test_probability_range() -> None:
    model = DASRiskModel()
    probabilities = model(torch.randn(8, 20, 96))["escalation_probability"]
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
