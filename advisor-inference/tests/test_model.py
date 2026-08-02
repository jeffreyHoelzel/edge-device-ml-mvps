import torch

from model import NUM_CLASSES, NUM_IMPACT_LEVELS, RFInterferenceCNN


def test_model_output_shapes() -> None:
    model = RFInterferenceCNN()
    outputs = model(torch.randn(3, 1, 64, 128))
    assert outputs["class_logits"].shape == (3, NUM_CLASSES)
    assert outputs["frequency_bounds"].shape == (3, 2)
    assert outputs["impact_logits"].shape == (3, NUM_IMPACT_LEVELS)


def test_frequency_bounds_are_normalized_and_ordered() -> None:
    model = RFInterferenceCNN()
    bounds = model(torch.randn(8, 1, 64, 128))["frequency_bounds"]
    assert torch.all(bounds >= 0.0)
    assert torch.all(bounds <= 1.0)
    assert torch.all(bounds[:, 0] <= bounds[:, 1])
