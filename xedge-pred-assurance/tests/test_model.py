import torch

from model import ModelConfig, PredictiveAssuranceModel


def test_model_output_shapes() -> None:
    model = PredictiveAssuranceModel(ModelConfig(conv_channels=8, gru_hidden=10))
    output = model(torch.randn(4, 30, 7))

    assert set(output) == {"incident_logits", "cause_logits", "severity_logits"}
    assert output["incident_logits"].shape == (4,)
    assert output["cause_logits"].shape == (4, 4)
    assert output["severity_logits"].shape == (4, 3)
