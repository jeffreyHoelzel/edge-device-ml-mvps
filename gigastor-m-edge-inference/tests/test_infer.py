import numpy as np
import pytest
import torch

from generate_data import CAUSES, SEVERITIES, generate_dataset
from infer import build_event
from model import RootCauseGRU, load_checkpoint, save_checkpoint


def test_root_cause_probabilities_are_ranked_and_sum_to_one() -> None:
    torch.manual_seed(3)
    dataset = generate_dataset(samples=10, sequence_length=8, seed=3)
    model = RootCauseGRU(input_size=dataset["features"].shape[-1])
    model.set_normalization(torch.tensor(dataset["features"].mean(axis=(0, 1))), torch.tensor(dataset["features"].std(axis=(0, 1))))
    event = build_event(model.eval(), dataset["features"][0], dataset["baselines"][0])
    ranking = event["root_cause_ranking"]
    probabilities = [item["probability"] for item in ranking]

    assert event["severity"] in SEVERITIES
    assert len(ranking) == len(CAUSES)
    assert sum(probabilities) == pytest.approx(1.0, abs=3e-6)
    assert probabilities == sorted(probabilities, reverse=True)


def test_checkpoint_can_be_saved_and_reloaded(tmp_path) -> None:
    torch.manual_seed(5)
    model = RootCauseGRU(input_size=9).eval()
    features = torch.rand(1, 6, 9)
    before = model(features)[0]
    path = tmp_path / "model.pt"
    save_checkpoint(model, path)
    after = load_checkpoint(path)(features)[0]
    assert np.allclose(before.detach().numpy(), after.detach().numpy())
