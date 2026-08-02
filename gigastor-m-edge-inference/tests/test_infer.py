import numpy as np
import pytest
import torch

from generate_data import CAUSES, SEVERITIES, generate_dataset
from infer import build_event
from model import RootCauseGRU, load_checkpoint, save_checkpoint
from stream_demo import validate_event_count
from data_contract import validate_dataset


def test_root_cause_probabilities_are_ranked_and_sum_to_one() -> None:
    torch.manual_seed(3)
    dataset = generate_dataset(samples=12, sequence_length=8, seed=3)
    model = RootCauseGRU(input_size=dataset["features"].shape[-1])
    model.set_normalization(torch.tensor(dataset["features"].mean(axis=(0, 1))), torch.tensor(dataset["features"].std(axis=(0, 1))))
    with torch.no_grad():
        model.incident_head.weight.zero_()
        model.incident_head.bias[:] = torch.tensor([0.0, 10.0])
    event = build_event(model.eval(), dataset["features"][0], dataset["baselines"][0])
    ranking = event["root_cause_ranking"]
    probabilities = [item["probability"] for item in ranking]

    assert event["severity"] in SEVERITIES
    assert event["incident_detected"] is True
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


def test_model_exposes_binary_incident_head() -> None:
    model = RootCauseGRU(input_size=9)
    _, _, incident_logits = model(torch.rand(2, 3, 9))
    assert incident_logits.shape == (2, 2)


@pytest.mark.parametrize(("bias", "detected"), [((-10.0, 10.0), True), ((10.0, -10.0), False)])
def test_event_uses_incident_gate(bias, detected) -> None:
    dataset = generate_dataset(samples=6)
    model = RootCauseGRU(input_size=9).eval()
    with torch.no_grad():
        model.incident_head.weight.zero_()
        model.incident_head.bias[:] = torch.tensor(bias)
    event = build_event(model, dataset["features"][0], dataset["baselines"][0])
    assert event["incident_detected"] is detected
    assert event["event_type"] == ("application_performance_incident" if detected else "application_performance_status")
    assert event["root_cause_ranking"] or not detected


@pytest.mark.parametrize("sequence_length", [0, 1, 2])
def test_generator_rejects_short_sequences(sequence_length) -> None:
    with pytest.raises(ValueError, match="at least 3"):
        generate_dataset(samples=6, sequence_length=sequence_length)


def test_inference_rejects_short_sequences() -> None:
    model = RootCauseGRU(input_size=9).eval()
    with pytest.raises(ValueError, match="at least 3"):
        build_event(model, np.zeros((2, 9), dtype=np.float32), np.ones(9, dtype=np.float32))


@pytest.mark.parametrize("top_k", [0, -1, len(CAUSES) + 1])
def test_inference_rejects_invalid_ranking_limit(top_k) -> None:
    dataset = generate_dataset(samples=6, seed=4)
    with pytest.raises(ValueError, match="top_k"):
        build_event(RootCauseGRU(input_size=9).eval(), dataset["features"][0], dataset["baselines"][0], top_k=top_k)


@pytest.mark.parametrize("events, available", [(0, 2), (2, 2)])
def test_stream_accepts_available_event_counts(events, available) -> None:
    validate_event_count(events, available)


@pytest.mark.parametrize("events", [-1, 3])
def test_stream_rejects_unavailable_event_counts(events) -> None:
    with pytest.raises(ValueError, match="events"):
        validate_event_count(events, 2)


def test_dataset_contract_rejects_missing_and_nonfinite_values() -> None:
    dataset = generate_dataset(samples=6)
    broken = dict(dataset)
    del broken["baselines"]
    with pytest.raises(ValueError, match="missing"):
        validate_dataset(broken)
    broken = dict(dataset)
    broken["features"] = broken["features"].copy()
    broken["features"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_dataset(broken)


def test_generator_includes_deterministic_normal_traffic() -> None:
    first = generate_dataset(samples=12, seed=8)
    second = generate_dataset(samples=12, seed=8)
    assert np.array_equal(first["incident"], second["incident"])
    assert first["incident"].sum() == 10
