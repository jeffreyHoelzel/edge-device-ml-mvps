import json
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=PROJECT_DIR,
        check=True,
        text=True,
        capture_output=True,
    )


def test_public_scripts_complete_a_seeded_cpu_pipeline(tmp_path) -> None:
    dataset = tmp_path / "train.npz"
    sample = tmp_path / "sample.npy"
    preview = tmp_path / "preview.png"
    model = tmp_path / "model.pt"
    _run(
        "generate_data.py",
        "--samples",
        "10",
        "--seed",
        "23",
        "--output",
        str(dataset),
        "--sample-output",
        str(sample),
        "--preview",
        str(preview),
    )
    _run("train.py", "--data", str(dataset), "--epochs", "1", "--batch-size", "5", "--model-output", str(model))
    inference = _run(
        "infer.py",
        str(sample),
        "--model",
        str(model),
        "--confidence-threshold",
        "0",
        "--frequency-start-hz",
        "3500000000",
        "--frequency-stop-hz",
        "3600000000",
    )
    result = json.loads(inference.stdout)
    assert dataset.is_file()
    assert sample.is_file()
    assert preview.is_file()
    assert model.is_file()
    assert set(result) == {
        "event_type",
        "class",
        "confidence",
        "frequency_start_normalized",
        "frequency_stop_normalized",
        "estimated_service_impact",
        "detection_status",
        "confidence_threshold",
        "frequency_start_hz",
        "frequency_stop_hz",
    }
    assert result["detection_status"] in {"detected", "no_interference", "abstained"}
