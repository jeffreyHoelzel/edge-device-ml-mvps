import pytest

from stream_demo import build_parser


def test_stream_demo_defaults_to_simulation() -> None:
    args = build_parser().parse_args([])

    assert args.simulate is False
    assert args.stdin is False


def test_stream_demo_accepts_explicit_simulation_and_stdin_modes() -> None:
    parser = build_parser()

    assert parser.parse_args(["--simulate"]).simulate is True
    assert parser.parse_args(["--stdin"]).stdin is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--simulate", "--stdin"])
