from pathlib import Path

from oceanscale.cli import _build_parser, _data_dir, _resolve_hover_checkpoint


def test_hover_checkpoint_falls_back_to_bundled_sb3() -> None:
    resolved = _resolve_hover_checkpoint(_data_dir())

    assert resolved == ("sb3", _data_dir() / "bluerov2_station_keep_final.zip")


def test_hover_checkpoint_prefers_skrl_policy(tmp_path: Path) -> None:
    (tmp_path / "bluerov2_station_keep_final.zip").touch()
    skrl_policy = tmp_path / "bluerov2_skrl_policy.pt"
    skrl_policy.touch()

    assert _resolve_hover_checkpoint(tmp_path) == ("skrl", skrl_policy)


def test_hover_checkpoint_missing_returns_none(tmp_path: Path) -> None:
    assert _resolve_hover_checkpoint(tmp_path) is None


def test_cli_parser_exposes_demo_and_train() -> None:
    parser = _build_parser()

    demo_args = parser.parse_args(["demo", "bluerov2-hover"])
    mvp_args = parser.parse_args(["demo", "underwater-mvp", "--steps", "8"])
    train_args = parser.parse_args(["train", "bluerov2-hover", "--total", "8"])

    assert demo_args.command == "demo"
    assert demo_args.task == "bluerov2-hover"
    assert mvp_args.command == "demo"
    assert mvp_args.task == "underwater-mvp"
    assert mvp_args.steps == 8
    assert train_args.command == "train"
    assert train_args.task == "bluerov2-hover"
    assert train_args.total == 8
