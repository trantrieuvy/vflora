from fed_adapter.cli.split_data import build_parser as build_split_parser
from fed_adapter.cli.train import build_parser as build_train_parser


def test_train_parser_defaults_without_ml_imports():
    args = build_train_parser().parse_args(
        ["--data-root", "data_wiz", "--output-dir", "runs/test"]
    )

    assert args.variant == "nonlinear-cumulative-flora"
    assert args.model == "tinyllama"
    assert args.rounds == 3


def test_split_parser_accepts_wizard_stratified_recipe():
    args = build_split_parser().parse_args(
        [
            "--dataset",
            "wizard",
            "--mode",
            "stratified_keep_sizes",
            "--source-root",
            "data_wiz",
            "--output-root",
            "data_wiz_stratified",
        ]
    )

    assert args.dataset == "wizard"
    assert args.mode == "stratified_keep_sizes"


def test_write_round_metadata_records_variant(tmp_path):
    import argparse
    import json
    import pytest

    torch = pytest.importorskip("torch")
    from fed_adapter.cli.train import _write_round_metadata

    round_A = {"layer.q_proj": torch.zeros(2, 3)}
    round_B = {"layer.q_proj": torch.zeros(4, 2)}
    frozen_A = {"layer.q_proj": torch.zeros(2, 3)}
    frozen_B = {"layer.q_proj": torch.zeros(4, 2)}
    args = argparse.Namespace(rank=16, alpha=32, heterogeneous=False)

    _write_round_metadata(
        tmp_path,
        0,
        "nonlinear-cumulative-flora",
        [0, 1],
        {0: 16, 1: 16},
        {0: 0.5, 1: 0.5},
        round_A,
        round_B,
        frozen_A,
        frozen_B,
        args,
    )

    metadata = json.loads((tmp_path / "round_config.json").read_text())
    assert metadata["variant"] == "nonlinear-cumulative-flora"
    assert metadata["round"] == 0
    assert metadata["round_rank"] == 2
