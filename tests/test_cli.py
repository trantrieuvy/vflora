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
