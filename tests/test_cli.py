from fed_adapter.cli.split_data import build_parser as build_split_parser
from fed_adapter.cli.generate_manifest import build_parser as build_manifest_parser
from fed_adapter.cli.generate_manifest import manifest_rows
from fed_adapter.cli.train import build_parser as build_train_parser
from fed_adapter.cli.train import VARIANT_ALIASES


def test_train_parser_defaults_without_ml_imports():
    args = build_train_parser().parse_args(
        ["--data-root", "data_wiz", "--output-dir", "runs/test"]
    )

    assert args.task_family == "instruction"
    assert args.variant is None
    assert args.model == "tinyllama"
    assert args.rounds == 3


def test_train_parser_accepts_ffa_aliases_without_ml_imports():
    args = build_train_parser().parse_args(
        [
            "--variant",
            "ffa",
            "--data-root",
            "data_wiz",
            "--output-dir",
            "runs/test",
            "--activation",
            "relu",
            "--a-init-std",
            "0.01",
        ]
    )

    assert VARIANT_ALIASES[args.variant] == "nonlinear-ffa"
    assert args.activation == "relu"
    assert args.a_init_std == 0.01


def test_train_parser_accepts_rolora_alias_without_ml_imports():
    args = build_train_parser().parse_args(
        [
            "--variant",
            "rolora",
            "--data-root",
            "data_wiz",
            "--output-dir",
            "runs/test",
            "--calibration-path",
            "calibration.json",
        ]
    )

    assert VARIANT_ALIASES[args.variant] == "nonlinear-rolora"
    assert args.calibration_path.name == "calibration.json"
    assert args.distill_steps == 200


def test_train_parser_accepts_glue_federatedllm_aliases_without_ml_imports():
    args = build_train_parser().parse_args(
        [
            "--task-family",
            "glue",
            "--method",
            "linear_flora_cumulative",
            "--model",
            "roberta",
            "--task-name",
            "mrpc",
            "--data-root",
            "data_mrpc_stratified",
            "--output-dir",
            "runs/test",
            "--num_clients",
            "10",
            "--num_communication_rounds",
            "5",
            "--lora_r",
            "8",
            "--lora_alpha",
            "16",
            "--heter",
            "True",
            "--local_val_set_size",
            "0.1",
            "--max_rounds_per_invocation",
            "2",
            "--resume_from_latest",
        ]
    )

    assert args.task_family == "glue"
    assert args.method == "linear_flora_cumulative"
    assert args.model == "roberta"
    assert args.task_name == "mrpc"
    assert args.num_clients == 10
    assert args.rounds == 5
    assert args.rank == 8
    assert args.alpha == 16
    assert args.heter == "True"
    assert args.local_val_size == 0.1
    assert args.max_rounds_per_invocation == 2
    assert args.resume_from_latest is True


def test_manifest_parser_smoke_rows_include_ffa():
    args = build_manifest_parser().parse_args(["--phase", "smoke"])
    rows = manifest_rows(args)

    assert any(row["variant"] == "nonlinear-ffa" for row in rows)
    assert {row["rounds"] for row in rows} == {1}


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


def test_split_parser_accepts_glue_underscore_aliases():
    args = build_split_parser().parse_args(
        [
            "--dataset",
            "glue",
            "--mode",
            "stratified",
            "--task_name",
            "qnli",
            "--num_clients",
            "10",
            "--source_split_dir",
            "data_qnli_stratified/10",
            "--output_root",
            "data_qnli_stratified",
        ]
    )

    assert args.dataset == "glue"
    assert args.mode == "stratified"
    assert args.task_name == "qnli"
    assert args.num_clients == 10
    assert args.source_split_dir.as_posix() == "data_qnli_stratified/10"


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


def test_write_ffa_round_metadata_records_global_rank(tmp_path):
    import argparse
    import json
    import pytest

    torch = pytest.importorskip("torch")
    from fed_adapter.cli.train import _write_ffa_round_metadata

    args = argparse.Namespace(
        rank=16,
        alpha=32,
        activation="gelu",
        a_init_std=0.02,
        heterogeneous=True,
    )

    _write_ffa_round_metadata(
        tmp_path,
        0,
        "nonlinear-ffa",
        [0, 1],
        {0: 64, 1: 32},
        {0: 0.25, 1: 0.75},
        {"layer.q_proj": torch.zeros(64, 3)},
        {"layer.q_proj": torch.zeros(4, 64)},
        64,
        args,
    )

    metadata = json.loads((tmp_path / "round_config.json").read_text())
    assert metadata["variant"] == "nonlinear-ffa"
    assert metadata["rank_semantics"] == "ffa_global_B"
    assert metadata["global_rank"] == 64


def test_rolora_validation_rejects_heterogeneous_and_global_test():
    import argparse
    import pytest
    from pathlib import Path

    from fed_adapter.cli.train import _validate_train_args

    args = argparse.Namespace(
        heterogeneous=True,
        calibration_path=Path("calibration.json"),
        rounds=2,
        distill_calibration_size=512,
        distill_max_tokens=8192,
        distill_steps=200,
        distill_batch_size=64,
    )
    with pytest.raises(ValueError, match="homogeneous"):
        _validate_train_args(args, "nonlinear-rolora")

    args.heterogeneous = False
    args.calibration_path = Path("global_test.json")
    with pytest.raises(ValueError, match="global_test"):
        _validate_train_args(args, "nonlinear-rolora")


def test_calibration_prompts_ignore_output(tmp_path):
    import json

    from fed_adapter.cli.train import _load_calibration_prompts
    from fed_adapter.data.prompting import get_template

    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            [
                {
                    "instruction": "Classify this.",
                    "input": "Premise text",
                    "output": "SECRET_LABEL",
                }
            ]
        )
    )

    prompts = _load_calibration_prompts(path, get_template("alpaca"), limit=10, seed=0)

    assert len(prompts) == 1
    assert "Classify this." in prompts[0]
    assert "Premise text" in prompts[0]
    assert "SECRET_LABEL" not in prompts[0]
