import pytest

pd = pytest.importorskip("pandas")

from fed_adapter.analysis.tuning import (
    load_live_tuning_results,
    load_tuning_results,
    manifest_records,
    select_plateaus,
    summarize_tuning_results,
)


def test_load_tuning_results_reads_vflora_run_layout(tmp_path):
    log_path = (
        tmp_path
        / "tuning-nonlinear-ffa-wiz-tinyllama-homo-e1-r3"
        / "seed0"
        / "10"
        / "log.txt"
    )
    log_path.parent.mkdir(parents=True)
    log_path.write_text("0.1\n12.0\n0.13\n")

    scores = load_tuning_results(tmp_path)

    assert list(scores["Round"]) == [1, 2, 3]
    assert list(scores["Accuracy"]) == [10.0, 12.0, 13.0]
    assert set(scores["Variant key"]) == {"nonlinear-ffa"}
    assert list(scores["Client count"]) == [10, 10, 10]


def test_load_tuning_results_reads_legacy_flora_glue_log_path(tmp_path):
    log_path = (
        tmp_path
        / "tuning-flora-mrpc_stratified-roberta-base-homo-e1-r2"
        / "seed0"
        / "10log.txt"
    )
    log_path.parent.mkdir(parents=True)
    log_path.write_text("0.7\n0.75\n")

    scores = load_tuning_results(tmp_path)

    assert list(scores["Round"]) == [1, 2]
    assert list(scores["Accuracy"]) == [70.0, 75.0]
    assert set(scores["Variant key"]) == {"flora"}
    assert set(scores["Dataset"]) == {"mrpc_stratified"}
    assert set(scores["Model key"]) == {"roberta-base"}
    assert set(scores["Client count"]) == {10}


def test_select_plateaus_prefers_low_compute_near_best():
    summary = pd.DataFrame(
        [
            _summary_row(epochs=1, round_value=1, accuracy=10.0),
            _summary_row(epochs=1, round_value=2, accuracy=12.0),
            _summary_row(epochs=3, round_value=1, accuracy=12.5),
            _summary_row(epochs=3, round_value=2, accuracy=12.8),
        ]
    )

    _, selected = select_plateaus(summary, tolerance=1.0)

    assert len(selected) == 1
    row = selected.iloc[0]
    assert int(row["Selected epochs"]) == 1
    assert int(row["Selected round"]) == 2


def test_live_tuning_parser_reads_stdout_scores(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    output_dir = (
        tmp_path
        / "runs"
        / "tuning-nonlinear-ffa-wiz-tinyllama-homo-e1-r3"
        / "seed0"
    )
    stdout = logs / "vflora_tuning_123_1.out"
    stdout.write_text(
        f"output_dir={output_dir}\n"
        "Acc round 0: 0.25\n"
        "Acc round 1: 26.0\n"
    )

    scores = load_live_tuning_results(logs, run_roots=tmp_path / "runs")

    assert list(scores["Round"]) == [1, 2]
    assert list(scores["Accuracy"]) == [25.0, 26.0]
    assert set(scores["Run status"]) == {"Live partial"}


def test_manifest_records_deduplicates_rows():
    requests = pd.DataFrame(
        [
            {
                "variant": "nonlinear-ffa",
                "dataset": "wiz",
                "model": "tinyllama",
                "setting": "homo",
                "epochs": 1,
                "rounds": 3,
                "seed": 0,
            },
            {
                "variant": "nonlinear-ffa",
                "dataset": "wiz",
                "model": "tinyllama",
                "setting": "homo",
                "epochs": 1,
                "rounds": 3,
                "seed": 0,
            },
        ]
    )

    assert len(manifest_records(requests)) == 1


def test_summarize_tuning_results_adds_compute_cost(tmp_path):
    log_path = (
        tmp_path
        / "tuning-nonlinear-ffa-wiz-tinyllama-homo-e2-r2"
        / "seed0"
        / "10"
        / "log.txt"
    )
    log_path.parent.mkdir(parents=True)
    log_path.write_text("0.1\n0.2\n")

    summary = summarize_tuning_results(load_tuning_results(tmp_path))

    assert list(summary["Compute cost"]) == [2, 4]


def test_summarize_tuning_results_keeps_client_count_in_case_identity(tmp_path):
    base = tmp_path / "tuning-ffa-qqp_stratified-roberta-base-homo-e1-r1" / "seed0"
    ten_log = base / "10" / "log.txt"
    twenty_log = base / "20" / "log.txt"
    ten_log.parent.mkdir(parents=True)
    twenty_log.parent.mkdir(parents=True)
    ten_log.write_text("0.5\n")
    twenty_log.write_text("0.8\n")

    summary = summarize_tuning_results(load_tuning_results(tmp_path))

    assert sorted(summary["Client count"].tolist()) == [10, 20]
    assert sorted(summary["Mean accuracy"].tolist()) == [50.0, 80.0]


def _summary_row(*, epochs, round_value, accuracy):
    return {
        "Variant key": "nonlinear-ffa",
        "Variant": "Nonlinear FFA",
        "Dataset": "wiz",
        "Dataset label": "Wizard",
        "Model key": "tinyllama",
        "Model": "TinyLlama",
        "Setting key": "homo",
        "Setting": "Homo",
        "Local epochs": epochs,
        "Round": round_value,
        "Mean accuracy": accuracy,
        "Std accuracy": 0.0,
        "Seed count": 1,
        "Seeds": "0",
        "Max config rounds": 3,
        "Compute cost": epochs * round_value,
    }
