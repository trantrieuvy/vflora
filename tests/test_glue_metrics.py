import numpy as np
import pytest

from fed_adapter.cli.train_glue import compute_glue_metrics, _round_metric_summary


def test_mrpc_and_qqp_keep_accuracy_as_primary_metric():
    predictions = np.array([1, 0, 1, 0])
    labels = np.array([1, 1, 0, 0])

    for task_name in ("mrpc", "qqp"):
        metric = _MetricStub({"accuracy": 0.5, "f1": 0.5})
        metrics = compute_glue_metrics(task_name, predictions, labels, metric=metric)
        summary = _round_metric_summary(task_name, metrics, None)

        assert metric.predictions.dtype.kind in {"i", "u"}
        assert metric.references.dtype.kind in {"i", "u"}
        assert metrics["accuracy"] == pytest.approx(0.5)
        assert "f1" in metrics
        assert metrics["combined_score"] == pytest.approx(0.5)
        assert summary["primary_metric"] == "accuracy"
        assert summary["primary_score"] == pytest.approx(0.5)


def test_stsb_primary_metric_is_pearson():
    metric = _MetricStub({"pearson": 1.0, "spearmanr": 1.0})
    metrics = compute_glue_metrics(
        "stsb",
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.0, 1.0, 2.0, 3.0]),
        metric=metric,
    )
    summary = _round_metric_summary("stsb", metrics, None)

    assert metric.predictions.dtype.kind == "f"
    assert metric.references.dtype.kind == "f"
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["combined_score"] == pytest.approx(1.0)
    assert summary["primary_metric"] == "pearson"
    assert summary["primary_score"] == pytest.approx(1.0)


def test_mnli_primary_metric_averages_matched_and_mismatched_accuracy():
    summary = _round_metric_summary(
        "mnli",
        {"accuracy": 0.8},
        {"accuracy": 0.6},
    )

    assert summary["primary_metric"] == "accuracy"
    assert summary["primary_score"] == pytest.approx(0.7)
    assert summary["mnli_matched"] == {"accuracy": 0.8}
    assert summary["mnli_mismatched"] == {"accuracy": 0.6}


class _MetricStub:
    def __init__(self, result):
        self.result = result
        self.predictions = None
        self.references = None

    def compute(self, *, predictions, references):
        self.predictions = predictions
        self.references = references
        return self.result
