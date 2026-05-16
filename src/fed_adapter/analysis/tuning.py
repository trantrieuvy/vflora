"""Helpers for epoch/round tuning analysis."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable


MANIFEST_COLUMNS = ["variant", "dataset", "model", "setting", "epochs", "rounds", "seed"]

VARIANT_KEYS = (
    "linear-cumulative-flora",
    "nonlinear-cumulative-flora",
    "nonlinear-ffa",
)
VARIANT_PATTERN = "|".join(re.escape(variant) for variant in VARIANT_KEYS)
RUN_DIR_RE = re.compile(
    rf"^tuning-(?P<variant>{VARIANT_PATTERN})-(?P<dataset>.+)-"
    r"(?P<model>tinyllama|llama-7b|gpt2)-(?P<setting>homo|heter)-"
    r"e(?P<epochs>\d+)-r(?P<rounds>\d+)$"
)
LIVE_OUTPUT_DIR_RE = re.compile(r"^output_dir=(?P<output_dir>.+)$", re.MULTILINE)
LIVE_ACCURACY_RE = re.compile(
    r"Acc round\s+(?P<round>\d+):\s+"
    r"(?P<accuracy>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)

VARIANT_LABELS = {
    "linear-cumulative-flora": "Linear Cumulative FLoRA",
    "nonlinear-cumulative-flora": "Nonlinear Cumulative FLoRA",
    "nonlinear-ffa": "Nonlinear FFA",
}
MODEL_LABELS = {
    "tinyllama": "TinyLlama",
    "llama-7b": "Llama-7B",
    "gpt2": "GPT-2",
}
SETTING_LABELS = {
    "homo": "Homo",
    "heter": "Heter",
}
DATASET_LABELS = {
    "wiz": "Wizard",
    "wiz_stratified": "Wizard stratified",
    "dolly": "Dolly",
    "dolly_stratified": "Dolly stratified",
}


class _PandasProxy:
    def __getattr__(self, name: str):
        try:
            import pandas as pandas_module
        except ImportError as exc:
            raise RuntimeError(
                "Tuning analysis requires pandas. Install the training requirements "
                "with `pip install -r requirements-train.txt`."
            ) from exc
        globals()["pd"] = pandas_module
        return getattr(pandas_module, name)


pd = _PandasProxy()


def load_tuning_results(
    base_dir: str | Path | Iterable[str | Path] = ".",
    complete_only: bool = False,
) -> pd.DataFrame:
    """Discover tuning run directories and return one row per evaluated round."""
    run_records = []
    for run_dir in _iter_tuning_run_dirs(base_dir):
        match = RUN_DIR_RE.match(run_dir.name)
        if not match:
            continue
        info = match.groupdict()
        for seed_dir in sorted(run_dir.glob("seed*")):
            seed = _parse_seed(seed_dir)
            if seed is None:
                continue
            score_path = _score_path(seed_dir)
            if score_path is None:
                continue
            rounds = int(info["rounds"])
            raw_scores = _read_scores(score_path)
            if complete_only and len(raw_scores) < rounds:
                continue
            _append_run_record(
                run_records,
                variant=info["variant"],
                dataset=info["dataset"],
                model=info["model"],
                setting=info["setting"],
                epochs=int(info["epochs"]),
                rounds=rounds,
                seed=seed,
                scores=raw_scores[:rounds],
                run_dir=run_dir,
                score_path=score_path,
                observed_rounds=len(raw_scores),
            )
    return _records_to_scores_frame(_deduplicate_run_records(run_records))


def load_live_tuning_results(
    log_dir: str | Path | Iterable[str | Path] = "logs",
    *,
    run_roots: str | Path | Iterable[str | Path] | None = None,
    pattern: str = "vflora_tuning_*.out",
) -> pd.DataFrame:
    """Parse in-progress tuning scores from launcher stdout logs."""
    allowed_roots = (
        None
        if run_roots is None
        else [_resolve_path(path) for path in _iter_base_paths(run_roots)]
    )
    run_records = []
    for log_path in _iter_live_log_paths(log_dir, pattern):
        text = log_path.read_text(errors="replace")
        output_matches = list(LIVE_OUTPUT_DIR_RE.finditer(text))
        if not output_matches:
            continue
        output_dir = output_matches[-1].group("output_dir").strip().strip('"').strip("'")
        seed_dir = Path(output_dir.rstrip("/"))
        seed = _parse_seed(seed_dir)
        if seed is None:
            continue
        run_dir = seed_dir.parent
        match = RUN_DIR_RE.match(run_dir.name)
        if not match:
            continue
        resolved_run_dir = _resolve_path(run_dir)
        if allowed_roots and not any(_is_relative_to(resolved_run_dir, root) for root in allowed_roots):
            continue

        info = match.groupdict()
        rounds = int(info["rounds"])
        scores_by_round = {}
        for score_match in LIVE_ACCURACY_RE.finditer(text):
            round_idx = int(score_match.group("round"))
            if round_idx < rounds:
                scores_by_round[round_idx] = _normalize_accuracy(float(score_match.group("accuracy")))
        if not scores_by_round:
            continue
        scores = [scores_by_round[round_idx] for round_idx in sorted(scores_by_round)]
        _append_run_record(
            run_records,
            variant=info["variant"],
            dataset=info["dataset"],
            model=info["model"],
            setting=info["setting"],
            epochs=int(info["epochs"]),
            rounds=rounds,
            seed=seed,
            scores=scores,
            run_dir=run_dir,
            score_path=log_path,
            observed_rounds=len(scores),
            complete_run=False,
            result_source="launcher stdout",
            run_status="Live partial",
            extra_fields={"Log mtime": log_path.stat().st_mtime},
        )
    return _records_to_scores_frame(_deduplicate_run_records(run_records, include_run_dir=True))


def summarize_tuning_results(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-round scores across seeds."""
    if scores.empty:
        return _empty_summary_frame()

    group_columns = [
        "Variant key",
        "Variant",
        "Dataset",
        "Dataset label",
        "Model key",
        "Model",
        "Setting key",
        "Setting",
        "Local epochs",
        "Round",
    ]
    for optional_column in ["Result source", "Run status"]:
        if optional_column in scores.columns:
            group_columns.append(optional_column)

    summary = (
        scores.groupby(group_columns)
        .agg(
            **{
                "Mean accuracy": ("Accuracy", "mean"),
                "Std accuracy": ("Accuracy", "std"),
                "Seed count": ("Seed", "nunique"),
                "Seeds": ("Seed", lambda values: ", ".join(str(v) for v in sorted(set(values)))),
                "Max config rounds": ("Config rounds", "max"),
            }
        )
        .reset_index()
    )
    summary["Std accuracy"] = summary["Std accuracy"].fillna(0.0)
    summary["Compute cost"] = summary["Local epochs"] * summary["Round"]
    return summary.sort_values(
        ["Dataset", "Variant key", "Model key", "Setting key", "Local epochs", "Round"]
    )


def select_plateaus(
    summary: pd.DataFrame,
    tolerance: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-epoch plateau rows and one efficient selected row per case."""
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    data = _summary_with_compute_cost(summary)
    epoch_columns = _epoch_group_columns()
    case_columns = _case_group_columns()

    plateau_rows = []
    for group_values, group in data.groupby(epoch_columns, sort=False):
        curve = group.sort_values("Round")
        best_row = curve.loc[curve["Mean accuracy"].idxmax()]
        best_accuracy = float(best_row["Mean accuracy"])
        plateau_row = best_row
        for _, row in curve.iterrows():
            within_best = best_accuracy - float(row["Mean accuracy"]) <= tolerance
            no_large_future_gain = _future_gain(curve, int(row["Round"])) <= tolerance
            if within_best and no_large_future_gain:
                plateau_row = row
                break
        plateau_rows.append(
            dict(zip(epoch_columns, group_values))
            | {
                "Plateau round": int(plateau_row["Round"]),
                "Plateau accuracy": float(plateau_row["Mean accuracy"]),
                "Best round": int(best_row["Round"]),
                "Best accuracy": best_accuracy,
                "Max round observed": int(curve["Round"].max()),
            }
        )

    selected_rows = []
    for group_values, group in data.groupby(case_columns, sort=False):
        best_row = group.loc[group["Mean accuracy"].idxmax()]
        best_accuracy = float(best_row["Mean accuracy"])
        eligible = group[group["Mean accuracy"] >= best_accuracy - tolerance].copy()
        selected = eligible.sort_values(
            ["Compute cost", "Round", "Local epochs", "Mean accuracy"],
            ascending=[True, True, True, False],
        ).iloc[0]
        selected_rows.append(
            dict(zip(case_columns, group_values))
            | {
                "Selected epochs": int(selected["Local epochs"]),
                "Selected round": int(selected["Round"]),
                "Selected accuracy": float(selected["Mean accuracy"]),
                "Selected std": float(selected["Std accuracy"]),
                "Selected seed count": int(selected["Seed count"]),
                "Selected seeds": selected["Seeds"],
                "Selected compute cost": int(selected["Compute cost"]),
                "Best epochs": int(best_row["Local epochs"]),
                "Best round": int(best_row["Round"]),
                "Best accuracy": best_accuracy,
                "Accuracy gap to best": best_accuracy - float(selected["Mean accuracy"]),
            }
        )

    return (
        pd.DataFrame(plateau_rows).sort_values(epoch_columns),
        pd.DataFrame(selected_rows).sort_values(case_columns),
    )


def compute_epoch_round_selection_metrics(
    summary: pd.DataFrame,
    *,
    tolerance: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return curve diagnostics, optimal pairs, and marginal gains."""
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    data = _summary_with_compute_cost(summary)
    epoch_columns = _epoch_group_columns()
    case_columns = _case_group_columns()
    diagnostic_rows = []
    marginal_rows = []

    for group_values, group in data.groupby(epoch_columns, sort=False):
        curve = group.sort_values("Round")
        best_row = curve.loc[curve["Mean accuracy"].idxmax()]
        final_row = curve.iloc[-1]
        best_accuracy = float(best_row["Mean accuracy"])
        best_round = int(best_row["Round"])
        later_rounds = curve[curve["Round"] > best_round]
        worst_later_degradation = (
            best_accuracy - float(later_rounds["Mean accuracy"].min())
            if not later_rounds.empty
            else 0.0
        )
        base = dict(zip(epoch_columns, group_values))
        diagnostic_rows.append(
            base
            | {
                "Best round": best_round,
                "Best accuracy": best_accuracy,
                "Best compute cost": int(best_row["Compute cost"]),
                "Final round": int(final_row["Round"]),
                "Final accuracy": float(final_row["Mean accuracy"]),
                "Post-peak degradation": best_accuracy - float(final_row["Mean accuracy"]),
                "Worst later degradation": worst_later_degradation,
                "Later rounds hurt > tolerance": worst_later_degradation > tolerance,
            }
        )
        previous = None
        for _, row in curve.iterrows():
            if previous is not None:
                marginal_rows.append(
                    base
                    | {
                        "Round": int(row["Round"]),
                        "Accuracy": float(row["Mean accuracy"]),
                        "Previous round": int(previous["Round"]),
                        "Previous accuracy": float(previous["Mean accuracy"]),
                        "Marginal accuracy gain": float(row["Mean accuracy"]) - float(previous["Mean accuracy"]),
                        "Compute cost": int(row["Compute cost"]),
                    }
                )
            previous = row

    optimal_rows = []
    for group_values, group in data.groupby(case_columns, sort=False):
        best_row = group.loc[group["Mean accuracy"].idxmax()]
        best_accuracy = float(best_row["Mean accuracy"])
        eligible = group[group["Mean accuracy"] >= best_accuracy - tolerance].copy()
        efficient = eligible.sort_values(
            ["Compute cost", "Round", "Local epochs", "Mean accuracy"],
            ascending=[True, True, True, False],
        ).iloc[0]
        communication_efficient = eligible.sort_values(
            ["Round", "Compute cost", "Local epochs", "Mean accuracy"],
            ascending=[True, True, True, False],
        ).iloc[0]
        optimal_rows.append(
            dict(zip(case_columns, group_values))
            | {
                "Selected epochs": int(efficient["Local epochs"]),
                "Selected round": int(efficient["Round"]),
                "Selected accuracy": float(efficient["Mean accuracy"]),
                "Selected compute cost": int(efficient["Compute cost"]),
                "Best epochs": int(best_row["Local epochs"]),
                "Best round": int(best_row["Round"]),
                "Best accuracy": best_accuracy,
                "Best compute cost": int(best_row["Compute cost"]),
                "Accuracy gap to best": best_accuracy - float(efficient["Mean accuracy"]),
                "Communication-efficient epochs": int(communication_efficient["Local epochs"]),
                "Communication-efficient round": int(communication_efficient["Round"]),
                "Communication-efficient accuracy": float(communication_efficient["Mean accuracy"]),
                "Communication-efficient compute cost": int(communication_efficient["Compute cost"]),
                "Communication-efficient gap to best": best_accuracy - float(communication_efficient["Mean accuracy"]),
                "Near-best tolerance": float(tolerance),
            }
        )

    return (
        pd.DataFrame(diagnostic_rows).sort_values(epoch_columns),
        pd.DataFrame(optimal_rows).sort_values(case_columns),
        pd.DataFrame(marginal_rows).sort_values(epoch_columns + ["Round"]) if marginal_rows else pd.DataFrame(),
    )


def manifest_records(requests: pd.DataFrame) -> list[dict[str, int | str]]:
    """Return deduplicated manifest records from a request frame."""
    if requests.empty:
        return []
    missing = [column for column in MANIFEST_COLUMNS if column not in requests.columns]
    if missing:
        raise ValueError(f"Request frame is missing manifest columns: {missing}")
    return requests[MANIFEST_COLUMNS].drop_duplicates().to_dict("records")


def write_manifest(requests: pd.DataFrame, output_path: str | Path) -> None:
    """Write a tuning manifest TSV."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(manifest_records(requests))


def make_tuning_round_curves(
    summary: pd.DataFrame,
    dataset: str,
    model: str,
    setting: str | None = None,
):
    """Build a Plotly line chart for tuning curves."""
    import plotly.express as px

    data = summary[(summary["Dataset"] == dataset) & (summary["Model key"] == model)]
    if setting is not None:
        data = data[data["Setting key"] == setting]
    if data.empty:
        return None

    plot_data = data.copy()
    plot_data["Local epochs"] = plot_data["Local epochs"].astype(str)
    fig = px.line(
        plot_data,
        x="Round",
        y="Mean accuracy",
        color="Local epochs",
        line_dash="Variant",
        facet_col="Setting" if setting is None else None,
        markers=True,
        template="plotly_white",
        title=f"{DATASET_LABELS.get(dataset, dataset)} {MODEL_LABELS.get(model, model)} tuning curves",
        labels={"Mean accuracy": "Accuracy (%)", "Local epochs": "Local epochs"},
    )
    fig.update_layout(height=480, legend_title_text="")
    fig.update_xaxes(dtick=1)
    return fig


def _normalize_accuracy(value: float) -> float:
    return value * 100 if value <= 1 else value


def _read_scores(path: Path) -> list[float]:
    scores = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            scores.append(_normalize_accuracy(float(line)))
    return scores


def _score_path(seed_dir: Path) -> Path | None:
    candidates = [seed_dir / "10" / "log.txt", seed_dir / "10log.txt"]
    return next((path for path in candidates if path.exists()), None)


def _parse_seed(seed_dir: Path) -> int | None:
    if not seed_dir.name.startswith("seed"):
        return None
    seed_text = seed_dir.name.removeprefix("seed")
    return int(seed_text) if seed_text.isdigit() else None


def _iter_base_paths(base_dir: str | Path | Iterable[str | Path]):
    if isinstance(base_dir, (str, Path)):
        yield Path(base_dir)
    else:
        for path in base_dir:
            yield Path(path)


def _iter_tuning_run_dirs(base_dir: str | Path | Iterable[str | Path]):
    search_roots = []
    for base_path in _iter_base_paths(base_dir):
        search_roots.append(base_path)
        grouped_root = base_path / "epoch_round_tuning"
        if grouped_root.is_dir():
            search_roots.append(grouped_root)
    seen = set()
    for root in search_roots:
        for run_dir in sorted(root.glob("tuning-*")):
            if not run_dir.is_dir():
                continue
            key = run_dir.resolve(strict=False)
            if key in seen:
                continue
            seen.add(key)
            yield run_dir


def _iter_live_log_paths(
    log_dir: str | Path | Iterable[str | Path] = "logs",
    pattern: str = "vflora_tuning_*.out",
):
    raw_paths = [log_dir] if isinstance(log_dir, (str, Path)) else list(log_dir)
    seen = set()
    for raw_path in raw_paths:
        path = Path(raw_path)
        path_text = str(raw_path)
        if any(marker in path_text for marker in "*?[]"):
            candidates = path.parent.glob(path.name)
        elif path.is_dir():
            candidates = path.glob(pattern)
        elif path.exists():
            candidates = [path]
        else:
            candidates = []
        for candidate in sorted(candidates):
            key = candidate.resolve(strict=False)
            if key not in seen and candidate.is_file():
                seen.add(key)
                yield candidate


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _append_run_record(
    run_records: list[dict],
    *,
    variant: str,
    dataset: str,
    model: str,
    setting: str,
    epochs: int,
    rounds: int,
    seed: int,
    scores: list[float],
    run_dir: Path,
    score_path: Path,
    observed_rounds: int | None = None,
    complete_run: bool | None = None,
    result_source: str = "score log",
    run_status: str | None = None,
    extra_fields: dict | None = None,
) -> None:
    if not scores:
        return
    observed_rounds = len(scores) if observed_rounds is None else int(observed_rounds)
    complete_run = observed_rounds >= int(rounds) if complete_run is None else bool(complete_run)
    record = {
        "Variant key": variant,
        "Variant": VARIANT_LABELS.get(variant, variant),
        "Dataset": dataset,
        "Dataset label": DATASET_LABELS.get(dataset, dataset),
        "Model key": model,
        "Model": MODEL_LABELS.get(model, model),
        "Setting key": setting,
        "Setting": SETTING_LABELS.get(setting, setting),
        "Local epochs": int(epochs),
        "Config rounds": int(rounds),
        "Seed": int(seed),
        "Scores": scores,
        "Run dir": str(run_dir),
        "Score path": str(score_path),
        "Observed rounds": observed_rounds,
        "Complete run": complete_run,
        "Result source": result_source,
        "Run status": run_status or ("Complete" if complete_run else "Partial log"),
        "Score count": observed_rounds,
    }
    if extra_fields:
        record.update(extra_fields)
    run_records.append(record)


def _deduplicate_run_records(run_records: list[dict], include_run_dir: bool = False) -> list[dict]:
    if not run_records:
        return []
    id_columns = [
        "Variant key",
        "Dataset",
        "Model key",
        "Setting key",
        "Local epochs",
        "Seed",
    ]
    if include_run_dir:
        id_columns.append("Run dir")
    selected = []
    for _, group in pd.DataFrame(run_records).groupby(id_columns, sort=False):
        sort_columns = ["Score count"]
        if "Log mtime" in group.columns:
            sort_columns.append("Log mtime")
        else:
            sort_columns.append("Config rounds")
        selected.append(group.sort_values(sort_columns, ascending=False).iloc[0].to_dict())
    return selected


def _records_to_scores_frame(run_records: list[dict]) -> pd.DataFrame:
    if not run_records:
        return _empty_scores_frame()
    row_keys = [
        "Variant key",
        "Variant",
        "Dataset",
        "Dataset label",
        "Model key",
        "Model",
        "Setting key",
        "Setting",
        "Local epochs",
        "Config rounds",
        "Seed",
        "Run dir",
        "Score path",
        "Observed rounds",
        "Complete run",
        "Result source",
        "Run status",
    ]
    rows = []
    for record in run_records:
        for round_idx, accuracy in enumerate(record["Scores"], start=1):
            rows.append(
                {key: record[key] for key in row_keys if key in record}
                | {"Round": round_idx, "Accuracy": float(accuracy)}
            )
    return pd.DataFrame(rows).sort_values(
        ["Dataset", "Variant key", "Model key", "Setting key", "Local epochs", "Seed", "Round"]
    )


def _future_gain(curve: pd.DataFrame, round_value: int) -> float:
    current = curve.loc[curve["Round"] == round_value, "Mean accuracy"].iloc[0]
    future = curve[(curve["Round"] > round_value) & (curve["Round"] <= round_value + 2)]
    if future.empty:
        return 0.0
    return float(future["Mean accuracy"].max() - current)


def _summary_with_compute_cost(summary: pd.DataFrame) -> pd.DataFrame:
    data = summary.copy()
    if "Compute cost" not in data.columns:
        data["Compute cost"] = data["Local epochs"].astype(int) * data["Round"].astype(int)
    return data


def _case_group_columns() -> list[str]:
    return [
        "Variant key",
        "Variant",
        "Dataset",
        "Dataset label",
        "Model key",
        "Model",
        "Setting key",
        "Setting",
    ]


def _epoch_group_columns() -> list[str]:
    return _case_group_columns() + ["Local epochs"]


def _empty_scores_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Variant key",
            "Variant",
            "Dataset",
            "Dataset label",
            "Model key",
            "Model",
            "Setting key",
            "Setting",
            "Local epochs",
            "Config rounds",
            "Seed",
            "Round",
            "Accuracy",
            "Run dir",
            "Score path",
            "Observed rounds",
            "Complete run",
            "Result source",
            "Run status",
        ]
    )


def _empty_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=_case_group_columns()
        + [
            "Local epochs",
            "Round",
            "Mean accuracy",
            "Std accuracy",
            "Seed count",
            "Seeds",
            "Max config rounds",
            "Compute cost",
        ]
    )
