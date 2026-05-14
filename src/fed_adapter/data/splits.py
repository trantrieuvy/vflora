"""Federated dataset split generation utilities."""

from __future__ import annotations

import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MIN_DIRICHLET_CLIENT_SIZE = 40


@dataclass(frozen=True)
class SplitRequest:
    dataset: str
    mode: str
    num_clients: int
    output_root: Path
    source_root: Path | None = None
    dataset_path: Path | None = None
    alpha: float = 0.5
    seed: int = 42
    test_per_category: int = 10
    num_length_buckets: int = 5


def create_split(request: SplitRequest) -> Path:
    random.seed(request.seed)
    if request.mode == "dirichlet":
        return create_dolly_dirichlet_split(request)
    if request.mode == "stratified_keep_sizes":
        if request.dataset == "dolly":
            return create_dolly_stratified_split(request)
        if request.dataset == "wizard":
            return create_wizard_stratified_split(request)
    raise ValueError(f"Unsupported split request: {request.dataset}/{request.mode}")


def create_dolly_dirichlet_split(request: SplitRequest) -> Path:
    if request.dataset != "dolly":
        raise ValueError("dirichlet mode currently supports only dataset='dolly'")
    if request.dataset_path is None:
        raise ValueError("dataset_path is required for Dolly dirichlet splitting")

    np = _require_numpy()
    pd = _require_pandas()
    np.random.seed(request.seed)

    df = pd.read_json(request.dataset_path, orient="records", lines=True)
    sorted_df = df.sort_values(by=["category"])
    sampled_df = (
        sorted_df.groupby("category")
        .apply(lambda group: group.sample(n=request.test_per_category, random_state=request.seed))
        .reset_index(level=0, drop=True)
    )
    remaining_df = sorted_df.drop(index=sampled_df.index)

    partitions = _dirichlet_partition_indices(remaining_df, request.num_clients, request.alpha)
    train_records = remaining_df.reset_index(drop=True).to_dict(orient="records")
    test_records = sampled_df.reset_index(drop=True).to_dict(orient="records")
    client_records = [
        remaining_df.loc[indices].reset_index(drop=True).to_dict(orient="records")
        for indices in partitions
    ]

    output_dir = _make_output_dir(request.output_root, request.num_clients)
    _write_split(
        output_dir,
        train_records,
        test_records,
        client_records,
        {
            "dataset": request.dataset,
            "split_mode": request.mode,
            "seed": request.seed,
            "alpha": request.alpha,
            "num_clients": request.num_clients,
            "global_test_policy": f"balanced_holdout_{request.test_per_category}_per_category",
            "client_size_policy": f"dirichlet_alpha_{request.alpha}",
            "client_sizes": [len(records) for records in client_records],
            "client_category_counts": _client_category_counts(client_records),
        },
    )
    return output_dir


def create_dolly_stratified_split(request: SplitRequest) -> Path:
    source_dir = _source_dir(request)
    _require_split_files(
        source_dir,
        request.num_clients,
        required_global_files=("global_training.json", "global_test.json"),
    )
    output_dir = _make_output_dir(request.output_root, request.num_clients)
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("output_root must differ from source_root")

    train_records = _read_json(source_dir / "global_training.json")
    test_records = _read_json(source_dir / "global_test.json")
    quotas = [len(_read_json(source_dir / f"local_training_{i}.json")) for i in range(request.num_clients)]
    client_records, label_counts = build_stratified_clients(
        train_records,
        quotas,
        seed=request.seed,
        labels=[record["category"] for record in train_records],
    )

    _write_split(
        output_dir,
        train_records,
        test_records,
        client_records,
        {
            "dataset": request.dataset,
            "split_mode": request.mode,
            "seed": request.seed,
            "source_root": str(request.source_root),
            "client_size_policy": "preserve_source_client_sizes",
            "client_sizes": [len(records) for records in client_records],
            "client_category_counts": {
                str(client_id): counts for client_id, counts in label_counts.items()
            },
        },
    )
    return output_dir


def create_wizard_stratified_split(request: SplitRequest) -> Path:
    source_dir = _source_dir(request)
    _require_split_files(source_dir, request.num_clients)
    output_dir = _make_output_dir(request.output_root, request.num_clients)
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("output_root must differ from source_root")

    source_clients = [_read_json(source_dir / f"local_training_{i}.json") for i in range(request.num_clients)]
    train_records = [record for client in source_clients for record in client]
    quotas = [len(records) for records in source_clients]
    labels, edges = wizard_stratification_labels(train_records, request.num_length_buckets)
    client_records, label_counts = build_stratified_clients(
        train_records,
        quotas,
        seed=request.seed,
        labels=labels,
    )

    test_records = _read_json(source_dir / "global_test.json") if (source_dir / "global_test.json").exists() else []
    _write_split(
        output_dir,
        train_records if (source_dir / "global_training.json").exists() else [],
        test_records,
        client_records,
        {
            "dataset": request.dataset,
            "split_mode": request.mode,
            "seed": request.seed,
            "source_root": str(request.source_root),
            "client_size_policy": "preserve_source_client_sizes",
            "stratification_policy": "task_family_plus_instruction_length_bucket",
            "instruction_length_bucket_edges": edges,
            "client_sizes": [len(records) for records in client_records],
            "global_label_counts": _counts(labels),
            "client_label_counts": {
                str(client_id): counts for client_id, counts in label_counts.items()
            },
        },
    )
    return output_dir


def build_stratified_clients(
    records: list[dict],
    quotas: list[int],
    seed: int,
    labels: list[str],
) -> tuple[list[list[dict]], dict[int, dict[str, int]]]:
    if len(records) != len(labels):
        raise ValueError("records and labels must have the same length")
    if sum(quotas) != len(records):
        raise ValueError("client quotas must sum to the number of records")

    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for record, label in zip(records, labels):
        by_label[label].append(record)

    label_names = sorted(by_label, key=lambda label: (-len(by_label[label]), label))
    for label in label_names:
        rng.shuffle(by_label[label])

    remaining = list(quotas)
    clients = [[] for _ in quotas]
    counts = {client_id: {} for client_id in range(len(quotas))}

    for label_index, label in enumerate(label_names):
        label_records = by_label[label]
        allocation = (
            list(remaining)
            if label_index == len(label_names) - 1
            else _allocate_counts(len(label_records), remaining)
        )
        offset = 0
        for client_id, count in enumerate(allocation):
            if count == 0:
                continue
            clients[client_id].extend(label_records[offset : offset + count])
            counts[client_id][label] = count
            remaining[client_id] -= count
            offset += count
        if offset != len(label_records):
            raise ValueError(f"records left unassigned for label {label}")

    if any(remaining):
        raise ValueError("stratified allocation did not fill every client quota")
    for client in clients:
        rng.shuffle(client)
    return clients, counts


def wizard_stratification_labels(records: Iterable[dict], num_length_buckets: int) -> tuple[list[str], list[int]]:
    records = list(records)
    lengths = [_word_count(record.get("instruction", "")) for record in records]
    edges = _quantile_edges(lengths, num_length_buckets)
    labels = [
        f"{_wizard_task_family(record)}__instruction_len_q{_bucket_for(length, edges)}"
        for record, length in zip(records, lengths)
    ]
    return labels, edges


def _dirichlet_partition_indices(remaining_df, num_clients: int, alpha: float) -> list[list[int]]:
    np = _require_numpy()
    min_size = 0
    num_rows = len(remaining_df)
    categories = remaining_df["category"].unique().tolist()

    while min_size < MIN_DIRICHLET_CLIENT_SIZE:
        partitions = [[] for _ in range(num_clients)]
        for category in categories:
            indices = remaining_df.loc[remaining_df["category"] == category].index.values
            np.random.shuffle(indices)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array(
                [
                    share * (len(client_indices) < num_rows / num_clients)
                    for share, client_indices in zip(proportions, partitions)
                ]
            )
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
            partitions = [
                client_indices + split.tolist()
                for client_indices, split in zip(partitions, np.split(indices, split_points))
            ]
            min_size = min(len(client_indices) for client_indices in partitions)
    return partitions


def _allocate_counts(total: int, capacities: list[int]) -> list[int]:
    if total > sum(capacities):
        raise ValueError("total exceeds remaining capacities")
    if total == 0:
        return [0 for _ in capacities]

    capacity_sum = sum(capacities)
    raw = [(total * capacity) / capacity_sum for capacity in capacities]
    allocation = [math.floor(value) for value in raw]
    remainder = total - sum(allocation)
    order = sorted(
        range(len(capacities)),
        key=lambda i: (-(raw[i] - allocation[i]), -capacities[i], i),
    )
    while remainder:
        for client_id in order:
            if allocation[client_id] < capacities[client_id]:
                allocation[client_id] += 1
                remainder -= 1
                break
        else:
            raise ValueError("unable to allocate remaining records")
    return allocation


def _wizard_task_family(record: dict) -> str:
    text = str(record.get("instruction", "")).lower()
    markers = {
        "code": ("python", "javascript", "java ", "c++", "sql", "html", "css", "code", "program"),
        "math": ("calculate", "equation", "solve", "probability", "statistics", "formula"),
        "rewrite_translate": ("translate", "rewrite", "rephrase", "paraphrase", "grammar", "edit"),
        "analysis_classification": ("summarize", "summary", "extract", "classify", "sentiment"),
        "creative_writing": ("story", "poem", "song", "script", "dialogue", "creative"),
        "recommendation_advice": ("recommend", "suggest", "advice", "tips", "best way"),
        "planning": ("plan", "schedule", "itinerary", "steps", "strategy", "process"),
        "structured_data": ("table", "json", "csv", "excel", "database", "chart"),
    }
    for family, family_markers in markers.items():
        if any(marker in text for marker in family_markers):
            return family
    if text.startswith(("what ", "why ", "how ", "when ", "where ", "who ")):
        return "qa_explanation"
    return "general"


def _word_count(text: object) -> int:
    return len(str(text).split())


def _quantile_edges(values: list[int], num_buckets: int) -> list[int]:
    if num_buckets < 1:
        raise ValueError("num_length_buckets must be at least 1")
    if num_buckets == 1 or not values:
        return []
    sorted_values = sorted(values)
    edges = []
    for index in range(1, num_buckets):
        position = (index / num_buckets) * (len(sorted_values) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            edges.append(sorted_values[lower])
        else:
            fraction = position - lower
            edges.append(int(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction))
    return sorted(set(edges))


def _bucket_for(value: int, edges: list[int]) -> int:
    for bucket, edge in enumerate(edges):
        if value <= edge:
            return bucket
    return len(edges)


def _require_split_files(
    source_dir: Path,
    num_clients: int,
    required_global_files: tuple[str, ...] = (),
) -> None:
    missing = [source_dir / filename for filename in required_global_files]
    missing.extend(
        source_dir / f"local_training_{client_id}.json"
        for client_id in range(num_clients)
    )
    missing = [path for path in missing if not path.exists()]
    if not missing:
        return

    missing_preview = "\n".join(f"  - {path}" for path in missing[:5])
    if len(missing) > 5:
        missing_preview += f"\n  ... and {len(missing) - 5} more"
    raise FileNotFoundError(
        "Could not find the source federated split files.\n"
        f"Expected source split directory: {source_dir.resolve()}\n"
        f"Current working directory: {Path.cwd()}\n"
        "For relative paths, run the command from the repository root, or pass "
        "an absolute --source-root path.\n"
        "Missing files:\n"
        f"{missing_preview}"
    )

def _source_dir(request: SplitRequest) -> Path:
    if request.source_root is None:
        raise ValueError("source_root is required for stratified_keep_sizes")
    return request.source_root / str(request.num_clients)


def _make_output_dir(root: Path, num_clients: int) -> Path:
    output_dir = root / str(num_clients)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_split(
    output_dir: Path,
    train_records: list[dict],
    test_records: list[dict],
    client_records: list[list[dict]],
    metadata: dict,
) -> None:
    if train_records:
        _write_json(output_dir / "global_training.json", train_records)
    if test_records:
        _write_json(output_dir / "global_test.json", test_records)
    for client_id, records in enumerate(client_records):
        _write_json(output_dir / f"local_training_{client_id}.json", records)
    metadata = {
        **metadata,
        "global_training_size": len(train_records),
        "global_test_size": len(test_records),
    }
    _write_json(output_dir / "split_metadata.json", metadata)


def _client_category_counts(client_records: list[list[dict]]) -> dict[str, dict[str, int]]:
    return {
        str(client_id): _counts(record["category"] for record in records)
        for client_id, records in enumerate(client_records)
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(Counter(values).items())}


def _read_json(path: Path) -> list[dict]:
    with path.open() as infile:
        return json.load(infile)


def _write_json(path: Path, payload: object) -> None:
    with path.open("w") as outfile:
        json.dump(payload, outfile)


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Dolly dirichlet splitting requires numpy") from exc
    return np


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Dolly dirichlet splitting requires pandas") from exc
    return pd

