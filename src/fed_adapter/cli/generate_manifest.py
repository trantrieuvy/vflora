"""Generate tab-separated manifests for epoch/round tuning jobs."""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path


MANIFEST_COLUMNS = ["variant", "dataset", "model", "setting", "epochs", "rounds", "seed"]


DEFAULT_VARIANTS = [
    "linear-cumulative-flora",
    "nonlinear-cumulative-flora",
    "nonlinear-ffa",
]
DEFAULT_DATASETS = ["wiz", "wiz_stratified", "dolly_stratified"]
DEFAULT_SETTINGS = ["homo", "heter"]
DEFAULT_EPOCHS = [1, 2, 3, 5]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["smoke", "tinyllama-coarse", "custom"],
        default="tinyllama-coarse",
    )
    parser.add_argument("--output", type=Path, default=Path("tuning_manifests/tinyllama_coarse.tsv"))
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=["tinyllama"])
    parser.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS)
    parser.add_argument("--epochs", nargs="+", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    return parser


def manifest_rows(args: argparse.Namespace) -> list[dict[str, int | str]]:
    if args.phase == "smoke":
        return [
            {
                "variant": variant,
                "dataset": "wiz",
                "model": "tinyllama",
                "setting": "homo",
                "epochs": 1,
                "rounds": 1,
                "seed": 0,
            }
            for variant in DEFAULT_VARIANTS
        ]

    rows = []
    if args.phase == "tinyllama-coarse":
        variants = DEFAULT_VARIANTS
        datasets = DEFAULT_DATASETS
        models = ["tinyllama"]
        settings = DEFAULT_SETTINGS
        epochs = DEFAULT_EPOCHS
        seeds = [0]
    else:
        variants = args.variants
        datasets = args.datasets
        models = args.models
        settings = args.settings
        epochs = args.epochs
        seeds = args.seeds

    for variant, dataset, model, setting, epoch_count, seed in product(
        variants,
        datasets,
        models,
        settings,
        epochs,
        seeds,
    ):
        rows.append(
            {
                "variant": variant,
                "dataset": dataset,
                "model": model,
                "setting": setting,
                "epochs": int(epoch_count),
                "rounds": int(args.rounds),
                "seed": int(seed),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = manifest_rows(args)
    write_manifest(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


def write_manifest(rows: list[dict[str, int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
