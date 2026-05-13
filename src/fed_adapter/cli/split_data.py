"""Generate federated data splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from fed_adapter.data.splits import SplitRequest, create_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate federated client data splits.")
    parser.add_argument("--dataset", choices=("dolly", "wizard"), required=True)
    parser.add_argument("--mode", choices=("dirichlet", "stratified_keep_sizes"), required=True)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-per-category", type=int, default=10)
    parser.add_argument("--num-length-buckets", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = create_split(
        SplitRequest(
            dataset=args.dataset,
            mode=args.mode,
            num_clients=args.num_clients,
            output_root=args.output_root,
            source_root=args.source_root,
            dataset_path=args.dataset_path,
            alpha=args.alpha,
            seed=args.seed,
            test_per_category=args.test_per_category,
            num_length_buckets=args.num_length_buckets,
        )
    )
    print(f"Wrote {args.dataset} split to {output_dir}")


if __name__ == "__main__":
    main()

