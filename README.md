# V-FLoRA: Federated Fine-Tuning of LoRA Variants

V-FLoRA studies federated instruction tuning with heterogeneous, cumulative, and nonlinear low-rank adapter variants. It builds on FLoRA-style federated aggregation and extends the setup with controlled data partitioning and multi-round residual adapter composition.

## Highlights

- Federated fine-tuning for instruction-following LLMs with LoRA-style adapters.
- Heterogeneous client ranks for non-IID federated settings.
- Cumulative linear FLoRA, where each round adds a fresh residual adapter.
- Nonlinear FLoRA, where each residual adapter uses `B * gelu(A * x)`.
- Dolly and WizardLM client split utilities, including stratified client allocation.

## Repository Layout

```text
src/fed_adapter/
  aggregation.py          # FedAvg, zero-padding, linear/nonlinear adapter stacking
  client.py               # backend-driven federated client abstraction
  config.py               # dataclass experiment configuration
  selection.py            # deterministic client participation policy
  adapters/base.py        # adapter backend interface
  adapters/residual.py    # cumulative linear/nonlinear residual LoRA layers
  cli/train.py            # first replication training path
  cli/split_data.py       # federated split generation
  data/prompting.py       # built-in Alpaca-style prompt formatter
  data/schema.py          # Dolly/Alpaca/instruction-output normalization
  data/splits.py          # Dolly/Wizard split generation
tests/                    # focused unit tests for the rewritten core
docs/                     # replication and extraction notes
```

## Installation

For local development:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pytest
pip install -e .
```

For training:

```bash
pip install -r requirements-train.txt
```

## Data Preparation

Generated datasets are not committed by default. Current split modes:

- Dolly Dirichlet client partitioning.
- Dolly stratified partitioning while preserving legacy client sizes.
- WizardLM stratified partitioning by task family and instruction length.

Example:

```bash
python -m fed_adapter.cli.split_data ^
  --dataset wizard ^
  --mode stratified_keep_sizes ^
  --num-clients 10 ^
  --source-root data_wiz ^
  --output-root data_wiz_stratified
```

## Quickstart

Run the lightweight test suite:

```bash
pytest
```

Run the first nonlinear V-FLoRA replication path:

```bash
python -m fed_adapter.cli.train ^
  --variant nonlinear ^
  --model tinyllama ^
  --data-root data_wiz ^
  --output-dir runs/nonlinear-tinyllama-wiz ^
  --num-clients 10 ^
  --rounds 3 ^
  --rank 16 ^
  --alpha 32 ^
  --eval-path mmlu_test_1444.jsonl ^
  --seed 0
```

Use `--variant cumulative-linear` for the cumulative linear residual variant, and add `--heterogeneous --local-ranks 64,32,16,16,8,8,4,4,4,4` for heterogeneous-rank runs.

See `docs/REPLICATION.md` for the full recipe.

## Results

Add a compact results table here once the final experiment set is regenerated from this repository.

Recommended columns:

- Dataset
- Model
- Method
- Rank setting
- Rounds
- Seed count
- MMLU or task score
- Notes

## Attribution

This work builds on the FLoRA paper: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations. This repository is a new structured implementation for V-FLoRA experiments; the original public FederatedLLM/FLoRA codebase is cited for the baseline method and experimental context.

## License

A license will be added after clarifying upstream licensing constraints for any remaining inherited material.