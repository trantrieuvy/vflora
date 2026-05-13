# V-FLoRA: Federated Fine-Tuning of LoRA Variants

V-FLoRA studies strategies for federated instruction tuning with LoRA-style adapter variants. The repo is organized around finding strong federated tuning choices across adapter methods, dataset stratification, and local-epoch/communication-round schedules.

## Highlights

- Federated fine-tuning for instruction-following LLMs with LoRA-style adapters.
- Linear cumulative FLoRA, where each round adds a fresh linear residual adapter.
- Nonlinear cumulative FLoRA, where each residual adapter uses `B * gelu(A * x)`.
- Nonlinear FFA, where `A` is frozen and only `B` is trained and averaged. Port in progress.
- Heterogeneous client ranks for non-IID federated settings.
- Dolly and WizardLM client split utilities, including stratified client allocation.
- Epoch/round tuning workflow based on manifest files. Port in progress.

## Repository Layout

```text
src/fed_adapter/
  aggregation.py          # FedAvg, zero-padding, linear/nonlinear adapter stacking
  client.py               # backend-driven federated client abstraction
  config.py               # dataclass experiment configuration
  selection.py            # deterministic client participation policy
  adapters/base.py        # adapter backend interface
  adapters/residual.py    # linear/nonlinear cumulative FLoRA layers
  cli/train.py            # first replication training path
  cli/split_data.py       # federated split generation
  data/prompting.py       # built-in Alpaca-style prompt formatter
  data/schema.py          # Dolly/Alpaca/instruction-output normalization
  data/splits.py          # Dolly/Wizard split generation
tests/                    # focused unit tests for the rewritten core
docs/                     # method, replication, and extraction notes
```

## Method Overview

V-FLoRA compares adapter methods, data splitting strategies, and epoch/round schedules to identify strong federated tuning strategies.

Current method names:

- `linear-cumulative-flora`
- `nonlinear-cumulative-flora`
- `nonlinear-ffa` - port in progress
- `layercraft-variants` - optional backend planned for broader adapter sweeps

See `docs/METHODS.md` for formulas and Mermaid diagrams showing client updates, server aggregation, and cumulative adapter state.

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

LayerCraft is not required for the current direct implementations of `linear-cumulative-flora` and `nonlinear-cumulative-flora`. It will be used as an optional backend for LayerCraft adapter-variant experiments:

```bash
pip install git+https://github.com/trantrieuvy/layercraft.git
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

Run the first nonlinear cumulative FLoRA replication path:

```bash
python -m fed_adapter.cli.train ^
  --variant nonlinear-cumulative-flora ^
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

Use `--variant linear-cumulative-flora` for the cumulative linear residual variant, and add `--heterogeneous --local-ranks 64,32,16,16,8,8,4,4,4,4` for heterogeneous-rank runs.

See `docs/REPLICATION.md` for the full run recipe.

## Results

Add a compact results table here once the final experiment set is regenerated from this repository.

Recommended columns:

- Dataset
- Model
- Method
- Rank setting
- Split strategy
- Local epochs
- Communication rounds
- Seed count
- MMLU or task score
- Notes

## Attribution

This work builds on the FLoRA paper: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations. This repository is a new structured implementation for V-FLoRA experiments; the original public FederatedLLM/FLoRA codebase is cited for the baseline method and experimental context.

## License

A license will be added after clarifying upstream licensing constraints for any remaining inherited material.