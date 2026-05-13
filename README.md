# V-FLoRA: Federated Fine-Tuning of LoRA Variants

V-FLoRA studies federated instruction tuning with heterogeneous, cumulative, and nonlinear low-rank adapter variants. It builds on FLoRA-style federated aggregation and extends the setup with controlled data partitioning, LayerCraft adapter variants, and multi-round residual adapter composition.

## Highlights

- Federated fine-tuning for instruction-following LLMs with LoRA-style adapters.
- Heterogeneous client ranks for non-IID federated settings.
- Cumulative linear FLoRA, where each round adds a fresh residual adapter.
- Nonlinear FLoRA, where each residual adapter uses `B * sigma(A * x)`.
- LayerCraft integration for swapping adapter implementations while preserving the federated training loop.
- Dolly and WizardLM client split utilities, including stratified client allocation.

## Repository Layout

```text
src/fed_adapter/
  aggregation.py          # FedAvg, zero-padding, linear/nonlinear adapter stacking
  client.py               # backend-driven federated client abstraction
  config.py               # dataclass experiment configuration
  selection.py            # deterministic client participation policy
  adapters/base.py        # adapter backend interface
  data/schema.py          # Dolly/Alpaca/instruction-output normalization
tests/                    # CPU-focused unit tests for the rewritten core
docs/                     # extraction and rewrite planning notes
```

## Installation

The current scaffold is intentionally lightweight. For local development:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pytest
```

Install PyTorch in the environment where you want to run the tensor aggregation tests and future LLM training code.

## Data Preparation

Generated datasets are not committed by default. The original split-generation script will be ported into `src/fed_adapter/data/` after the training core is in place.

Planned split modes:

- Dolly Dirichlet client partitioning.
- Dolly stratified partitioning while preserving legacy client sizes.
- WizardLM stratified partitioning by task family and instruction length.

## Quickstart

Run the current lightweight test suite:

```bash
pytest
```

The current repository contains the rewritten core modules first. Full LLM training entrypoints will be added after the adapter backend and training loop are ported into the new structure.

## Experiments

- Experiment 1: expressivity comparison between baseline linear FLoRA, doubled-rank linear adapters, and nonlinear adapters.
- Experiment 2: multi-round nonlinear FLoRA with fresh residual adapters per communication round.
- Epoch/round tuning: compare local epoch count and communication-round tradeoffs.
- Data split comparison: legacy Dirichlet splits versus stratified client-preserving splits.

## Results

Add a compact results table here once the final experiment set is decided.

Recommended columns:

- Dataset
- Model
- Method
- Rank setting
- Rounds
- Seed count
- MMLU or task score
- Notes

Figures can live under `assets/figures/`.

## Attribution

This work builds on FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations. Portions of the baseline training flow are adapted from the original FLoRA implementation. The adapter experiments also use PEFT, Hugging Face Transformers, and LayerCraft.

## License

Add the license after checking the upstream FLoRA repository license and any LayerCraft licensing requirements.

