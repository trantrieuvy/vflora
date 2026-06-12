# V-FLoRA: Federated Fine-Tuning of LoRA Variants

V-FLoRA studies strategies for federated instruction tuning with LoRA-style adapter variants. The repo is organized around finding strong federated tuning choices across adapter methods, dataset stratification, and local-epoch/communication-round schedules.

## Highlights

- Federated fine-tuning for instruction-following LLMs and GLUE/RoBERTa with LoRA-style adapters.
- Linear cumulative FLoRA, where each round adds a fresh linear residual adapter.
- Nonlinear cumulative FLoRA, where each residual adapter uses `B * gelu(A * x)`.
- Nonlinear FFA, where `A` is frozen and only `B` is trained and averaged.
- Nonlinear RoLoRA, where training alternates between `A` and `B` with distillation on A-update rounds.
- Heterogeneous client ranks for non-IID federated settings.
- Dolly, WizardLM, and GLUE client split utilities, including stratified client allocation.
- Epoch/round tuning workflow based on manifest files and analysis helpers.

## Repository Layout

```text
src/fed_adapter/
  aggregation.py          # FedAvg, zero-padding, linear/nonlinear adapter stacking
  client.py               # backend-driven federated client abstraction
  config.py               # dataclass experiment configuration
  selection.py            # deterministic client participation policy
  adapters/base.py        # adapter backend interface
  adapters/ffa.py         # frozen-A/trainable-B FFA adapter
  adapters/flora.py       # normal linear FLoRA merge helpers
  adapters/rolora.py      # nonlinear RoLoRA alternating-factor adapter
  adapters/residual.py    # linear/nonlinear cumulative FLoRA layers
  analysis/tuning.py      # epoch/round tuning result parsing and plotting
  cli/generate_manifest.py # tuning manifest generation
  cli/train.py            # unified instruction/GLUE training entry point
  cli/train_glue.py       # federated RoBERTa GLUE training
  cli/split_data.py       # federated split generation
  data/prompting.py       # built-in Alpaca-style prompt formatter
  data/schema.py          # Dolly/Alpaca/instruction-output normalization
  data/splits.py          # Dolly/Wizard/GLUE split generation
tests/                    # focused unit tests for the rewritten core
docs/                     # method, replication, and extraction notes
```

## Method Overview

V-FLoRA compares adapter methods, data splitting strategies, and epoch/round schedules to identify strong federated tuning strategies.

Current method names:

- `linear-cumulative-flora`
- `nonlinear-cumulative-flora`
- `nonlinear-ffa`
- `nonlinear-rolora` (`rolora` alias)
- GLUE aliases: `flora`, `linear_flora_cumulative`, `nonlinear_flora`, `ffa`
- `layercraft-variants` - optional backend planned for broader adapter sweeps

See `docs/METHODS.md` for formulas and Mermaid diagrams showing client updates, server aggregation, and cumulative adapter state.

## Installation

V-FLoRA is intended to run from its own Conda environment, separate from the
FederatedLLM `flora` environment:

```bash
conda create -n vflora python=3.10
conda activate vflora
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

The same environment can also be created from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate vflora
pip install -e . --no-build-isolation
```

`--no-build-isolation` keeps the editable install on the already-created
cluster environment instead of asking pip to download build dependencies into a
temporary isolated environment.

LayerCraft is not required for the current direct implementations of `linear-cumulative-flora`, `nonlinear-cumulative-flora`, `nonlinear-ffa`, and `nonlinear-rolora`. It will be used as an optional backend for LayerCraft adapter-variant experiments:

```bash
pip install git+https://github.com/trantrieuvy/layercraft.git
```

## Data Preparation

Generated datasets are not committed by default. See `docs/DATA.md` for the full WizardLM and Dolly preparation workflow. Current split modes:

- Dolly Dirichlet client partitioning.
- Dolly stratified partitioning while preserving legacy client sizes.
- WizardLM stratified partitioning by task family and instruction length.
- GLUE IID or stratified partitioning for `cola`, `mnli`, `mrpc`, `qnli`, `qqp`, `rte`, `sst2`, `stsb`, and `wnli`.

Run split commands from the repository root, or pass absolute paths. For `stratified_keep_sizes`, `--source-root` must already contain an existing federated split such as `data_wiz/10/local_training_0.json` through `local_training_9.json`.

For WizardLM experiments, V-FLoRA follows the project workflow used in the FederatedLLM fork: start from the pre-generated Wizard split (`data_wiz/10/local_training_*.json`), combine those client files, then redistribute them with the stratified keep-sizes splitter. For Dolly experiments, generate `data_dolly` from raw Dolly first, then create `data_dolly_stratified` from that source split.

Example:

```bash
python -m fed_adapter.cli.split_data \
  --dataset wizard \
  --mode stratified_keep_sizes \
  --num-clients 10 \
  --source-root data_wiz \
  --output-root data_wiz_stratified
```

GLUE splits use the same standalone CLI:

```bash
python -m fed_adapter.cli.split_data \
  --dataset glue \
  --task-name qnli \
  --mode stratified \
  --num-clients 10 \
  --output-root data_qnli_stratified \
  --seed 0
```

## Quickstart

Run the lightweight test suite:

```bash
pytest
```

Run the first nonlinear cumulative FLoRA replication path:

```bash
python -m fed_adapter.cli.train \
  --variant nonlinear-cumulative-flora \
  --model tinyllama \
  --data-root data_wiz \
  --output-dir runs/nonlinear-tinyllama-wiz \
  --num-clients 10 \
  --rounds 3 \
  --rank 16 \
  --alpha 32 \
  --eval-path mmlu_test_1444.jsonl \
  --seed 0
```

Use `--variant linear-cumulative-flora` for the cumulative linear residual variant, `--variant nonlinear-ffa` for frozen-A/trainable-B FFA, or `--variant nonlinear-rolora` for nonlinear RoLoRA. RoLoRA also accepts the short alias `rolora`; runs with more than one round require `--calibration-path` for prompt-only A-round distillation and currently use homogeneous rank only. Add `--heterogeneous --local-ranks 64,32,16,16,8,8,4,4,4,4` for heterogeneous-rank runs supported by the cumulative FLoRA and FFA variants.

Run a GLUE/RoBERTa federated workflow through the same entry point:

```bash
python -m fed_adapter.cli.train \
  --task-family glue \
  --method flora \
  --model roberta-base \
  --task-name mrpc \
  --data-root data_mrpc_stratified \
  --output-dir runs/glue-mrpc-flora \
  --num-clients 10 \
  --rounds 3 \
  --rank 4 \
  --alpha 4 \
  --local-epochs 1 \
  --seed 0
```

GLUE reports MNLI matched/mismatched overall accuracy, CoLA Matthew's correlation, STS-B Pearson correlation, and accuracy for the other tasks, including MRPC and QQP.

See `docs/REPLICATION.md` for the full run recipe. See `docs/SLURM.md` for example SLURM scripts.

Monitor segmented GLUE runs from the repository root with:

```bash
jupyter lab glue_high_round_monitoring.ipynb
```

The notebook defaults to the QNLI e20/r150 client-count run and updates from
completed `round_config.json` files while the pipeline is active.

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
