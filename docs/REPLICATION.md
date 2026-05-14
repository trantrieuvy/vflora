# Replicating V-FLoRA Variant Results

This repo is being ported from the working `FederatedLLM` experiment archive into a cleaner standalone implementation. The first supported path is:

- model: `tinyllama`
- dataset: WizardLM-style client JSON files
- variant: nonlinear cumulative FLoRA
- aggregation: stack `A` unweighted and stack weighted `B`

## Environment

```bash
conda create -n vflora python=3.10
conda activate vflora
pip install -e .
pip install -r requirements-train.txt
```

If the model requires Hugging Face authentication, set:

```bash
set HF_TOKEN=your_token
```

## Data

Generated data is intentionally not committed. Run commands from the repository root, or pass absolute paths. See `docs/DATA.md` for complete WizardLM and Dolly setup commands.

For WizardLM, V-FLoRA expects the same starting point used in the FederatedLLM experiments: an existing pre-generated 10-client Wizard split. In practice, copy or symlink `data_wiz` from the experiment archive/fork, or pass its absolute path via `--source-root`. The stratified splitter then combines the source clients and redistributes records while preserving the source client sizes.

The training CLI expects:

```text
data_wiz/10/local_training_0.json
...
data_wiz/10/local_training_9.json
```

To create a stratified Wizard split from an existing 10-client Wizard split, `--source-root` must already contain the source split:

```bash
python -m fed_adapter.cli.split_data \
  --dataset wizard \
  --mode stratified_keep_sizes \
  --num-clients 10 \
  --source-root data_wiz \
  --output-root data_wiz_stratified
```

## Nonlinear FLoRA

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
  --local-epochs 1 \
  --local-batch-size 128 \
  --micro-batch-size 16 \
  --learning-rate 3e-4 \
  --eval-path mmlu_test_1444.jsonl \
  --seed 0
```

## Cumulative Linear FLoRA

Use the same command with:

```bash
--variant linear-cumulative-flora
```

## Heterogeneous Ranks

Add:

```bash
--heterogeneous --local-ranks 64,32,16,16,8,8,4,4,4,4
```

## Outputs

Each round writes:

- `adapter_model_delta.bin`: the current round's aggregated adapter
- `adapter_model.bin`: cumulative frozen adapter through that round
- `round_config.json`: selected clients, ranks, weights, and parameter counts

If `--eval-path` is provided, `log.txt` contains one score per communication round.

