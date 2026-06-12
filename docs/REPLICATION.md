# Replicating V-FLoRA Variant Results

This repo ports reusable pieces from the working `FederatedLLM` experiment archive into a cleaner standalone implementation. The supported core paths are:

- model: `tinyllama`
- dataset: WizardLM-style client JSON files
- variants: linear cumulative FLoRA, nonlinear cumulative FLoRA, nonlinear FFA, and nonlinear RoLoRA
- aggregation: stacked residual adapters for FLoRA variants; B-only averaging for FFA
- model: `roberta-base`
- dataset: GLUE client JSON files
- variants: normal linear FLoRA, linear cumulative FLoRA, nonlinear cumulative FLoRA, and FFA

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

For GLUE/RoBERTa runs, create task-specific splits with the same data CLI:

```bash
python -m fed_adapter.cli.split_data \
  --dataset glue \
  --task-name mrpc \
  --mode stratified \
  --num-clients 10 \
  --output-root data_mrpc_stratified \
  --seed 0
```

MNLI writes `global_val.json` from `validation_matched` and `global_val_mismatched.json` from `validation_mismatched`.

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

## Nonlinear FFA

Use the same command with:

```bash
--variant nonlinear-ffa
```

FFA writes `A_frozen.bin` once at the run root, then writes each round's global
`B` to `adapter_model.bin`.

## Nonlinear RoLoRA

Use:

```bash
--variant nonlinear-rolora \
--calibration-path calibration_prompts.json
```

RoLoRA alternates B rounds and A rounds. B rounds average the active B factor
exactly. A rounds build an exact stacked nonlinear teacher, then distill it back
to the configured rank with prompt-only calibration records from
`--calibration-path`. Do not pass `global_test.json` as calibration data.

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

For `nonlinear-ffa`, `adapter_model_delta.bin` is not written because the global
state is the B-only adapter after averaging. The run root also contains
`A_frozen.bin` and `ffa_config.json`.

For `nonlinear-rolora`, each round writes the compressed shared adapter to
`adapter_model.bin`. A-round metadata includes distillation MSE diagnostics in
`round_config.json`. Exact teacher checkpoints are written only when
`--save-distill-teacher` is provided.

If `--eval-path` is provided, `log.txt` contains one score per communication round.

## GLUE/RoBERTa

Use the unified training CLI with `--task-family glue`. `--method` is accepted as a FederatedLLM-compatible alias for `--variant`.

```bash
python -m fed_adapter.cli.train \
  --task-family glue \
  --method flora \
  --model roberta-base \
  --task-name mrpc \
  --data-root data_mrpc_stratified \
  --output-dir runs/roberta-mrpc-flora/seed0 \
  --num-clients 10 \
  --rounds 20 \
  --rank 4 \
  --alpha 4 \
  --local-epochs 1 \
  --local-batch-size 32 \
  --micro-batch-size 16 \
  --learning-rate 5e-4 \
  --seed 0
```

Supported GLUE method names are:

```text
flora
linear_flora_cumulative
nonlinear_flora
ffa
```

The GLUE trainer writes `server_state.pt` and supports segmented resume:

```bash
python -m fed_adapter.cli.train \
  --task-family glue \
  --method flora \
  --model roberta-base \
  --task-name qnli \
  --data-root data_qnli_stratified \
  --output-dir runs/roberta-qnli-flora/seed0 \
  --num-clients 10 \
  --rounds 150 \
  --rank 4 \
  --alpha 4 \
  --local-epochs 20 \
  --max-rounds-per-invocation 10 \
  --retain-adapter-every-n-rounds 10 \
  --resume-from-latest \
  --seed 0
```

Do not pass `--resume-from-latest` for the first segment unless `server_state.pt` already exists.

Primary GLUE scores follow the experiment reporting rule: MNLI overall matched/mismatched accuracy, CoLA Matthew's correlation, STS-B Pearson correlation, and accuracy for all other tasks. MRPC and QQP therefore use accuracy as the score written to `log.txt`; their F1 and combined scores are retained in each round's `round_config.json`.


## SLURM

Example SLURM launchers are available in `scripts/`:

```bash
sbatch scripts/run_vflora_examples.slurm
sbatch --array=1-9 scripts/run_epoch_round_tuning.slurm
```

See `docs/SLURM.md` for details.
