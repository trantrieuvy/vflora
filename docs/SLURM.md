# SLURM Examples

These scripts are examples for clusters similar to the environment used for the original experiments: one A100 GPU on the `ampere` partition with a Conda environment.

Edit the `#SBATCH` lines to match your cluster before submitting.

## Smoke Test

Use this first to check that the SLURM environment, Conda environment, data path, imports, model loading, and one short federated round work:

```bash
sbatch scripts/smoke_train.slurm
```

The smoke test defaults to one communication round, `client_fraction=0.2`, smaller batch sizes, and no evaluation. Override settings as needed:

```bash
CONDA_ENV=vflora \
DATA_ROOT=data_wiz \
MODEL=tinyllama \
VARIANT=nonlinear-cumulative-flora \
sbatch scripts/smoke_train.slurm
```

Set `DEV_DATA_PATH=mmlu_test_1444.jsonl` if you also want to smoke-test evaluation.

## One Job With Several Example Runs

This runs a small set of currently implemented V-FLoRA methods sequentially in one allocation. For a smaller smoke test, comment out runs near the bottom of `scripts/run_vflora_examples.slurm`:

- `nonlinear-cumulative-flora`, homogeneous
- `nonlinear-cumulative-flora`, heterogeneous
- `linear-cumulative-flora`, homogeneous
- `linear-cumulative-flora`, heterogeneous
- `nonlinear-ffa`, homogeneous
- `nonlinear-ffa`, heterogeneous

```bash
sbatch scripts/run_vflora_examples.slurm
```

Common overrides:

```bash
CONDA_ENV=vflora \
DATA_ROOT=data_wiz \
DEV_DATA_PATH=mmlu_test_1444.jsonl \
RUN_ROOT=runs/slurm_examples \
SEED=0 \
sbatch scripts/run_vflora_examples.slurm
```

## Manifest-Based Epoch/Round Tuning

The manifest format is tab-separated:

```text
variant dataset model setting epochs rounds seed
```

FederatedLLM-style manifests are also accepted. They may use `method` as the first header and include extra columns:

```text
method dataset model setting epochs rounds seed num_clients lora_r lora_alpha local_val_set_size local_train_monitor_size
```

Example manifest:

```text
tuning_manifests/tinyllama_example.tsv
```

Run every row sequentially in one job:

```bash
sbatch scripts/run_epoch_round_tuning.slurm
```

Run rows as a SLURM array. The example manifest has nine data rows, so use `1-9`:

```bash
sbatch --array=1-9 scripts/run_epoch_round_tuning.slurm
```

Generate a larger manifest:

```bash
python -m fed_adapter.cli.generate_manifest \
  --phase tinyllama-coarse \
  --output tuning_manifests/tinyllama_coarse.tsv
```

Use a different manifest:

```bash
MANIFEST=tuning_manifests/my_manifest.tsv \
sbatch --array=1-20 scripts/run_epoch_round_tuning.slurm
```

Dry-run without launching Python training:

```bash
DRY_RUN=true bash scripts/run_epoch_round_tuning.slurm
```

For GLUE/RoBERTa rows, set `model` to `roberta` or `roberta-base`. The launcher dispatches to:

```bash
python -m fed_adapter.cli.train --task-family glue
```

Accepted GLUE methods in manifests are `flora`, `linear_flora_cumulative`, `nonlinear_flora`, and `ffa`. For `flora`, the completion check uses the legacy FederatedLLM score path `seedX/10log.txt`; other methods use `seedX/10/log.txt`.

Useful GLUE segmentation overrides:

```bash
ROBERTA_MAX_ROUNDS_PER_INVOCATION=10 \
ROBERTA_RETAIN_ADAPTER_EVERY_N_ROUNDS=10 \
ROBERTA_RESUME_FROM_LATEST=true \
MANIFEST=tuning_manifests/roberta_qnli.tsv \
sbatch --array=1-20 scripts/run_epoch_round_tuning.slurm
```

Leave `ROBERTA_RESUME_FROM_LATEST=false` for first segments that do not already have `server_state.pt`.

### QNLI E20/R30 Pipeline

The FederatedLLM QNLI sequence:

```bash
python scripts/create_glue_federated_split.py --task-name qnli --num-clients 3 --output-root data_qnli_stratified --mode stratified --seed 0
python scripts/create_glue_federated_split.py --task-name qnli --num-clients 10 --output-root data_qnli_stratified --mode stratified --seed 0
python scripts/create_glue_federated_split.py --task-name qnli --num-clients 20 --output-root data_qnli_stratified --mode stratified --seed 0
scripts/submit_qnli_e20r30_pipeline.sh 1 3
```

is now:

```bash
module load conda
conda activate flora

scripts/prepare_glue_splits.sh qnli 3 10 20
scripts/submit_qnli_e20r30_pipeline.sh 1 3
```

The QNLI wrapper uses:

```text
tuning_manifests/roberta_qnli_stratified_flora_ffa_rank4_seed0_e20_r30.tsv
```

It submits three 10-round segments for the six manifest rows. Segment 1 starts fresh; segments 2 and 3 resume from each row's `server_state.pt`.

## Required Data Layout

The scripts expect data roots like:

```text
data_wiz/10/local_training_0.json
...
data_wiz/10/local_training_9.json
```

For `dataset=wiz_stratified`, the runner expects:

```text
data_wiz_stratified/10/local_training_0.json
...
data_wiz_stratified/10/local_training_9.json
```

For a GLUE row such as `dataset=qnli_stratified`, the runner expects:

```text
data_qnli_stratified/10/local_training_0.json
...
data_qnli_stratified/10/local_training_9.json
data_qnli_stratified/10/global_val.json
```

For MNLI, `global_val_mismatched.json` is used when present.

See `docs/DATA.md` for dataset setup.

## Notes

- Instruction rows support `linear-cumulative-flora`, `nonlinear-cumulative-flora`, `nonlinear-ffa`, and `nonlinear-rolora`.
- GLUE rows support `flora`, `linear_flora_cumulative`, `nonlinear_flora`, and `ffa`.
- The scripts skip completed runs if the expected `log.txt` exists, unless `FORCE=true` is set.
- Outputs are written under `runs/` by default, which is ignored by Git.
