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

See `docs/DATA.md` for dataset setup.

## Notes

- `linear-cumulative-flora`, `nonlinear-cumulative-flora`, and `nonlinear-ffa` are supported by the current training CLI.
- The scripts skip completed runs if the expected `log.txt` exists, unless `FORCE=true` is set.
- Outputs are written under `runs/` by default, which is ignored by Git.
