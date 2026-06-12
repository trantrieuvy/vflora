# QNLI e20/r150 V-FLoRA Smoke

- Status: canceled after verification.
- Submitted: 2026-06-12 19:04:32 CEST.
- Last checked: 2026-06-12 19:06 CEST.
- Workdir: `/homes/neumann/trieu.vy.tran/vflora`.
- Submit command: `scripts/submit_qnli_e20r150_pipeline.sh 1 15`.
- Launcher: `scripts/submit_qnli_e20r150_pipeline.sh` -> `scripts/submit_roberta_glue_pipeline.sh`.
- Slurm script: `scripts/run_epoch_round_tuning.slurm`.
- Manifest: `tuning_manifests/roberta_qnli_stratified_flora_ffa_rank4_seed0_e20_r150.tsv`.
- Run root: `./epoch_round_tuning_qnli_client_count_e20_r150`.
- Logs: `logs/%x_%A_%a.out`, `logs/%x_%A_%a.err`.

## Submission Shape

- Submitted 90 single-row array jobs: 15 ten-round segments x 6 manifest rows.
- Job names: `qnli_e20r150_s001_r01` through `qnli_e20r150_s015_r06`.
- Job IDs: `40015` through `40104`.
- Segment 1 exports `ROBERTA_RESUME_FROM_LATEST=false`.
- Segments 2-15 export `ROBERTA_RESUME_FROM_LATEST=true` and `FORCE=true`.
- All segments export `ROBERTA_MAX_ROUNDS_PER_INVOCATION=10`, `ROBERTA_RETAIN_ADAPTER_EVERY_N_ROUNDS=0`, `RUN_ROOT=./epoch_round_tuning_qnli_client_count_e20_r150`, and the e20/r150 QNLI manifest.
- The helper also adds a lane dependency chain in addition to per-row segment dependencies, preserving the configured helper cap.

## Slurm Metadata

- Partition: `ampere`.
- GPU request: `gres/gpu:a100:1`.
- CPUs: `8`.
- Memory: `64G` from the submitter command-line override.
- Time limit: `2-00:00:00`.
- Conda environment: `vflora` from `scripts/run_epoch_round_tuning.slurm` default `CONDA_ENV`.

## Data Prep

Prepared in the standalone `vflora` environment:

```bash
conda run -n vflora bash scripts/prepare_glue_splits.sh qnli 3 10 20
```

Observed split metadata:

- `data_qnli_stratified/3`: 104,743 train records, 5,463 validation records, balanced label counts.
- `data_qnli_stratified/10`: 104,743 train records, 5,463 validation records, balanced label counts.
- `data_qnli_stratified/20`: 104,743 train records, 5,463 validation records, balanced label counts.

## Live Observation

Immediately after submission:

- `40015_1` through `40018_4` were running on `gpunode03`.
- `40019_[5]` and `40020_[6]` were pending for resources.
- `40021_[1]` through `40104_[6]` were pending on dependencies.
- `scontrol show job 40015` confirmed the submit line, workdir, stdout/stderr paths, `mem=64G`, `gres/gpu:a100:1`, and the e20/r150 manifest export.
- `scontrol show job 40021` confirmed segment-2 resume exports and `afterok:40015`.

The running jobs entered the GLUE trainer before cancellation:

- `logs/qnli_e20r150_s001_r01_40015_1.out`: `flora`, QNLI, 3 clients, executing rounds `0..9`.
- `logs/qnli_e20r150_s001_r02_40016_2.out`: `flora`, QNLI, 10 clients, executing rounds `0..9`.
- `logs/qnli_e20r150_s001_r03_40017_3.out`: `flora`, QNLI, 20 clients, executing rounds `0..9`.
- `logs/qnli_e20r150_s001_r04_40018_4.out`: `ffa`, QNLI, 3 clients, executing rounds `0..9`.

Log sweep found no tracebacks, import errors, missing-data errors, or missing-manifest errors. The only terminal error lines were the expected Slurm cancellation messages.

## Cancellation

Canceled all submitted IDs explicitly:

```bash
scancel 40015 40016 40017 40018 40019 40020 40021 40022 40023 40024 40025 40026 40027 40028 40029 40030 40031 40032 40033 40034 40035 40036 40037 40038 40039 40040 40041 40042 40043 40044 40045 40046 40047 40048 40049 40050 40051 40052 40053 40054 40055 40056 40057 40058 40059 40060 40061 40062 40063 40064 40065 40066 40067 40068 40069 40070 40071 40072 40073 40074 40075 40076 40077 40078 40079 40080 40081 40082 40083 40084 40085 40086 40087 40088 40089 40090 40091 40092 40093 40094 40095 40096 40097 40098 40099 40100 40101 40102 40103 40104
```

Post-cancel checks:

- `squeue --me` showed no remaining jobs from this submission.
- `sacct` reported `CANCELLED by 21197` for `40015`-`40104`; only the first four had nonzero runtime, about one minute.
