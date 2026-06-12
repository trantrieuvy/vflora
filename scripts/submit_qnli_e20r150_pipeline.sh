#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export REPO_DIR
export MANIFEST="${MANIFEST:-tuning_manifests/roberta_qnli_stratified_flora_ffa_rank4_seed0_e20_r150.tsv}"
export RUN_ROOT="${RUN_ROOT:-./epoch_round_tuning_qnli_client_count_e20_r150}"
export JOB_PREFIX="${JOB_PREFIX:-qnli_e20r150}"
export TOTAL_SEGMENTS="${TOTAL_SEGMENTS:-15}"
export SEGMENT_ROUNDS="${SEGMENT_ROUNDS:-10}"
export SBATCH_MEM="${SBATCH_MEM:-64G}"
export ROBERTA_RETAIN_ADAPTER_EVERY_N_ROUNDS="${ROBERTA_RETAIN_ADAPTER_EVERY_N_ROUNDS:-0}"

exec bash "${REPO_DIR}/scripts/submit_roberta_glue_pipeline.sh" "$@"
