#!/bin/bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/prepare_glue_splits.sh TASK_NAME [NUM_CLIENTS...]

Examples:
  bash scripts/prepare_glue_splits.sh qnli 3 10 20
  TASK_NAME=mrpc OUTPUT_ROOT=data_mrpc_stratified bash scripts/prepare_glue_splits.sh mrpc 10

Environment overrides:
  MODE=stratified
  SEED=0
  OUTPUT_ROOT=data_${TASK_NAME}_${MODE}
  SOURCE_SPLIT_DIR=/path/to/existing/split
  DRY_RUN=true
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

TASK_NAME="${1,,}"
shift

if [[ $# -gt 0 ]]; then
  CLIENT_COUNTS=("$@")
else
  CLIENT_COUNTS=(3 10 20)
fi

MODE="${MODE:-stratified}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data_${TASK_NAME}_${MODE}}"
DRY_RUN="${DRY_RUN:-false}"

cd "${REPO_DIR}"

for num_clients in "${CLIENT_COUNTS[@]}"; do
  if ! [[ "${num_clients}" =~ ^[0-9]+$ && "${num_clients}" -gt 0 ]]; then
    echo "NUM_CLIENTS must be positive integers; got ${num_clients}." >&2
    exit 2
  fi

  cmd=(
    python -m fed_adapter.cli.split_data
    --dataset glue
    --task-name "${TASK_NAME}"
    --num-clients "${num_clients}"
    --output-root "${OUTPUT_ROOT}"
    --mode "${MODE}"
    --seed "${SEED}"
  )
  if [[ -n "${SOURCE_SPLIT_DIR:-}" ]]; then
    cmd+=(--source-split-dir "${SOURCE_SPLIT_DIR}")
  fi

  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
done
