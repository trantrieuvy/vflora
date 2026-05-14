#!/bin/bash
#SBATCH --job-name=vflora_tuning
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END

set -euo pipefail

MANIFEST="${MANIFEST:-tuning_manifests/tinyllama_example.tsv}"
CONDA_ENV="${CONDA_ENV:-vflora}"
DEV_DATA_PATH="${DEV_DATA_PATH:-mmlu_test_1444.jsonl}"
RUN_ROOT="${RUN_ROOT:-runs/epoch_round_tuning}"
FORCE="${FORCE:-false}"
DRY_RUN="${DRY_RUN:-false}"
MANIFEST_INDEX_BASE="${MANIFEST_INDEX_BASE:-1}"

mkdir -p logs "${RUN_ROOT}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "true" ]]; then
  module load conda
  conda activate "${CONDA_ENV}"
fi

run_row() {
  local variant="$1"
  local dataset="$2"
  local model="$3"
  local setting="$4"
  local epochs="$5"
  local rounds="$6"
  local seed="$7"

  local data_root="data_${dataset}"
  local output_dir="${RUN_ROOT%/}/tuning-${variant}-${dataset}-${model}-${setting}-e${epochs}-r${rounds}/seed${seed}"
  local log_path="${output_dir}/10/log.txt"
  local extra_args=()

  if [[ ! -d "${data_root}/10" ]]; then
    echo "Missing ${data_root}/10. See docs/DATA.md before launching." >&2
    exit 1
  fi

  if [[ "${FORCE}" != "true" && -s "${log_path}" ]]; then
    echo "Skipping completed run: ${output_dir}"
    return
  fi

  case "${setting}" in
    homo)
      ;;
    heter)
      extra_args+=(--heterogeneous --local-ranks "64,32,16,16,8,8,4,4,4,4")
      ;;
    *)
      echo "Unknown setting: ${setting}" >&2
      exit 1
      ;;
  esac

  local cmd=(
    python -m fed_adapter.cli.train
    --variant "${variant}"
    --model "${model}"
    --data-root "${data_root}"
    --output-dir "${output_dir}"
    --num-clients 10
    --rounds "${rounds}"
    --rank 16
    --alpha 32
    --local-epochs "${epochs}"
    --local-batch-size 128
    --micro-batch-size 16
    --learning-rate 3e-4
    --eval-path "${DEV_DATA_PATH}"
    --seed "${seed}"
    "${extra_args[@]}"
  )

  echo "============================================"
  echo "variant=${variant} dataset=${dataset} model=${model} setting=${setting}"
  echo "epochs=${epochs} rounds=${rounds} seed=${seed}"
  echo "output_dir=${output_dir}"
  echo "============================================"

  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  line_number=$((SLURM_ARRAY_TASK_ID + 2 - MANIFEST_INDEX_BASE))
  row="$(sed -n "${line_number}p" "${MANIFEST}")"
  if [[ -z "${row}" ]]; then
    echo "No manifest row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
  fi
  IFS=$'\t' read -r variant dataset model setting epochs rounds seed <<< "${row}"
  run_row "${variant}" "${dataset}" "${model}" "${setting}" "${epochs}" "${rounds}" "${seed}"
else
  while IFS=$'\t' read -r variant dataset model setting epochs rounds seed; do
    [[ -z "${variant}" || "${variant}" == "variant" ]] && continue
    run_row "${variant}" "${dataset}" "${model}" "${setting}" "${epochs}" "${rounds}" "${seed}"
  done < "${MANIFEST}"
fi