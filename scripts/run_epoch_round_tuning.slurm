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

MANIFEST="${MANIFEST:-tuning_manifests/tinyllama_example.tsv}"
CONDA_ENV="${CONDA_ENV:-vflora}"
DEV_DATA_PATH="${DEV_DATA_PATH:-mmlu_test_1444.jsonl}"
RUN_ROOT="${RUN_ROOT:-runs/epoch_round_tuning}"
FORCE="${FORCE:-false}"
DRY_RUN="${DRY_RUN:-false}"
MANIFEST_INDEX_BASE="${MANIFEST_INDEX_BASE:-1}"
ROBERTA_LORA_R="${ROBERTA_LORA_R:-8}"
ROBERTA_LORA_ALPHA="${ROBERTA_LORA_ALPHA:-8}"
ROBERTA_RESUME_FROM_LATEST="${ROBERTA_RESUME_FROM_LATEST:-false}"
ROBERTA_MAX_ROUNDS_PER_INVOCATION="${ROBERTA_MAX_ROUNDS_PER_INVOCATION:-0}"
ROBERTA_RETAIN_ADAPTER_EVERY_N_ROUNDS="${ROBERTA_RETAIN_ADAPTER_EVERY_N_ROUNDS:-1}"
TINYLLAMA_RESUME_FROM_LATEST="${TINYLLAMA_RESUME_FROM_LATEST:-false}"
TINYLLAMA_MAX_ROUNDS_PER_INVOCATION="${TINYLLAMA_MAX_ROUNDS_PER_INVOCATION:-0}"
TINYLLAMA_RETAIN_ADAPTER_EVERY_N_ROUNDS="${TINYLLAMA_RETAIN_ADAPTER_EVERY_N_ROUNDS:-1}"

mkdir -p logs "${RUN_ROOT}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "true" ]]; then
  module load conda
  conda activate "${CONDA_ENV}"
fi

is_true() {
  case "${1:-}" in
    true|True|TRUE|1|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

glue_task_for_dataset() {
  local dataset="$1"
  case "${dataset}" in
    cola|cola_stratified) echo "cola" ;;
    mnli|mnli_stratified) echo "mnli" ;;
    mrpc|mrpc_stratified) echo "mrpc" ;;
    qnli|qnli_stratified) echo "qnli" ;;
    qqp|qqp_stratified) echo "qqp" ;;
    rte|rte_stratified|rte_dirichlet) echo "rte" ;;
    sst2|sst2_stratified) echo "sst2" ;;
    stsb|stsb_stratified) echo "stsb" ;;
    wnli|wnli_stratified) echo "wnli" ;;
    *) echo "${dataset}" ;;
  esac
}

heter_rank_list_for() {
  local model="$1"
  local num_clients="$2"
  local base_ranks=()
  if [[ "${model}" == "roberta-base" || "${model}" == "roberta" ]]; then
    base_ranks=(32 16 8 8 4 4 2 2 2 2)
  else
    base_ranks=(64 32 16 16 8 8 4 4 4 4)
  fi
  local fallback_rank="${base_ranks[$((${#base_ranks[@]} - 1))]}"
  local ranks=()
  local client_id
  for ((client_id = 0; client_id < num_clients; client_id++)); do
    if (( client_id < ${#base_ranks[@]} )); then
      ranks+=("${base_ranks[${client_id}]}")
    else
      ranks+=("${fallback_rank}")
    fi
  done
  local joined=""
  local rank
  for rank in "${ranks[@]}"; do
    if [[ -z "${joined}" ]]; then
      joined="${rank}"
    else
      joined="${joined},${rank}"
    fi
  done
  printf '%s\n' "${joined}"
}

run_row() {
  local variant="$1"
  local dataset="$2"
  local model="$3"
  local setting="$4"
  local epochs="$5"
  local rounds="$6"
  local seed="$7"
  local num_clients="${8:-10}"
  local row_rank="${9:-}"
  local row_alpha="${10:-}"
  local local_val_size="${11:-0}"
  local local_train_monitor_size="${12:-0}"

  local data_root="data_${dataset}"
  local output_dir="${RUN_ROOT%/}/tuning-${variant}-${dataset}-${model}-${setting}-e${epochs}-r${rounds}/seed${seed}"
  local log_path="${output_dir}/${num_clients}/log.txt"
  local extra_args=()
  local task_family="instruction"

  if [[ "${model}" == "roberta" || "${model}" == "roberta-base" ]]; then
    task_family="glue"
    model="roberta-base"
    if [[ "${variant}" == "flora" ]]; then
      log_path="${output_dir}/${num_clients}log.txt"
    fi
  fi

  if [[ ! -d "${data_root}/${num_clients}" ]]; then
    echo "Missing ${data_root}/${num_clients}. See docs/DATA.md before launching." >&2
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
      extra_args+=(--heterogeneous --local-ranks "$(heter_rank_list_for "${model}" "${num_clients}")")
      ;;
    *)
      echo "Unknown setting: ${setting}" >&2
      exit 1
      ;;
  esac

  local cmd=()
  if [[ "${task_family}" == "glue" ]]; then
    local effective_rank="${row_rank:-${ROBERTA_LORA_R}}"
    local effective_alpha="${row_alpha:-${ROBERTA_LORA_ALPHA}}"
    cmd=(
      python -m fed_adapter.cli.train
      --task-family glue
      --method "${variant}"
      --model "${model}"
      --task-name "$(glue_task_for_dataset "${dataset}")"
      --data-root "${data_root}"
      --output-dir "${output_dir}"
      --num-clients "${num_clients}"
      --rounds "${rounds}"
      --rank "${effective_rank}"
      --alpha "${effective_alpha}"
      --local-epochs "${epochs}"
      --local-batch-size 32
      --micro-batch-size 16
      --learning-rate 5e-4
      --warmup-ratio 0.06
      --weight-decay 0.1
      --cutoff-len 512
      --local-val-size "${local_val_size}"
      --local-train-monitor-size "${local_train_monitor_size}"
      --max-rounds-per-invocation "${ROBERTA_MAX_ROUNDS_PER_INVOCATION}"
      --retain-adapter-every-n-rounds "${ROBERTA_RETAIN_ADAPTER_EVERY_N_ROUNDS}"
      --seed "${seed}"
      "${extra_args[@]}"
    )
    if is_true "${ROBERTA_RESUME_FROM_LATEST}"; then
      cmd+=(--resume-from-latest)
    fi
  else
    cmd=(
      python -m fed_adapter.cli.train
      --task-family instruction
      --variant "${variant}"
      --model "${model}"
      --data-root "${data_root}"
      --output-dir "${output_dir}"
      --num-clients "${num_clients}"
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
    if is_true "${TINYLLAMA_RESUME_FROM_LATEST}"; then
      cmd+=(--resume-from-latest)
    fi
    cmd+=(
      --max-rounds-per-invocation "${TINYLLAMA_MAX_ROUNDS_PER_INVOCATION}"
      --retain-adapter-every-n-rounds "${TINYLLAMA_RETAIN_ADAPTER_EVERY_N_ROUNDS}"
    )
  fi

  echo "============================================"
  echo "variant=${variant} dataset=${dataset} model=${model} setting=${setting}"
  echo "epochs=${epochs} rounds=${rounds} seed=${seed} num_clients=${num_clients}"
  echo "output_dir=${output_dir}"
  echo "local_val_size=${local_val_size} local_train_monitor_size=${local_train_monitor_size}"
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
  IFS=$'\t' read -r variant dataset model setting epochs rounds seed num_clients rank alpha local_val_size local_train_monitor_size <<< "${row}"
  run_row "${variant}" "${dataset}" "${model}" "${setting}" "${epochs}" "${rounds}" "${seed}" "${num_clients}" "${rank}" "${alpha}" "${local_val_size}" "${local_train_monitor_size}"
else
  while IFS=$'\t' read -r variant dataset model setting epochs rounds seed num_clients rank alpha local_val_size local_train_monitor_size; do
    [[ -z "${variant}" || "${variant}" == "variant" || "${variant}" == "method" ]] && continue
    run_row "${variant}" "${dataset}" "${model}" "${setting}" "${epochs}" "${rounds}" "${seed}" "${num_clients}" "${rank}" "${alpha}" "${local_val_size}" "${local_train_monitor_size}"
  done < "${MANIFEST}"
fi
