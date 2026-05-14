#!/bin/bash
#SBATCH --job-name=vflora_examples
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END

# This script launches several demonstration runs sequentially in one allocation.
# Comment out any run_method lines at the bottom if you want a smaller smoke test.

CONDA_ENV="${CONDA_ENV:-vflora}"
DATA_ROOT="${DATA_ROOT:-data_wiz}"
DEV_DATA_PATH="${DEV_DATA_PATH:-mmlu_test_1444.jsonl}"
RUN_ROOT="${RUN_ROOT:-runs/slurm_examples}"
SEED="${SEED:-0}"
MODEL="${MODEL:-tinyllama}"
NUM_CLIENTS="${NUM_CLIENTS:-10}"
ROUNDS="${ROUNDS:-3}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"

mkdir -p logs "${RUN_ROOT}"

module load conda
conda activate "${CONDA_ENV}"

if [[ ! -d "${DATA_ROOT}/${NUM_CLIENTS}" ]]; then
  echo "Missing ${DATA_ROOT}/${NUM_CLIENTS}. See docs/DATA.md before launching." >&2
  exit 1
fi

if [[ ! -f "${DEV_DATA_PATH}" ]]; then
  echo "Warning: DEV_DATA_PATH not found: ${DEV_DATA_PATH}" >&2
  echo "The commands below will still run if you remove --eval-path or provide the file." >&2
fi

run_method() {
  local variant="$1"
  local setting="$2"
  local extra_args=()

  if [[ "${setting}" == "heter" ]]; then
    extra_args+=(--heterogeneous --local-ranks "64,32,16,16,8,8,4,4,4,4")
  fi

  local output_dir="${RUN_ROOT}/${variant}-${MODEL}-${setting}/seed${SEED}"

  echo "============================================"
  echo "variant=${variant} model=${MODEL} setting=${setting} seed=${SEED}"
  echo "output_dir=${output_dir}"
  echo "============================================"

  python -m fed_adapter.cli.train \
    --variant "${variant}" \
    --model "${MODEL}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${output_dir}" \
    --num-clients "${NUM_CLIENTS}" \
    --rounds "${ROUNDS}" \
    --rank 16 \
    --alpha 32 \
    --local-epochs "${LOCAL_EPOCHS}" \
    --local-batch-size 128 \
    --micro-batch-size 16 \
    --learning-rate 3e-4 \
    --eval-path "${DEV_DATA_PATH}" \
    --seed "${SEED}" \
    "${extra_args[@]}"
}

# Currently implemented in vflora.
run_method nonlinear-cumulative-flora homo
run_method nonlinear-cumulative-flora heter
run_method linear-cumulative-flora homo
run_method linear-cumulative-flora heter

# nonlinear-ffa will be added here once the method is ported.
