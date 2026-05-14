#!/bin/bash
#SBATCH --job-name=vflora_smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL

# Minimal SLURM smoke test for the federated training CLI.
# It checks: conda env, imports, data path, model loading, one federated round,
# and output writing. By default it skips evaluation to keep the run short.

CONDA_ENV="${CONDA_ENV:-vflora}"
DATA_ROOT="${DATA_ROOT:-data_wiz}"
RUN_ROOT="${RUN_ROOT:-runs/smoke}"
MODEL="${MODEL:-tinyllama}"
VARIANT="${VARIANT:-nonlinear-cumulative-flora}"
SEED="${SEED:-0}"
NUM_CLIENTS="${NUM_CLIENTS:-10}"
CLIENT_FRACTION="${CLIENT_FRACTION:-0.2}"
ROUNDS="${ROUNDS:-1}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
DEV_DATA_PATH="${DEV_DATA_PATH:-}"

mkdir -p logs "${RUN_ROOT}"

module load conda
conda activate "${CONDA_ENV}"

python - <<'PY'
import fed_adapter
print("fed_adapter import OK")
PY

if [[ ! -d "${DATA_ROOT}/${NUM_CLIENTS}" ]]; then
  echo "Missing ${DATA_ROOT}/${NUM_CLIENTS}. See docs/DATA.md before launching." >&2
  exit 1
fi

cmd=(
  python -m fed_adapter.cli.train
  --variant "${VARIANT}"
  --model "${MODEL}"
  --data-root "${DATA_ROOT}"
  --output-dir "${RUN_ROOT}/${VARIANT}-${MODEL}/seed${SEED}"
  --num-clients "${NUM_CLIENTS}"
  --client-fraction "${CLIENT_FRACTION}"
  --rounds "${ROUNDS}"
  --rank 16
  --alpha 32
  --local-epochs "${LOCAL_EPOCHS}"
  --local-batch-size "${LOCAL_BATCH_SIZE}"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --learning-rate 3e-4
  --seed "${SEED}"
)

if [[ -n "${DEV_DATA_PATH}" ]]; then
  cmd+=(--eval-path "${DEV_DATA_PATH}")
fi

echo "Running smoke command:"
printf '%q ' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

echo "Smoke test completed. Outputs written under ${RUN_ROOT}."
