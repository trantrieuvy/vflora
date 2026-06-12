#!/bin/bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/submit_roberta_glue_pipeline.sh START_SEGMENT [END_SEGMENT] --previous-job JOBID [--dry-run]
  bash scripts/submit_roberta_glue_pipeline.sh START_SEGMENT [END_SEGMENT] --previous-row-jobs ROW=JOB,... --previous-lane-jobs LANE=JOB,... [--dry-run]
  bash scripts/submit_roberta_glue_pipeline.sh 1 [END_SEGMENT] [--dry-run]

Submits one single-row Slurm array job per (segment, manifest row), with:
  - per-row afterok dependencies, so segment N row R waits for segment N-1 row R
  - a lane dependency chain, so at most CAP helper-submitted jobs can run at once
  - optional seeding from an active previous array job

Environment overrides:
  CAP=8
  SEGMENT_ROUNDS=10
  TOTAL_SEGMENTS=3
  RUN_ROOT=./epoch_round_tuning_qnli_client_count_e20_r30
  MANIFEST=tuning_manifests/roberta_qnli_stratified_flora_ffa_rank4_seed0_e20_r30.tsv
  JOB_PREFIX=qnli_e20r30
  SBATCH_TIME=2-00:00:00
  SBATCH_MEM=64G
  ROBERTA_RETAIN_ADAPTER_EVERY_N_ROUNDS=0
USAGE
}

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MANIFEST="${MANIFEST:-tuning_manifests/roberta_qnli_stratified_flora_ffa_rank4_seed0_e20_r30.tsv}"
RUN_ROOT="${RUN_ROOT:-./epoch_round_tuning_qnli_client_count_e20_r30}"
CAP="${CAP:-8}"
SEGMENT_ROUNDS="${SEGMENT_ROUNDS:-10}"
TOTAL_SEGMENTS="${TOTAL_SEGMENTS:-3}"
RETAIN_ADAPTER_EVERY_N_ROUNDS="${ROBERTA_RETAIN_ADAPTER_EVERY_N_ROUNDS:-0}"
JOB_PREFIX="${JOB_PREFIX:-qnli_e20r30}"
SBATCH_MEM="${SBATCH_MEM:-64G}"
DRY_RUN="false"
PREVIOUS_JOB_ID=""
PREVIOUS_ROW_JOBS=""
PREVIOUS_LANE_JOBS=""

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

START_SEGMENT="$1"
shift
END_SEGMENT="${TOTAL_SEGMENTS}"

if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  END_SEGMENT="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --previous-job)
      if [[ $# -lt 2 ]]; then
        echo "--previous-job requires a Slurm job id." >&2
        exit 2
      fi
      PREVIOUS_JOB_ID="$2"
      shift
      ;;
    --previous-row-jobs)
      if [[ $# -lt 2 ]]; then
        echo "--previous-row-jobs requires ROW=JOB,... ." >&2
        exit 2
      fi
      PREVIOUS_ROW_JOBS="$2"
      shift
      ;;
    --previous-lane-jobs)
      if [[ $# -lt 2 ]]; then
        echo "--previous-lane-jobs requires LANE=JOB,... ." >&2
        exit 2
      fi
      PREVIOUS_LANE_JOBS="$2"
      shift
      ;;
    --dry-run)
      DRY_RUN="true"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

if ! [[ "${START_SEGMENT}" =~ ^[0-9]+$ && "${END_SEGMENT}" =~ ^[0-9]+$ ]]; then
  echo "START_SEGMENT and END_SEGMENT must be numeric." >&2
  exit 2
fi
if ! [[ "${CAP}" =~ ^[0-9]+$ && "${CAP}" -gt 0 ]]; then
  echo "CAP must be a positive integer; got ${CAP}." >&2
  exit 2
fi
if (( START_SEGMENT < 1 || END_SEGMENT > TOTAL_SEGMENTS || START_SEGMENT > END_SEGMENT )); then
  echo "Segments must satisfy 1 <= START <= END <= ${TOTAL_SEGMENTS}; got ${START_SEGMENT}-${END_SEGMENT}." >&2
  exit 2
fi
if [[ -n "${PREVIOUS_JOB_ID}" && -n "${PREVIOUS_ROW_JOBS}" ]]; then
  echo "Use either --previous-job or --previous-row-jobs, not both." >&2
  exit 2
fi
if [[ -n "${PREVIOUS_ROW_JOBS}" && -z "${PREVIOUS_LANE_JOBS}" ]]; then
  echo "--previous-row-jobs requires --previous-lane-jobs to preserve the global cap." >&2
  exit 2
fi
if [[ -n "${PREVIOUS_LANE_JOBS}" && -z "${PREVIOUS_ROW_JOBS}" ]]; then
  echo "--previous-lane-jobs requires --previous-row-jobs." >&2
  exit 2
fi
if (( START_SEGMENT > 1 )) && [[ -z "${PREVIOUS_JOB_ID}" && -z "${PREVIOUS_ROW_JOBS}" ]]; then
  echo "Starting after segment 1 requires --previous-job or --previous-row-jobs." >&2
  exit 2
fi
if [[ ! -f "${REPO_DIR}/${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

row_count="$(awk 'NR > 1 && NF {count++} END {print count + 0}' "${REPO_DIR}/${MANIFEST}")"
if (( row_count == 0 )); then
  echo "Manifest has no data rows: ${MANIFEST}" >&2
  exit 1
fi

declare -a lane_prev
declare -a lane_load
declare -A row_prev
declare -A active_row_lane

for lane in $(seq 1 "${CAP}"); do
  lane_prev["${lane}"]=""
  lane_load["${lane}"]=0
done

for row in $(seq 1 "${row_count}"); do
  row_prev["${row}"]=""
done

parse_previous_row_jobs() {
  [[ -z "${PREVIOUS_ROW_JOBS}" ]] && return

  local pair key value
  IFS=',' read -r -a pairs <<< "${PREVIOUS_ROW_JOBS}"
  for pair in "${pairs[@]}"; do
    key="${pair%%[=:]*}"
    value="${pair#*[=:]}"
    if ! [[ "${key}" =~ ^[0-9]+$ && "${value}" =~ ^[0-9]+$ ]]; then
      echo "Invalid --previous-row-jobs entry: ${pair}" >&2
      exit 2
    fi
    if (( key < 1 || key > row_count )); then
      echo "Previous row key ${key} is outside manifest row range 1-${row_count}." >&2
      exit 2
    fi
    row_prev["${key}"]="${value}"
  done

  for row in $(seq 1 "${row_count}"); do
    if [[ -z "${row_prev[${row}]}" ]]; then
      echo "--previous-row-jobs is missing row ${row}." >&2
      exit 2
    fi
  done
}

parse_previous_lane_jobs() {
  [[ -z "${PREVIOUS_LANE_JOBS}" ]] && return

  local pair key value
  IFS=',' read -r -a pairs <<< "${PREVIOUS_LANE_JOBS}"
  for pair in "${pairs[@]}"; do
    key="${pair%%[=:]*}"
    value="${pair#*[=:]}"
    if ! [[ "${key}" =~ ^[0-9]+$ && "${value}" =~ ^[0-9]+$ ]]; then
      echo "Invalid --previous-lane-jobs entry: ${pair}" >&2
      exit 2
    fi
    if (( key < 1 || key > CAP )); then
      echo "Previous lane key ${key} is outside lane range 1-${CAP}." >&2
      exit 2
    fi
    lane_prev["${key}"]="${value}"
    lane_load["${key}"]=1
  done

  for lane in $(seq 1 "${CAP}"); do
    if [[ -z "${lane_prev[${lane}]}" ]]; then
      echo "--previous-lane-jobs is missing lane ${lane}." >&2
      exit 2
    fi
  done
}

seed_active_previous_lanes() {
  [[ -z "${PREVIOUS_JOB_ID}" ]] && return

  local active_output
  if ! active_output="$(squeue -h -r -j "${PREVIOUS_JOB_ID}" -o '%i')"; then
    echo "Could not inspect active tasks for previous job ${PREVIOUS_JOB_ID}." >&2
    exit 1
  fi
  local active_ids=()
  if [[ -n "${active_output}" ]]; then
    mapfile -t active_ids <<< "${active_output}"
  fi

  local active_count="${#active_ids[@]}"
  if (( active_count > CAP )); then
    echo "Previous job ${PREVIOUS_JOB_ID} has ${active_count} queued/running tasks, which exceeds CAP=${CAP}." >&2
    exit 1
  fi

  local lane=1
  local job_id
  for job_id in "${active_ids[@]}"; do
    [[ -z "${job_id}" ]] && continue
    lane_prev["${lane}"]="${job_id}"
    lane_load["${lane}"]=1
    if [[ "${job_id}" =~ ^${PREVIOUS_JOB_ID}_([0-9]+)$ ]]; then
      local row="${BASH_REMATCH[1]}"
      active_row_lane["${row}"]="${lane}"
      row_prev["${row}"]="${job_id}"
    fi
    lane=$((lane + 1))
  done
}

join_dependencies() {
  local deps=("$@")
  local joined=""
  local dep
  for dep in "${deps[@]}"; do
    [[ -z "${dep}" ]] && continue
    if [[ -z "${joined}" ]]; then
      joined="${dep}"
    else
      joined="${joined}:${dep}"
    fi
  done
  printf '%s\n' "${joined}"
}

pick_lane_for() {
  local segment="$1"
  local row="$2"

  if (( segment == START_SEGMENT )) && [[ -n "${active_row_lane[${row}]:-}" ]]; then
    PICKED_LANE="${active_row_lane[${row}]}"
    lane_load["${PICKED_LANE}"]=$((lane_load["${PICKED_LANE}"] + 1))
    return
  fi

  local best_lane=1
  local best_load="${lane_load[1]}"
  local lane
  for lane in $(seq 2 "${CAP}"); do
    if (( lane_load["${lane}"] < best_load )); then
      best_lane="${lane}"
      best_load="${lane_load[${lane}]}"
    fi
  done
  PICKED_LANE="${best_lane}"
  lane_load["${PICKED_LANE}"]=$((lane_load["${PICKED_LANE}"] + 1))
}

submit_row() {
  local segment="$1"
  local row="$2"
  local lane="$3"
  shift 3
  local deps=("$@")

  local job_name
  job_name="$(printf '%s_s%03d_r%02d' "${JOB_PREFIX}" "${segment}" "${row}")"

  local resume="true"
  local force_export=",FORCE=true"
  if (( segment == 1 )); then
    resume="false"
    force_export=""
  fi

  local export_vars
  export_vars="ALL,MANIFEST=${MANIFEST},RUN_ROOT=${RUN_ROOT},ROBERTA_RESUME_FROM_LATEST=${resume},ROBERTA_MAX_ROUNDS_PER_INVOCATION=${SEGMENT_ROUNDS},ROBERTA_RETAIN_ADAPTER_EVERY_N_ROUNDS=${RETAIN_ADAPTER_EVERY_N_ROUNDS}${force_export}"

  local cmd=(
    sbatch
    --parsable
    "--job-name=${job_name}"
    "--array=${row}"
    "--export=${export_vars}"
  )

  if [[ -n "${SBATCH_TIME:-}" ]]; then
    cmd+=("--time=${SBATCH_TIME}")
  fi
  if [[ -n "${SBATCH_MEM:-}" ]]; then
    cmd+=("--mem=${SBATCH_MEM}")
  fi

  local joined_deps
  joined_deps="$(join_dependencies "${deps[@]}")"
  if [[ -n "${joined_deps}" ]]; then
    cmd+=("--dependency=afterok:${joined_deps}")
  fi

  cmd+=("${REPO_DIR}/scripts/run_epoch_round_tuning.slurm")

  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '%q ' "${cmd[@]}" >&2
    printf '\n' >&2
    printf 'DRY_s%03d_r%02d\n' "${segment}" "${row}"
    return
  fi

  local job_id
  if ! job_id="$("${cmd[@]}")"; then
    echo "${job_name}: sbatch failed; aborting pipeline submission." >&2
    exit 1
  fi
  echo "${job_name}: submitted ${job_id} on lane ${lane}; deps=${joined_deps:-none}" >&2
  printf '%s\n' "${job_id}"
}

cd "${REPO_DIR}"
parse_previous_row_jobs
parse_previous_lane_jobs
seed_active_previous_lanes

for segment in $(seq "${START_SEGMENT}" "${END_SEGMENT}"); do
  rows=()
  if (( segment == START_SEGMENT )); then
    for row in $(seq 1 "${row_count}"); do
      if [[ -n "${active_row_lane[${row}]:-}" ]]; then
        rows+=("${row}")
      fi
    done
    for row in $(seq 1 "${row_count}"); do
      if [[ -z "${active_row_lane[${row}]:-}" ]]; then
        rows+=("${row}")
      fi
    done
  else
    for row in $(seq 1 "${row_count}"); do
      rows+=("${row}")
    done
  fi

  for row in "${rows[@]}"; do
    pick_lane_for "${segment}" "${row}"
    lane="${PICKED_LANE}"
    deps=()
    if [[ -n "${row_prev[${row}]:-}" ]]; then
      deps+=("${row_prev[${row}]}")
    fi
    if [[ -n "${lane_prev[${lane}]:-}" ]]; then
      duplicate="false"
      for dep in "${deps[@]}"; do
        if [[ "${dep}" == "${lane_prev[${lane}]}" ]]; then
          duplicate="true"
        fi
      done
      if [[ "${duplicate}" == "false" ]]; then
        deps+=("${lane_prev[${lane}]}")
      fi
    fi

    job_id="$(submit_row "${segment}" "${row}" "${lane}" "${deps[@]}")"
    row_prev["${row}"]="${job_id}"
    lane_prev["${lane}"]="${job_id}"
  done
done
