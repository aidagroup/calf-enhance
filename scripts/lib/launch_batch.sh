#!/usr/bin/env bash

: "${ROOT_DIR:?ROOT_DIR must be set before sourcing launch_batch.sh}"
: "${ENV_ID:?ENV_ID must be set before sourcing launch_batch.sh}"
: "${ENV_SHORT:?ENV_SHORT must be set before sourcing launch_batch.sh}"
: "${TRAIN_SCRIPT:?TRAIN_SCRIPT must be set before sourcing launch_batch.sh}"
: "${EXPERIMENT_PREFIX:?EXPERIMENT_PREFIX must be set before sourcing launch_batch.sh}"
: "${SESSION_PREFIX:?SESSION_PREFIX must be set before sourcing launch_batch.sh}"
: "${BATCH_LABEL:?BATCH_LABEL must be set before sourcing launch_batch.sh}"

LOG_DIR="${ROOT_DIR}/run/logs"
mkdir -p "${LOG_DIR}"

IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0,1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "GPUS must contain at least one GPU id" >&2
  exit 1
fi

run_index=0

for seed in "${SEEDS[@]}"; do
  gpu="${GPU_LIST[$((run_index % ${#GPU_LIST[@]}))]}"
  session="${SESSION_PREFIX}_s${seed}_g${gpu}"
  experiment="${EXPERIMENT_PREFIX}__${ENV_ID}"
  log_file="${LOG_DIR}/${session}.log"

  cmd=(
    uv run python "${TRAIN_SCRIPT}"
    "${COMMON_ARGS[@]}"
    "${EXTRA_ARGS[@]}"
    --seed "${seed}"
    --device "cuda:${gpu}"
    --mlflow.experiment-name "${experiment}"
    --mlflow.run-name "${session}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'dry-run %s: ' "${session}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    tmux new-session -d -s "${session}" \
      "cd '${ROOT_DIR}' && ${cmd[*]} 2>&1 | tee '${log_file}'"
    echo "started ${session}: ${cmd[*]}"
  fi

  run_index=$((run_index + 1))
done

echo "prepared ${run_index} ${BATCH_LABEL} runs across GPUs: ${GPU_LIST[*]}"
