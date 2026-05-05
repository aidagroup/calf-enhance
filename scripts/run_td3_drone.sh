#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/run/logs"
mkdir -p "${LOG_DIR}"

IFS=',' read -r -a GPUS <<< "${GPUS:-0,1}"
SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9})
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-3000000}"
DRY_RUN="${DRY_RUN:-0}"
ENV_ID="UnderwaterDrone-v0"
ENV_SHORT="drone"

COMMON_ARGS=(
  --env-id "${ENV_ID}"
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --learning-rate 0.0003
  --num-envs 1
  --buffer-size 1000000
  --gamma 0.99
  --tau 0.005
  --batch-size 256
  --policy-noise 0.2
  --exploration-noise 0.1
  --learning-starts 25000
  --policy-frequency 2
  --noise-clip 0.5
  --rolling-average-window 20
  --torch-deterministic
  --no-capture-video
)

CALFQ_ARGS=(
  --calfq-critic-improvement-threshold 0.01
  --calfq-p-relax-init 0.9
  --calfq-p-relax-decay 0.96
  --calfq-anneal
  --calfq-anneal-frac 0.9
  --no-calfq-selective-buffer
)

RUN_SPECS=(
  "td3:run/train_td3.py:train_td3"
  "rrl_td3:run/train_td3_residual.py:train_td3_residual"
  "calfq_td3:run/train_calfq.py:train_calfq"
)

run_index=0

launch_run() {
  local algo="$1"
  local script="$2"
  local experiment_prefix="$3"
  local seed="$4"
  local gpu="${GPUS[$((run_index % ${#GPUS[@]}))]}"
  local session="td3_${algo}_${ENV_SHORT}_s${seed}_g${gpu}"
  local experiment="${experiment_prefix}__${ENV_ID}"
  local log_file="${LOG_DIR}/${session}.log"

  local -a cmd=(
    uv run python "${script}"
    "${COMMON_ARGS[@]}"
    --seed "${seed}"
    --device "cuda:${gpu}"
    --mlflow.experiment-name "${experiment}"
    --mlflow.run-name "${session}"
  )
  if [[ "${algo}" == "calfq_td3" ]]; then
    cmd+=("${CALFQ_ARGS[@]}")
  fi

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
}

for spec in "${RUN_SPECS[@]}"; do
  IFS=':' read -r algo script experiment_prefix <<< "${spec}"
  for seed in "${SEEDS[@]}"; do
    launch_run "${algo}" "${script}" "${experiment_prefix}" "${seed}"
  done
done

echo "prepared ${run_index} TD3-family ${ENV_SHORT} runs across GPUs: ${GPUS[*]}"
