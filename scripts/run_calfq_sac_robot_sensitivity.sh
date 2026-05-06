#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/run/logs"
mkdir -p "${LOG_DIR}"

IFS=',' read -r -a GPUS <<< "${GPUS:-0,1}"
SEEDS=(${SEEDS:-1 2 3 4 5 6 7 8 9 10})
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-3000000}"
DRY_RUN="${DRY_RUN:-0}"
ENV_ID="RobotNavigationConstSpeedCatch-v0"

COMMON_ARGS=(
  --env-id "${ENV_ID}"
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --num-envs 1
  --buffer-size 1000000
  --gamma 0.99
  --tau 0.005
  --batch-size 256
  --learning-starts 5000
  --policy-lr 0.0003
  --q-lr 0.001
  --policy-frequency 2
  --target-network-frequency 1
  --alpha 0.2
  --autotune
  --rolling-average-window 20
  --calfq-critic-improvement-threshold 0.01
  --calfq-anneal
  --calfq-anneal-frac 0.9
  --no-calfq-selective-buffer
  --torch-deterministic
  --no-capture-video
)

# Format: series:p_relax_init:p_relax_decay.
# The paper compares the default SAC schedule to this higher-trust schedule.
SERIES_SPECS=(
  "default:0.9:0.96"
  "pr08_d0995:0.8:0.995"
)

run_index=0

for spec in "${SERIES_SPECS[@]}"; do
  IFS=':' read -r series p_relax_init p_relax_decay <<< "${spec}"
  for seed in "${SEEDS[@]}"; do
    gpu="${GPUS[$((run_index % ${#GPUS[@]}))]}"
    session="calfq_sac_robot_${series}_s${seed}_g${gpu}"
    experiment="train_sac_calfq_sensitivity_${series}__${ENV_ID}"
    log_file="${LOG_DIR}/${session}.log"

    cmd=(
      uv run python run/train_sac_calfq.py
      "${COMMON_ARGS[@]}"
      --seed "${seed}"
      --device "cuda:${gpu}"
      --calfq-p-relax-init "${p_relax_init}"
      --calfq-p-relax-decay "${p_relax_decay}"
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
done

echo "prepared ${run_index} CALF-SAC robot sensitivity runs across GPUs: ${GPUS[*]}"
