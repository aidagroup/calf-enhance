#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/run/logs"
mkdir -p "${LOG_DIR}"

IFS=',' read -r -a GPUS <<< "${GPUS:-0,1}"
SEEDS=(${SEEDS:-0 1 2 3 4})
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-3000000}"
DRY_RUN="${DRY_RUN:-0}"

COMMON_ARGS=(
  --env-id UnderwaterDrone-v0
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
  --calfq-critic-improvement-threshold 0.01
  --calfq-anneal
  --calfq-selective-buffer
  --torch-deterministic
  --no-capture-video
)

# Format: series:p_relax_init:p_relax_decay:anneal_frac.
# These values reproduce the CALF-TD3 UnderwaterDrone ablation/sensitivity runs.
SERIES_SPECS=(
  "af015:0.8:0.995:0.15"
  "af040:0.8:0.995:0.40"
  "af065:0.8:0.995:0.65"
  "pr10_d0995:1.0:0.995:0.90"
  "pr08_d09995:0.8:0.9995:0.90"
  "pr10_d09995:1.0:0.9995:0.90"
)

run_index=0

for spec in "${SERIES_SPECS[@]}"; do
  IFS=':' read -r series p_relax_init p_relax_decay anneal_frac <<< "${spec}"
  for seed in "${SEEDS[@]}"; do
    gpu="${GPUS[$((run_index % ${#GPUS[@]}))]}"
    session="calfq_td3_drone_${series}_s${seed}_g${gpu}"
    experiment="train_calfq_ablation_${series}__UnderwaterDrone-v0"
    log_file="${LOG_DIR}/${session}.log"

    cmd=(
      uv run python run/train_calfq.py
      "${COMMON_ARGS[@]}"
      --seed "${seed}"
      --device "cuda:${gpu}"
      --calfq-p-relax-init "${p_relax_init}"
      --calfq-p-relax-decay "${p_relax_decay}"
      --calfq-anneal-frac "${anneal_frac}"
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

echo "prepared ${run_index} CALF-TD3 ablation/sensitivity runs across GPUs: ${GPUS[*]}"
