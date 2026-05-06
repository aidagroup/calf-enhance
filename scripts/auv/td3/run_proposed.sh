#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-3000000}"
SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9})
ENV_ID="UnderwaterDrone-v0"
ENV_SHORT="drone"
TRAIN_SCRIPT="run/train_calfq.py"
EXPERIMENT_PREFIX="train_calfq"
SESSION_PREFIX="td3_calfq_td3_${ENV_SHORT}"
BATCH_LABEL="proposed TD3-backbone AUV"

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
EXTRA_ARGS=(
  --calfq-critic-improvement-threshold 0.01
  --calfq-p-relax-init 0.9
  --calfq-p-relax-decay 0.96
  --calfq-anneal
  --calfq-anneal-frac 0.9
  --no-calfq-selective-buffer
)

source "${ROOT_DIR}/scripts/lib/launch_batch.sh"
