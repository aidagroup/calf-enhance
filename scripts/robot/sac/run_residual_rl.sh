#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-3000000}"
SEEDS=(${SEEDS:-1 2 3 4 5 6 7 8 9 10})
ENV_ID="RobotNavigationConstSpeedCatch-v0"
ENV_SHORT="robot"
TRAIN_SCRIPT="run/train_sac_residual.py"
EXPERIMENT_PREFIX="train_sac_residual"
SESSION_PREFIX="sac_rrl_sac_${ENV_SHORT}"
BATCH_LABEL="SAC residual RL robot"

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
  --torch-deterministic
  --no-capture-video
)
EXTRA_ARGS=()

source "${ROOT_DIR}/scripts/lib/launch_batch.sh"
