# Experiment Launch Scripts

This directory contains bash launchers for reproducing the main experiments
and the ablation/sensitivity runs from the paper.

All scripts assume that the Python environment is synchronized with `uv` and
that the MLflow stack from `docker-compose.yaml` is already running. Each run
is started in its own detached `tmux` session and writes a terminal log to
`run/logs/`.

## Common Controls

- `GPUS=0,1`: comma-separated GPU ids used in round-robin order.
- `SEEDS='0 1 2'`: space-separated seeds. If omitted, AUV scripts use seeds
  `0..9`; robot scripts use seeds `1..10`.
- `TOTAL_TIMESTEPS=3000000`: number of environment steps per run.
- `DRY_RUN=1`: print the exact commands without creating `tmux` sessions.

Example:

```bash
DRY_RUN=1 GPUS=0,1 SEEDS='0 1' ./scripts/auv/td3/run_proposed.sh
```

Without `DRY_RUN=1`, all runs in that script are launched immediately, one
`tmux` session per seed. A script never launches multiple algorithms at once:
it always corresponds to one environment, one backbone, and one method.

## Main Experiments

The main launchers are split by environment and backbone:

```text
scripts/
  auv/
    td3/
      run_td3.sh
      run_residual_rl.sh
      run_proposed.sh
    sac/
      run_sac.sh
      run_residual_rl.sh
      run_proposed.sh
  robot/
    td3/
      run_td3.sh
      run_residual_rl.sh
      run_proposed.sh
    sac/
      run_sac.sh
      run_residual_rl.sh
      run_proposed.sh
```

Run one batch at a time, for example:

```bash
GPUS=0,1 ./scripts/auv/td3/run_td3.sh
GPUS=0,1 ./scripts/auv/td3/run_residual_rl.sh
GPUS=0,1 ./scripts/auv/td3/run_proposed.sh

GPUS=0,1 ./scripts/robot/sac/run_sac.sh
GPUS=0,1 ./scripts/robot/sac/run_residual_rl.sh
GPUS=0,1 ./scripts/robot/sac/run_proposed.sh
```

The proposed-method scripts use the published main-run hyperparameters:
`p_0=0.9`, `lambda_0=0.96`, `calfq_anneal_frac=0.9`, and
`calfq_critic_improvement_threshold=0.01`.

## Ablation And Sensitivity

Additional paper runs live under `scripts/ablation_and_sensitivity/`:

```bash
GPUS=0,1 ./scripts/ablation_and_sensitivity/run_calfq_td3_drone_ablation.sh
GPUS=0,1 ./scripts/ablation_and_sensitivity/run_calfq_sac_robot_sensitivity.sh
```

`run_calfq_td3_drone_ablation.sh` reproduces the TD3-backbone AUV schedules
`af015`, `af040`, `af065`, and `pr10_d09995`.

`run_calfq_sac_robot_sensitivity.sh` compares the SAC-backbone robot default
schedule `p_0=0.9`, `lambda_0=0.96` against the higher-trust schedule
`p_0=0.8`, `lambda_0=0.995`.

## Implementation Notes

The main launchers share `scripts/lib/launch_batch.sh`. That helper only
handles common execution mechanics: GPU round-robin assignment, `tmux` session
creation, MLflow experiment/run naming, terminal logs, and `DRY_RUN` output.
The actual experiment hyperparameters are defined in the concrete launcher
that you execute.
