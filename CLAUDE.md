# CLAUDE.md — calfq-td3

## Project overview

Source code for the **CALF-enhance** paper: *"An Agency-Transferring Model-Free Policy Enhancement Technique"*.
The method embeds a functional but suboptimal baseline policy into RL training, progressively transferring agency to the learning policy until it operates fully independently.
The codebase implements the proposed algorithm (CALFQ) and baselines (TD3, TD3-Residual) across several continuous-control Gymnasium environments.

## Repository layout

```
src/                        # Library code (importable as src.*)
  __init__.py               # Gymnasium env registration, path constants
  config.py                 # Pydantic-settings config (reads .env)
  controller.py             # Hand-crafted nominal controllers per environment
  envs/                     # Custom Gymnasium environments
  utils/
    mlflow.py               # mlflow_monitoring decorator — wraps training functions
    metrics_controller.py   # MetricsCollector — batched metric logging
    artifact_uploader.py    # Background artifact upload to MLflow/MinIO
    robot_nav_logging.py    # Trajectory logging for robot-nav envs

run/                        # Executable scripts (not importable)
  train_calfq.py            # CALFQ training (main algorithm)
  train_td3.py              # TD3 baseline
  train_td3_residual.py     # TD3 with residual policy
  lidarnav/                 # LidarNav-specific training scripts
  eval_nominal.py           # Evaluate nominal controllers
  *_rollout.py              # Policy rollout / evaluation
  *_nominal_video.py        # Video generation from nominal controllers
  json_to_video.py          # JSON trajectory → MP4
  train_*_5seeds.sh         # Multi-seed batch training
  *.ipynb                   # Analysis notebooks
  expdata/                  # Final experiment data (CSV, JSON, MP4, PDF)

docker-compose.yaml         # MLflow stack: PostgreSQL + MinIO + MLflow server
.env.example                # Template for credentials (.env is gitignored)
```

## Workspace rules

- `run/expdata/` — final experiment outputs. All results go here.
- `run/logs/`, `run/mlruns/`, `run/videos/` — intermediate artifacts. Gitignored but kept locally — do not delete.
- Credentials live in `.env` (never commit). See `.env.example`.

## Tech stack

- **Python** ≥ 3.13, managed with **uv** — always use `uv run`, never manually activate venvs
- **PyTorch** ≥ 2.6, **Gymnasium** 0.28.1
- **Stable-Baselines3** 2.0.0 — used only for its replay buffer, not for training loops
- **MLflow** ≥ 2.21.3 — experiment tracking via docker-compose stack (Postgres + MinIO + MLflow)
- **tyro** for CLI argument parsing, **loguru** for logging
- **pygame** / **matplotlib** / **moviepy** / **opencv** for rendering and video

## Running experiments

Always run training in a **tmux session** named `<experiment>-seed-<N>` (e.g., `train_calfq-UnderwaterDrone-v0-seed-1`).

```bash
uv run python run/train_calfq.py --env_id UnderwaterDrone-v0 --seed 1   # single run
uv run bash run/train_calfq_5seeds.sh                                    # multi-seed
```

Dependencies: `uv sync` to install, `uv add <pkg>` to add.

## MLflow conventions

One MLflow **experiment** = one training script + one environment (e.g., `train_calfq-UnderwaterDrone-v0`).
Each seed produces a separate **run** within that experiment.

```bash
docker compose up -d    # start tracking infrastructure
```

The stack (Postgres, MinIO, MLflow Server) can run on a remote host — set `EXPERIMENT_TRACKING_HOST` in `.env`.
The `mlflow_monitoring` decorator in `src/utils/mlflow.py` handles run creation, param logging, git auto-commit (disable with `MLFLOW_DISABLE_GIT=1`), and background artifact upload.

## Code conventions

- CLI arguments via `tyro.cli(Args)` with `@dataclass` — follow existing pattern.
- Wrap training functions with `@mlflow_monitoring()`.
- Each environment has a `MetricsCollector` subclass for env-specific metrics.
- Nominal controllers in `src/controller.py` must support both single and batched observations.
- Environments are registered in `src/__init__.py` via `gym.register()`.

## Target environments (paper)

- `UnderwaterDrone-v0` — 6-DOF underwater drone: reach goal, avoid contaminated zones
- `RobotNavigationConstSpeedCatch-v0` — kinematic robot: constant speed, collect object, reach goal

Other environments (LidarNav, RobotNavigation variants, RobotDynamics) exist in the codebase but are not the focus of the paper.
