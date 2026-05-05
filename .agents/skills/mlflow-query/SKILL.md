---
name: mlflow-query
description: Query MLflow tracking server — search experiments/runs, fetch metrics, download artifacts.
---

# MLflow query

You retrieve experiment data from the MLflow tracking server.

Read `CLAUDE.md` first for repo conventions (experiment naming, run semantics).

## Input

The user describes what data they need. Examples:
- "Достань метрики из эксперимента train_calfq__UnderwaterDrone-v0"
- "Покажи все раны с лучшим episode_return"
- "Какие метрики есть в эксперименте X?"
- "Список экспов"
- "Скачай артефакты из последнего рана"

The argument is passed as: $ARGUMENTS

## How experiments and runs are organized

- One MLflow **experiment** = one training script + one environment.
  Naming convention: `{script_name}__{env_id}` (double underscore, e.g., `train_calfq__UnderwaterDrone-v0`).
- Each **run** within an experiment is typically a single seed.
- However, an experiment may contain **many runs beyond just seeds** — this happens when hypotheses are being tested: different hyperparameters, code changes, ablations, etc., all under the same experiment name. Do not assume all runs in an experiment are comparable; always check parameters and tags to understand what each run represents.

## Connection setup

The tracking server is remote. Credentials and host are in `.env` (loaded by `src/config.py`).

To connect, set environment variables before any `mlflow` calls:

```python
import os
from src.config import config

os.environ["AWS_ACCESS_KEY_ID"] = config.AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = config.AWS_SECRET_ACCESS_KEY
os.environ["AWS_DEFAULT_REGION"] = config.AWS_DEFAULT_REGION
os.environ["MLFLOW_S3_ENDPOINT_URL"] = config.MLFLOW_S3_ENDPOINT_URL

import mlflow
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
client = mlflow.MlflowClient()
```

Always run scripts with `uv run` from the repo root so that `src.config` resolves `.env` correctly.

## Common operations

### List experiments

```python
experiments = client.search_experiments()
for exp in experiments:
    print(f"{exp.experiment_id}: {exp.name}")
```

### List runs in an experiment

```python
runs = client.search_runs(
    experiment_ids=["<experiment_id>"],
    order_by=["start_time DESC"],
)
for r in runs:
    print(
        f"run_id={r.info.run_id[:12]}  status={r.info.status}  "
        f"seed={r.data.params.get('seed', '?')}  start={r.info.start_time}"
    )
```

### List available metric names for a run

`run.data.metrics` contains all metric keys (with only the last value). Use it to discover what metrics exist:

```python
run = client.get_run("<run_id>")
metric_names = sorted(run.data.metrics.keys())
for name in metric_names:
    print(name)
```

To list metric names for an entire experiment, pick any finished run:

```python
runs = client.search_runs(
    experiment_ids=["<experiment_id>"],
    filter_string="attributes.status = 'FINISHED'",
    max_results=1,
)
if runs:
    print(sorted(runs[0].data.metrics.keys()))
```

### Search runs with filters

Filter by parameters (values are always strings):

```python
runs = client.search_runs(
    experiment_ids=["<experiment_id>"],
    filter_string='params.seed = "1"',
)
```

Filter by metrics:

```python
runs = client.search_runs(
    experiment_ids=["<experiment_id>"],
    filter_string="metrics.episode_return > 500",
    order_by=["metrics.episode_return DESC"],
)
```

### Get full metric history (timeseries)

`run.data.metrics` only has the last logged value. For the full timeseries:

```python
history = client.get_metric_history(run_id="<run_id>", key="episode_return")
steps = [m.step for m in history]
values = [m.value for m in history]
```

### List and download artifacts

```python
artifacts = client.list_artifacts(run_id="<run_id>")
for a in artifacts:
    print(a.path, a.is_dir)

# Download a specific artifact or directory
local_path = mlflow.artifacts.download_artifacts(
    run_id="<run_id>", artifact_path="<artifact_name>"
)
```

### Get runs as a DataFrame

```python
import mlflow
df = mlflow.search_runs(
    experiment_names=["train_calfq__UnderwaterDrone-v0"],
    order_by=["metrics.episode_return DESC"],
)
# df has columns: run_id, params.*, metrics.*, tags.*, start_time, etc.
```

## Output: saving queried data

All queried data **must** be saved to `run/expdata/`. Use strict, descriptive file names so that the contents are unambiguous from the name alone.

Naming pattern:

```
{experiment_name}__{metric_or_data_type}[__seed{N}].{ext}
```

Examples:
- `train_calfq__UnderwaterDrone-v0__episodic_return__all_seeds.csv`
- `train_calfq__UnderwaterDrone-v0__avoidance_score__seed0.csv`
- `train_td3__RobotNavigationConstSpeedCatch-v0__runs_summary.csv`
- `train_calfq__UnderwaterDrone-v0__metric_names.txt`
- `ablation-q-underwater__episodic_return__all_seeds.csv`

Rules for naming:
- The experiment name goes first, exactly as in MLflow.
- Then the metric name or data type (e.g., `episodic_return`, `runs_summary`, `metric_names`, `artifacts`).
- If data is per-seed, append `__seed{N}`. If aggregated across seeds, append `__all_seeds`.
- Use `.csv` for tabular data, `.txt` for lists, `.json` for structured data.
- **Never** use vague names like `data.csv` or `metrics.csv`.

## Procedure

1. Connect to MLflow using the setup above.
2. Identify the experiment (by name or list all).
3. Search runs with appropriate filters based on what the user asked.
4. Fetch detailed data (metric history, artifacts) as needed.
5. Save results to `run/expdata/` following the naming convention above.
6. Present results clearly — tables for comparisons, timeseries for training curves.

## Rules

- Always use `uv run` to execute any Python code.
- Never hardcode credentials — always read from `src.config`.
- When there are many runs in an experiment, help the user understand the landscape: group by key parameters, show creation dates, flag outliers.
- If the user asks for "best" runs, clarify which metric and direction (max/min) if ambiguous.
- If the query requires advanced MLflow API features beyond what is documented above, consult the Context7 MCP server (`resolve-library-id` → `query-docs` for `mlflow`) to look up current API details before guessing.
