# Plotting data snapshot

This directory keeps only the CSV files required to regenerate the
current article figures and final metrics tables.

## Active inputs

- `cleared-enriched/drone/{calf,td3,residual_td3,ablation_selective}.csv`
- `cleared-enriched/robot/{calf,td3,residual_td3,ablation,baseline_policy_calls,robot_nominal_eval_summary}.csv`
- `sac/cleared_td3params_preview/{drone,robot}/{sac,sac_residual,sac_calfq}.csv`
- `sac/cleared/robot/sac_calfq.csv`
- `ablation/calfq_td3_drone/{af015,af040,af065,pr10_d09995}.csv`
- `checkpoint_lambda_distance/drone_checkpoint_lambda_distance__summary.csv`

Everything else has been moved to `archive/`.

## Why these files remain active

- `cleared-enriched` is the canonical TD3-family snapshot used by the
  TD3 figures and by the TD3 rows in the final metrics tables.
- `sac/cleared_td3params_preview` is the canonical SAC-family snapshot
  used by the SAC figures and SAC rows in the final metrics tables.
- `sac/cleared/robot/sac_calfq.csv` is kept only for the SAC robot
  schedule-sensitivity figure.
- `ablation/calfq_td3_drone` contains only the schedule/removal-time
  ablation files used by current article figures.
- `checkpoint_lambda_distance` contains only the summarized data used by
  the checkpoint lambda-distance figure.

See `code/plotting/README.md` for the commands that consume these files.
