from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots


plt.style.use(["science"])

PLOTTING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PLOTTING_ROOT.parent
DATA_ROOT = PLOTTING_ROOT / "expdata" / "cleared-enriched"
SAC_DATA_ROOT = PLOTTING_ROOT / "expdata" / "sac" / "cleared_td3params_preview"
LEGACY_SAC_DATA_ROOT = PLOTTING_ROOT / "expdata" / "sac" / "cleared"
ABLATION_DATA_ROOT = PLOTTING_ROOT / "expdata" / "ablation" / "calfq_td3_drone"
CHECKPOINT_LAMBDA_DATA_ROOT = PLOTTING_ROOT / "expdata" / "checkpoint_lambda_distance"
OUTPUT_ROOT = REPO_ROOT / "gfx"
TRANSITION_STEP = int(0.9 * 3_000_000)
BASELINE_DISABLED_LABEL = "Baseline disabled"
OURS_TD3_LABEL = r"\textbf{Ours (TD3 backbone)}"
OURS_SAC_LABEL = r"\textbf{Ours (SAC backbone)}"

SCIENCE_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
ABLATION_COLOR = SCIENCE_COLORS[1]
OURS_COLOR = SCIENCE_COLORS[2]


def set_data_root(path: Path) -> None:
    global DATA_ROOT
    DATA_ROOT = path


def article_figure_sources() -> pd.DataFrame:
    rows = [
        ("drone", "calf", DATA_ROOT / "drone" / "calf.csv"),
        ("drone", "td3", DATA_ROOT / "drone" / "td3.csv"),
        ("drone", "residual_td3", DATA_ROOT / "drone" / "residual_td3.csv"),
        ("drone", "ablation_selective", DATA_ROOT / "drone" / "ablation_selective.csv"),
        ("robot", "calf", DATA_ROOT / "robot" / "calf.csv"),
        ("robot", "td3", DATA_ROOT / "robot" / "td3.csv"),
        ("robot", "residual_td3", DATA_ROOT / "robot" / "residual_td3.csv"),
        ("robot", "ablation", DATA_ROOT / "robot" / "ablation.csv"),
        (
            "robot",
            "baseline_policy_calls",
            DATA_ROOT / "robot" / "baseline_policy_calls.csv",
        ),
    ]
    return pd.DataFrame(rows, columns=["environment", "dataset", "path"])


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _dataset(environment: str, name: str) -> pd.DataFrame:
    path = DATA_ROOT / environment / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return _read_csv(path)


def _sac_dataset(environment: str, name: str) -> pd.DataFrame:
    path = SAC_DATA_ROOT / environment / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing SAC dataset: {path}")
    return _read_csv(path)


def _legacy_sac_dataset(environment: str, name: str) -> pd.DataFrame:
    path = LEGACY_SAC_DATA_ROOT / environment / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing legacy SAC dataset: {path}")
    return _read_csv(path)


def _calfq_td3_drone_ablation_dataset(name: str) -> pd.DataFrame:
    path = ABLATION_DATA_ROOT / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing CALF-TD3 drone ablation dataset: {path}")
    return _read_csv(path)


def _checkpoint_lambda_distance_summary() -> pd.DataFrame:
    path = CHECKPOINT_LAMBDA_DATA_ROOT / "drone_checkpoint_lambda_distance__summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint lambda-distance dataset: {path}")
    return _read_csv(path)


def _select_metrics(
    data: pd.DataFrame, metric_name: str, deduplicate: bool = False
) -> pd.DataFrame:
    out = data[data["key"] == metric_name].copy()
    if deduplicate:
        out = out.drop_duplicates(subset=["run_id", "key", "step", "value"])
    return out


def _save(fig: plt.Figure, filename: str, output_dir: Path | None) -> Path:
    out_dir = output_dir or OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def _figure_legend(
    fig: plt.Figure,
    axes,
    *,
    loc: str = "upper center",
    ncol: int | None = None,
    bbox_to_anchor=(0.5, 1.03),
):
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]

    by_label = {}
    for ax in np.ravel(axes):
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label == "_nolegend_" or label.startswith("_"):
                continue
            by_label.setdefault(label, handle)

    labels = list(by_label)
    handles = [by_label[label] for label in labels]
    fig.legend(
        handles,
        labels,
        loc=loc,
        ncol=ncol or len(labels),
        bbox_to_anchor=bbox_to_anchor,
    )


def _rolling_stats(df: pd.DataFrame, rolling_window: int = 100, grid_step: int = 5000):
    df = df.sort_values(["run_id", "step"]).copy()
    df["roll100"] = df.groupby("run_id")["value"].transform(
        lambda series: series.rolling(rolling_window, min_periods=1).mean()
    )

    grid = np.arange(df["step"].min(), df["step"].max() + 1, grid_step)
    grid_df = pd.DataFrame({"step": grid}).sort_values("step")

    def ffill_to_grid(run_id, group):
        group = group.sort_values("step")[["step", "roll100"]]
        out = pd.merge_asof(grid_df, group, on="step", direction="backward")
        out["run_id"] = run_id
        return out

    interp = pd.concat(
        [ffill_to_grid(run_id, group) for run_id, group in df.groupby("run_id")],
        ignore_index=True,
    )

    mean_by_step = interp.groupby("step")["roll100"].mean()
    std_by_step = interp.groupby("step")["roll100"].std()
    return mean_by_step, std_by_step


def _rolling_median_band(
    df: pd.DataFrame,
    rolling_window: int,
    grid_step: int = 1000,
    grid_start: int = 50_000,
    q_low: float = 0.25,
    q_high: float = 0.75,
):
    df = df.sort_values(["run_id", "step"]).copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["step"] >= grid_start]
    if df.empty:
        raise ValueError("No rolling goal-reaching data available for plotting.")
    df["rolling_value"] = df.groupby("run_id")["value"].transform(
        lambda series: series.rolling(rolling_window, min_periods=1).mean()
    )

    grid_begin = max(grid_start, int(df["step"].min()))
    grid = np.arange(grid_begin, df["step"].max() + 1, grid_step)
    grid_df = pd.DataFrame({"step": grid}).sort_values("step")

    def ffill_to_grid(run_id, group):
        group = group.sort_values("step")[["step", "rolling_value"]]
        out = pd.merge_asof(grid_df, group, on="step", direction="backward")
        out["run_id"] = run_id
        return out

    interp = pd.concat(
        [ffill_to_grid(run_id, group) for run_id, group in df.groupby("run_id")],
        ignore_index=True,
    ).dropna(subset=["rolling_value"])

    medians = interp.groupby("step")["rolling_value"].median()
    quantile_low = interp.groupby("step")["rolling_value"].quantile(q_low)
    quantile_high = interp.groupby("step")["rolling_value"].quantile(q_high)
    return medians, quantile_low, quantile_high


def _baseline_policy_band(
    df: pd.DataFrame,
    key: str = "episode_stats/n_safe_actions",
    grid_step: int = 1000,
    grid_start: int = 50_000,
    rolling_window: int = 5,
):
    df = df[df["key"] == key].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values(["run_id", "step"])

    grid_begin = max(grid_start, int(df["step"].min()))
    grid = np.arange(grid_begin, df["step"].max() + 1, grid_step)
    grid_df = pd.DataFrame({"step": grid}).sort_values("step")

    def ffill_to_grid(run_id, group):
        group = group.sort_values("step")[["step", "value"]]
        out = pd.merge_asof(grid_df, group, on="step", direction="backward")
        out["run_id"] = run_id
        return out

    interp = pd.concat(
        [ffill_to_grid(run_id, group) for run_id, group in df.groupby("run_id")],
        ignore_index=True,
    )

    means = (
        interp.groupby("step")["value"]
        .mean()
        .rolling(window=rolling_window, min_periods=1)
        .mean()
    )
    stds = (
        interp.groupby("step")["value"]
        .std()
        .rolling(window=rolling_window, min_periods=1)
        .mean()
    )
    return means * 100, stds * 100


def _metric_band(
    df: pd.DataFrame,
    key: str,
    grid_step: int = 1000,
    grid_start: int = 50_000,
    rolling_window: int = 5,
):
    df = df[df["key"] == key].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values(["run_id", "step"])

    grid_begin = max(grid_start, int(df["step"].min()))
    grid = np.arange(grid_begin, df["step"].max() + 1, grid_step)
    grid_df = pd.DataFrame({"step": grid}).sort_values("step")

    def ffill_to_grid(run_id, group):
        group = group.sort_values("step")[["step", "value"]]
        out = pd.merge_asof(grid_df, group, on="step", direction="backward")
        out["run_id"] = run_id
        return out

    interp = pd.concat(
        [ffill_to_grid(run_id, group) for run_id, group in df.groupby("run_id")],
        ignore_index=True,
    )

    means = (
        interp.groupby("step")["value"]
        .mean()
        .rolling(window=rolling_window, min_periods=1)
        .mean()
    )
    stds = (
        interp.groupby("step")["value"]
        .std()
        .rolling(window=rolling_window, min_periods=1)
        .mean()
    )
    return means, stds


def _learning_policy_band(df: pd.DataFrame):
    baseline_means, baseline_stds = _baseline_policy_band(df)
    return 100 - baseline_means, baseline_stds


def generate_baseline_policy_calls_percent(output_dir: Path | None = None) -> Path:
    drone_baseline_data = _dataset("drone", "calf")
    robot_baseline_data = _dataset("robot", "baseline_policy_calls")

    uw_means, uw_stds = _learning_policy_band(drone_baseline_data)
    robot_means, robot_stds = _learning_policy_band(robot_baseline_data)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.0), sharey=True)
    ax_left, ax_right = axes

    ax_left.plot(uw_means)
    ax_left.fill_between(
        uw_means.index,
        np.clip(uw_means - uw_stds, 0, 100),
        np.clip(uw_means + uw_stds, 0, 100),
        alpha=0.2,
    )
    ax_left.axvline(
        x=TRANSITION_STEP, color="black", linestyle="--", label=BASELINE_DISABLED_LABEL
    )
    ax_left.set_xlim(50_000, 3_000_000)
    ax_left.set_title("Contaminated-Zone AUV")
    ax_left.set_xlabel("Step")
    ax_left.grid()

    ax_right.plot(robot_means)
    ax_right.fill_between(
        robot_means.index,
        np.clip(robot_means - robot_stds, 0, 100),
        np.clip(robot_means + robot_stds, 0, 100),
        alpha=0.2,
    )
    ax_right.axvline(x=TRANSITION_STEP, color="black", linestyle="--", label="_nolegend_")
    ax_right.set_xlim(50_000, 3_000_000)
    ax_right.set_title("Treasure-Collecting Robot")
    ax_right.set_xlabel("Step")
    ax_right.grid()

    fig.supylabel(r"Fraction of learning policy\\calls to total episode steps (in \%)")
    _figure_legend(fig, axes, ncol=1, bbox_to_anchor=(0.5, 1.03))
    try:
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    except RuntimeError:
        fig.subplots_adjust(top=0.82, bottom=0.18, wspace=0.25)
    return _save(fig, "baseline_policy_calls_percent.pdf", output_dir)


def generate_schedule_parameters(output_dir: Path | None = None) -> Path:
    drone_data = _dataset("drone", "calf")

    relprob_mean, relprob_std = _metric_band(drone_data, "calfq/p_relax")
    lambda_mean, lambda_std = _metric_band(drone_data, "calfq/p_relax_decay")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6, 3.0))
    ax_left.plot(
        relprob_mean.index,
        relprob_mean,
        label=r"$p^{\mathrm{rel}}$",
        color=OURS_COLOR,
    )
    ax_left.fill_between(
        relprob_mean.index,
        relprob_mean - relprob_std,
        relprob_mean + relprob_std,
        alpha=0.2,
        color=OURS_COLOR,
    )
    ax_left.axvline(
        x=TRANSITION_STEP, color="black", linestyle="--", label=BASELINE_DISABLED_LABEL
    )
    ax_left.set_title(r"$p^{\mathrm{rel}}$")
    ax_left.set_xlim(50_000, 3_000_000)
    ax_left.set_ylim(0.79, 1.01)
    ax_left.set_xlabel("Step")
    ax_left.set_ylabel(r"$p^{\mathrm{rel}}$")
    ax_left.grid()

    ax_right.plot(
        lambda_mean.index,
        lambda_mean,
        label=r"$\lambda$",
        color=ABLATION_COLOR,
    )
    ax_right.fill_between(
        lambda_mean.index,
        lambda_mean - lambda_std,
        lambda_mean + lambda_std,
        alpha=0.2,
        color=ABLATION_COLOR,
    )
    ax_right.axvline(
        x=TRANSITION_STEP, color="black", linestyle="--", label="_nolegend_"
    )
    ax_right.set_title(r"$\lambda$")
    ax_right.set_xlim(50_000, 3_000_000)
    ax_right.set_ylim(0.994, 1.001)
    ax_right.set_xlabel("Step")
    ax_right.set_ylabel(r"$\lambda$")
    ax_right.grid()

    _figure_legend(fig, (ax_left, ax_right), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return _save(fig, "schedule_parameters.pdf", output_dir)


def generate_episode_return_comparison(output_dir: Path | None = None) -> Path:
    calfq_data = _dataset("drone", "calf")
    td3_data = _dataset("drone", "td3")
    td3_residual_data = _dataset("drone", "residual_td3")

    td3_means = (
        _select_metrics(td3_data, "charts/episodic_return_rolling_20")
        .groupby("step")["value"]
        .mean()
    )
    td3_stds = (
        _select_metrics(td3_data, "charts/episodic_return_rolling_20")
        .groupby("step")["value"]
        .std()
    )
    residual_means = (
        _select_metrics(td3_residual_data, "charts/episodic_return_rolling_20")
        .groupby("step")["value"]
        .mean()
    )
    residual_stds = (
        _select_metrics(td3_residual_data, "charts/episodic_return_rolling_20")
        .groupby("step")["value"]
        .std()
    )
    calfq_means = (
        _select_metrics(calfq_data, "charts/episodic_return_rolling_20")
        .groupby("step")["value"]
        .mean()
    )
    calfq_stds = (
        _select_metrics(calfq_data, "charts/episodic_return_rolling_20")
        .groupby("step")["value"]
        .std()
    )

    robot_calfq_data = _dataset("robot", "calf")
    robot_residual_data = _dataset("robot", "residual_td3")
    robot_td3_data = _dataset("robot", "td3")

    episodic_return_calfq = _select_metrics(robot_calfq_data, "charts/episodic_return")
    episodic_return_td3 = _select_metrics(
        robot_td3_data, "charts/episodic_return", deduplicate=True
    )
    episodic_return_residual_td3 = _select_metrics(
        robot_residual_data, "charts/episodic_return"
    )

    robot_td3_mean, robot_td3_std = _rolling_stats(
        episodic_return_td3.copy(), rolling_window=20, grid_step=1000
    )
    robot_rrl_mean, robot_rrl_std = _rolling_stats(
        episodic_return_residual_td3.copy(), rolling_window=20, grid_step=1000
    )
    robot_calf_mean, robot_calf_std = _rolling_stats(
        episodic_return_calfq.copy(), rolling_window=20, grid_step=1000
    )

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.0))
    ax_left, ax_right = axes

    ax_left.plot(td3_means.index, td3_means, label="TD3")
    ax_left.fill_between(
        td3_means.index, td3_means - td3_stds, td3_means + td3_stds, alpha=0.2
    )
    ax_left.plot(residual_means.index, residual_means, label="Residual TD3")
    ax_left.fill_between(
        residual_means.index,
        residual_means - residual_stds,
        residual_means + residual_stds,
        alpha=0.2,
    )
    ax_left.plot(calfq_means.index, calfq_means, label=OURS_TD3_LABEL)
    ax_left.fill_between(
        calfq_means.index,
        calfq_means - calfq_stds,
        calfq_means + calfq_stds,
        alpha=0.2,
    )
    ax_left.axvline(
        x=TRANSITION_STEP, color="black", linestyle="--", label=BASELINE_DISABLED_LABEL
    )
    ax_left.set_xlim(45_000, 3_000_000)
    ax_left.set_ylim(-4000, 0)
    ax_left.set_title("Contaminated-Zone AUV")
    ax_left.set_xlabel("Step")
    ax_left.grid()

    ax_right.plot(robot_td3_mean.index, robot_td3_mean, label="TD3")
    ax_right.fill_between(
        robot_td3_mean.index,
        robot_td3_mean - robot_td3_std,
        robot_td3_mean + robot_td3_std,
        alpha=0.2,
    )
    ax_right.plot(robot_rrl_mean.index, robot_rrl_mean, label="Residual TD3")
    ax_right.fill_between(
        robot_rrl_mean.index,
        robot_rrl_mean - robot_rrl_std,
        robot_rrl_mean + robot_rrl_std,
        alpha=0.2,
    )
    ax_right.plot(robot_calf_mean.index, robot_calf_mean, label=OURS_TD3_LABEL)
    ax_right.fill_between(
        robot_calf_mean.index,
        robot_calf_mean - robot_calf_std,
        robot_calf_mean + robot_calf_std,
        alpha=0.2,
    )
    ax_right.axvline(x=TRANSITION_STEP, color="black", linestyle="--", label="_nolegend_")
    ax_right.set_xlim(50_000, 3_000_000)
    ax_right.set_ylim(-60, 50)
    ax_right.set_title("Treasure-Collecting Robot")
    ax_right.set_xlabel("Step")
    ax_right.grid()

    fig.supylabel("Rolling 20-Episode Return")
    _figure_legend(fig, axes, ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "episode_return_comparison.pdf", output_dir)


def generate_goal_reaching_rate_comparison(output_dir: Path | None = None) -> Path:
    q_low = 0.25
    q_high = 0.75
    rolling_window = 75

    calf_goal_reaching = _dataset("drone", "calf")
    calf_goal_reaching = _select_metrics(
        calf_goal_reaching, "episode_stats/is_in_hole", deduplicate=True
    )[["run_id", "value", "step"]]

    td3_goal_reaching = _dataset("drone", "td3")
    td3_goal_reaching = _select_metrics(
        td3_goal_reaching, "episode_stats/is_in_hole", deduplicate=True
    )[["run_id", "value", "step"]]

    rrl_goal_reaching = _dataset("drone", "residual_td3")
    rrl_goal_reaching = _select_metrics(
        rrl_goal_reaching, "episode_stats/is_in_hole", deduplicate=True
    )[["run_id", "value", "step"]]

    uw_td3_median, uw_td3_low, uw_td3_high = _rolling_median_band(
        td3_goal_reaching.copy(),
        rolling_window=rolling_window,
        grid_step=1000,
        grid_start=50_000,
        q_low=q_low,
        q_high=q_high,
    )
    uw_rrl_median, uw_rrl_low, uw_rrl_high = _rolling_median_band(
        rrl_goal_reaching.copy(),
        rolling_window=rolling_window,
        grid_step=1000,
        grid_start=50_000,
        q_low=q_low,
        q_high=q_high,
    )
    uw_calf_median, uw_calf_low, uw_calf_high = _rolling_median_band(
        calf_goal_reaching.copy(),
        rolling_window=rolling_window,
        grid_step=1000,
        grid_start=50_000,
        q_low=q_low,
        q_high=q_high,
    )

    goal_reaching_calfq = _select_metrics(
        _dataset("robot", "calf"), "episode_stats/goal_reached"
    )
    goal_reaching_td3 = _select_metrics(
        _dataset("robot", "td3"), "episode_stats/goal_reached"
    )
    goal_reaching_residual_td3 = _select_metrics(
        _dataset("robot", "residual_td3"), "episode_stats/goal_reached"
    )

    robot_td3_median, robot_td3_low, robot_td3_high = _rolling_median_band(
        goal_reaching_td3.copy(),
        rolling_window=rolling_window,
        grid_step=1000,
        grid_start=50_000,
        q_low=q_low,
        q_high=q_high,
    )
    robot_rrl_median, robot_rrl_low, robot_rrl_high = _rolling_median_band(
        goal_reaching_residual_td3.copy(),
        rolling_window=rolling_window,
        grid_step=1000,
        grid_start=50_000,
        q_low=q_low,
        q_high=q_high,
    )
    robot_calf_median, robot_calf_low, robot_calf_high = _rolling_median_band(
        goal_reaching_calfq.copy(),
        rolling_window=rolling_window,
        grid_step=1000,
        grid_start=50_000,
        q_low=q_low,
        q_high=q_high,
    )

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.0), sharey=True)
    ax_left, ax_right = axes

    ax_left.plot(uw_td3_median.index, uw_td3_median * 100, label="TD3")
    ax_left.fill_between(
        uw_td3_median.index,
        uw_td3_low * 100,
        uw_td3_high * 100,
        alpha=0.2,
    )
    ax_left.plot(uw_rrl_median.index, uw_rrl_median * 100, label="Residual TD3")
    ax_left.fill_between(
        uw_rrl_median.index,
        uw_rrl_low * 100,
        uw_rrl_high * 100,
        alpha=0.2,
    )
    ax_left.plot(uw_calf_median.index, uw_calf_median * 100, label=OURS_TD3_LABEL)
    ax_left.fill_between(
        uw_calf_median.index,
        uw_calf_low * 100,
        uw_calf_high * 100,
        alpha=0.2,
    )
    ax_left.axvline(
        x=TRANSITION_STEP, color="black", linestyle="--", label=BASELINE_DISABLED_LABEL
    )
    ax_left.set_xlim(100000, 3_000_000)
    ax_left.set_ylim(0, 110)
    ax_left.set_title("Contaminated-Zone AUV")
    ax_left.set_xlabel("Step")
    ax_left.grid()

    ax_right.plot(robot_td3_median.index, robot_td3_median * 100, label="TD3")
    ax_right.fill_between(
        robot_td3_median.index,
        robot_td3_low * 100,
        robot_td3_high * 100,
        alpha=0.2,
    )
    ax_right.plot(robot_rrl_median.index, robot_rrl_median * 100, label="Residual TD3")
    ax_right.fill_between(
        robot_rrl_median.index,
        robot_rrl_low * 100,
        robot_rrl_high * 100,
        alpha=0.2,
    )
    ax_right.plot(robot_calf_median.index, robot_calf_median * 100, label=OURS_TD3_LABEL)
    ax_right.fill_between(
        robot_calf_median.index,
        robot_calf_low * 100,
        robot_calf_high * 100,
        alpha=0.2,
    )
    ax_right.axvline(x=TRANSITION_STEP, color="black", linestyle="--", label="_nolegend_")
    ax_right.set_xlim(50_000, 3_000_000)
    ax_right.set_ylim(0, 110)
    ax_right.set_title("Treasure-Collecting Robot")
    ax_right.set_xlabel("Step")
    ax_right.grid()

    fig.supylabel(
        rf"Rolling {rolling_window}-episode"
        + "\n"
        + r"goal-reaching rate (\%)"
        + "\n"
        + "Median with interquantile range",
        multialignment="center",
    )
    _figure_legend(fig, axes, ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "goal_reaching_rate_comparison.pdf", output_dir)


def generate_ablation_baseline_calls_robot(output_dir: Path | None = None) -> Path:
    ablation_data = _dataset("robot", "ablation")
    base_baseline_calls = _dataset("robot", "baseline_policy_calls")

    base_calls_mean, base_calls_std = _learning_policy_band(base_baseline_calls)
    abl_calls_mean, abl_calls_std = _learning_policy_band(ablation_data)

    fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.0))

    ax.plot(
        abl_calls_mean.index,
        abl_calls_mean,
        label=rf"{OURS_TD3_LABEL} ($\nu = \infty$)",
        color=ABLATION_COLOR,
    )
    ax.fill_between(
        abl_calls_mean.index,
        np.clip(abl_calls_mean - abl_calls_std, 0, 100),
        np.clip(abl_calls_mean + abl_calls_std, 0, 100),
        alpha=0.2,
        color=ABLATION_COLOR,
    )
    ax.plot(
        base_calls_mean.index,
        base_calls_mean,
        label=rf"{OURS_TD3_LABEL} (default, $\nu = 0.01$)",
        color=OURS_COLOR,
    )
    ax.fill_between(
        base_calls_mean.index,
        np.clip(base_calls_mean - base_calls_std, 0, 100),
        np.clip(base_calls_mean + base_calls_std, 0, 100),
        alpha=0.2,
        color=OURS_COLOR,
    )
    ax.axvline(
        x=TRANSITION_STEP, color="black", linestyle="--", label=BASELINE_DISABLED_LABEL
    )
    ax.set_xlim(50_000, 3_000_000)
    ax.set_title("Treasure-Collecting Robot")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"Fraction of learning policy \\calls to total episode steps (in \%)")
    ax.grid()

    _figure_legend(fig, ax, ncol=1, bbox_to_anchor=(0.5, 1.17))
    fig.tight_layout(rect=[0, 0, 1, 0.76])
    return _save(fig, "ablation_baseline_calls_robot.pdf", output_dir)


def generate_ablation_episode_return_drone_robot(output_dir: Path | None = None) -> Path:
    robot_ablation_data = _dataset("robot", "ablation")
    robot_base_data = _dataset("robot", "calf")
    drone_ablation_data = _dataset("drone", "ablation_selective")
    drone_base_data = _dataset("drone", "calf")

    def episodic_return_band(df, rolling_window=20, grid_step=1000):
        episodic = _select_metrics(df, "charts/episodic_return")
        return _rolling_stats(episodic.copy(), rolling_window, grid_step)

    def draw_ablation_return(
        ax, title, base_mean, base_std, abl_mean, abl_std, add_labels=False
    ):
        abl_label = rf"{OURS_TD3_LABEL} ($\nu = \infty$)" if add_labels else "_nolegend_"
        ours_label = (
            rf"{OURS_TD3_LABEL} (default, $\nu = 0.01$)"
            if add_labels
            else "_nolegend_"
        )
        baseline_label = BASELINE_DISABLED_LABEL if add_labels else "_nolegend_"

        ax.plot(abl_mean.index, abl_mean, label=abl_label, color=ABLATION_COLOR)
        ax.fill_between(
            abl_mean.index,
            abl_mean - abl_std,
            abl_mean + abl_std,
            alpha=0.2,
            color=ABLATION_COLOR,
        )

        ax.plot(base_mean.index, base_mean, label=ours_label, color=OURS_COLOR)
        ax.fill_between(
            base_mean.index,
            base_mean - base_std,
            base_mean + base_std,
            alpha=0.2,
            color=OURS_COLOR,
        )

        ax.axvline(x=TRANSITION_STEP, color="black", linestyle="--", label=baseline_label)
        ax.set_xlim(50_000, 3_000_000)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid()

    drone_base_mean, drone_base_std = episodic_return_band(drone_base_data, 40, 1000)
    drone_abl_mean, drone_abl_std = episodic_return_band(drone_ablation_data, 40, 1000)
    robot_base_mean, robot_base_std = episodic_return_band(robot_base_data, 40, 1000)
    robot_abl_mean, robot_abl_std = episodic_return_band(robot_ablation_data, 40, 1000)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6, 3.0))

    draw_ablation_return(
        ax_left,
        "Contaminated-Zone AUV",
        drone_base_mean,
        drone_base_std,
        drone_abl_mean,
        drone_abl_std,
        add_labels=True,
    )
    draw_ablation_return(
        ax_right,
        "Treasure-Collecting Robot",
        robot_base_mean,
        robot_base_std,
        robot_abl_mean,
        robot_abl_std,
    )

    ax_left.set_xlim(45_000, 3_000_000)
    ax_left.set_ylim(-3000, 0)
    ax_right.set_xlim(50_000, 3_000_000)
    ax_right.set_ylim(-60, 50)
    fig.supylabel("Rolling 40-Episode\nReturn", multialignment="center")

    _figure_legend(fig, (ax_left, ax_right), ncol=1, bbox_to_anchor=(0.5, 1.17))
    fig.tight_layout(rect=[0, 0, 1, 0.76])
    return _save(fig, "ablation_episode_return_drone_robot.pdf", output_dir)


def generate_calf_td3_drone_ablation_returns(output_dir: Path | None = None) -> Path:
    td3_relax_configs = [
        (
            "pr10_d09995",
            rf"{OURS_TD3_LABEL} ($p_0=1.0,\lambda_0=0.9995$)",
            SCIENCE_COLORS[3],
        ),
        (
            "default",
            rf"{OURS_TD3_LABEL} (default, $p_0 = 0.9$, $\lambda_0 = 0.96$)",
            OURS_COLOR,
        ),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.75))
    for dataset_name, label, color in td3_relax_configs:
        data = (
            _dataset("drone", "calf")
            if dataset_name == "default"
            else _calfq_td3_drone_ablation_dataset(dataset_name)
        )
        mean, std = _metric_band(
            data,
            "charts/episodic_return",
            grid_step=1000,
            grid_start=50_000,
            rolling_window=40,
        )
        ax.plot(mean.index, mean, label=label, color=color)
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2, color=color)
    ax.axvline(
        x=TRANSITION_STEP,
        color="black",
        linestyle="--",
        label=BASELINE_DISABLED_LABEL,
    )
    ax.set_xlim(45_000, 3_000_000)
    ax.set_ylim(-4000, 0)
    ax.set_title("Ours (TD3 backbone)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Rolling 40-Episode\nReturn", multialignment="center")
    ax.grid()

    _figure_legend(fig, ax, ncol=1, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    return _save(fig, "calf_td3_drone_ablation_returns.pdf", output_dir)


def generate_calf_sac_robot_hyperparameter_sensitivity(
    output_dir: Path | None = None,
) -> Path:
    q_low = 0.25
    q_high = 0.75
    rolling_window = 75

    sac_datasets = [
        (
            rf"{OURS_SAC_LABEL} (default, $p_0 = 0.9$, $\lambda_0 = 0.96$)",
            lambda env: _sac_dataset(env, "sac_calfq"),
            OURS_COLOR,
            "-",
        ),
        (
            rf"{OURS_SAC_LABEL} ($p_0=0.8,\lambda_0=0.995$)",
            lambda env: _legacy_sac_dataset(env, "sac_calfq"),
            ABLATION_COLOR,
            "--",
        ),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.75))
    for label, loader, color, linestyle in sac_datasets:
        data = loader("robot")
        metric = _select_metrics(data, "episode_stats/goal_reached", deduplicate=True)[
            ["run_id", "value", "step"]
        ]
        median, low, high = _rolling_median_band(
            metric.copy(),
            rolling_window=rolling_window,
            grid_step=1000,
            grid_start=50_000,
            q_low=q_low,
            q_high=q_high,
        )
        ax.plot(median.index, median * 100, label=label, color=color, linestyle=linestyle)
        ax.fill_between(median.index, low * 100, high * 100, alpha=0.18, color=color)
    ax.axvline(
        x=TRANSITION_STEP,
        color="black",
        linestyle="--",
        label=BASELINE_DISABLED_LABEL,
    )
    ax.set_xlim(50_000, 3_000_000)
    ax.set_ylim(0, 110)
    ax.set_title("Ours (SAC backbone)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Rolling 75-episode rate (\\%)")
    ax.grid()

    _figure_legend(fig, ax, ncol=1, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    return _save(fig, "calf_sac_robot_hyperparameter_sensitivity.pdf", output_dir)


def generate_calf_td3_drone_ablation_anneal_is_in_hole(
    output_dir: Path | None = None,
) -> Path:
    q_low = 0.25
    q_high = 0.75
    rolling_window = 75
    configs = [
        ("af015", r"$T_{\mathrm{tran}}=0.45$M", 450_000, SCIENCE_COLORS[0]),
        ("af040", r"$T_{\mathrm{tran}}=1.20$M", 1_200_000, SCIENCE_COLORS[1]),
        ("af065", r"$T_{\mathrm{tran}}=1.95$M", 1_950_000, SCIENCE_COLORS[3]),
        ("default", r"default ($T_{\mathrm{tran}}=2.70$M)", 2_700_000, OURS_COLOR),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(3.1, 2.8))
    for dataset_name, label, transition_step, color in configs:
        data = (
            _dataset("drone", "calf")
            if dataset_name == "default"
            else _calfq_td3_drone_ablation_dataset(dataset_name)
        )
        metric = _select_metrics(
            data,
            "episode_stats/is_in_hole",
            deduplicate=True,
        )[["run_id", "value", "step"]]
        median, low, high = _rolling_median_band(
            metric.copy(),
            rolling_window=rolling_window,
            grid_step=1000,
            grid_start=50_000,
            q_low=q_low,
            q_high=q_high,
        )
        ax.plot(median.index, median * 100, label=label, color=color)
        ax.fill_between(median.index, low * 100, high * 100, alpha=0.2, color=color)
        ax.axvline(x=transition_step, color=color, linestyle="--", alpha=0.75)
    ax.set_xlim(100_000, 3_000_000)
    ax.set_ylim(0, 110)
    ax.set_title("Baseline-removal time")
    ax.set_xlabel("Step")
    ax.set_ylabel(
        rf"Rolling {rolling_window}-episode" + "\n" + r"goal-reaching rate (\%)",
        multialignment="center",
    )
    ax.grid()
    _figure_legend(fig, ax, ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return _save(fig, "calf_td3_drone_ablation_anneal_is_in_hole.pdf", output_dir)


def generate_drone_checkpoint_lambda_distance(output_dir: Path | None = None) -> Path:
    data = _checkpoint_lambda_distance_summary().sort_values("lambda_decay")

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.75))
    ax.plot(
        data["lambda_decay"],
        data["mean"],
        marker="o",
        linewidth=2,
        color=OURS_COLOR,
        label=r"$p^{\mathrm{rel}}=1$",
    )
    ax.fill_between(
        data["lambda_decay"],
        data["mean"] - data["stderr"],
        data["mean"] + data["stderr"],
        alpha=0.2,
        color=OURS_COLOR,
    )
    ax.set_title("Contaminated-Zone AUV Navigation")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"MC estimate of an upper bound on $W_T$")
    ax.set_xlim(0.958, 1.002)
    ax.set_ylim(bottom=0)
    ax.grid()
    ax.legend(loc="upper right")

    fig.tight_layout()
    return _save(fig, "drone_checkpoint_lambda_distance.pdf", output_dir)


def _episode_return_band(df: pd.DataFrame, rolling_window: int = 20):
    episodic = _select_metrics(df, "charts/episodic_return")
    return _rolling_stats(episodic.copy(), rolling_window=rolling_window, grid_step=1000)


def generate_sac_episode_return_comparison(output_dir: Path | None = None) -> Path:
    datasets = {
        "drone": {
            "SAC": _sac_dataset("drone", "sac"),
            "Residual SAC": _sac_dataset("drone", "sac_residual"),
            OURS_SAC_LABEL: _sac_dataset("drone", "sac_calfq"),
        },
        "robot": {
            "SAC": _sac_dataset("robot", "sac"),
            "Residual SAC": _sac_dataset("robot", "sac_residual"),
            OURS_SAC_LABEL: _sac_dataset("robot", "sac_calfq"),
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.0))
    for ax, env, title in [
        (axes[0], "drone", "Contaminated-Zone AUV"),
        (axes[1], "robot", "Treasure-Collecting Robot"),
    ]:
        for label, data in datasets[env].items():
            mean, std = _episode_return_band(data)
            ax.plot(mean.index, mean, label=label)
            ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
        ax.axvline(
            x=TRANSITION_STEP,
            color="black",
            linestyle="--",
            label=BASELINE_DISABLED_LABEL if env == "drone" else "_nolegend_",
        )
        ax.set_xlim(50_000, 3_000_000)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid()

    axes[0].set_ylim(-4000, 0)
    axes[1].set_ylim(-60, 50)
    fig.supylabel("Rolling 20-Episode Return")
    _figure_legend(fig, axes, ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "sac_episode_return_comparison.pdf", output_dir)


def generate_sac_goal_reaching_rate_comparison(output_dir: Path | None = None) -> Path:
    q_low = 0.25
    q_high = 0.75
    rolling_window = 75
    configs = [
        (
            "drone",
            "episode_stats/is_in_hole",
            "Contaminated-Zone AUV",
            {
                "SAC": _sac_dataset("drone", "sac"),
                "Residual SAC": _sac_dataset("drone", "sac_residual"),
                OURS_SAC_LABEL: _sac_dataset("drone", "sac_calfq"),
            },
        ),
        (
            "robot",
            "episode_stats/goal_reached",
            "Treasure-Collecting Robot",
            {
                "SAC": _sac_dataset("robot", "sac"),
                "Residual SAC": _sac_dataset("robot", "sac_residual"),
                OURS_SAC_LABEL: _sac_dataset("robot", "sac_calfq"),
            },
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.0), sharey=True)
    for ax, (env, key, title, datasets) in zip(axes, configs):
        for label, data in datasets.items():
            metric = _select_metrics(data, key, deduplicate=True)[["run_id", "value", "step"]]
            median, low, high = _rolling_median_band(
                metric.copy(),
                rolling_window=rolling_window,
                grid_step=1000,
                grid_start=50_000,
                q_low=q_low,
                q_high=q_high,
            )
            ax.plot(median.index, median * 100, label=label)
            ax.fill_between(median.index, low * 100, high * 100, alpha=0.2)
        ax.axvline(
            x=TRANSITION_STEP,
            color="black",
            linestyle="--",
            label=BASELINE_DISABLED_LABEL if env == "drone" else "_nolegend_",
        )
        ax.set_xlim(50_000, 3_000_000)
        ax.set_ylim(0, 110)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid()

    fig.supylabel(
        rf"Rolling {rolling_window}-episode"
        + "\n"
        + r"goal-reaching rate (\%)"
        + "\n"
        + "Median with interquantile range",
        multialignment="center",
    )
    _figure_legend(fig, axes, ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "sac_goal_reaching_rate_comparison.pdf", output_dir)


def generate_sac_calf_hyperparameter_ablation_goal_reaching_rate(
    output_dir: Path | None = None,
) -> Path:
    q_low = 0.25
    q_high = 0.75
    rolling_window = 75
    configs = [
        ("drone", "episode_stats/is_in_hole", "Contaminated-Zone AUV"),
        ("robot", "episode_stats/goal_reached", "Treasure-Collecting Robot"),
    ]
    datasets = [
        (
            rf"{OURS_SAC_LABEL} (TD3-matched)",
            lambda env: _sac_dataset(env, "sac_calfq"),
            OURS_COLOR,
            "-",
        ),
        (
            f"{OURS_SAC_LABEL} (original)",
            lambda env: _legacy_sac_dataset(env, "sac_calfq"),
            ABLATION_COLOR,
            "--",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.0), sharey=True)
    for ax, (env, key, title) in zip(axes, configs):
        for label, loader, color, linestyle in datasets:
            data = loader(env)
            metric = _select_metrics(data, key, deduplicate=True)[["run_id", "value", "step"]]
            median, low, high = _rolling_median_band(
                metric.copy(),
                rolling_window=rolling_window,
                grid_step=1000,
                grid_start=50_000,
                q_low=q_low,
                q_high=q_high,
            )
            ax.plot(median.index, median * 100, label=label, color=color, linestyle=linestyle)
            ax.fill_between(median.index, low * 100, high * 100, alpha=0.18, color=color)
        ax.axvline(
            x=TRANSITION_STEP,
            color="black",
            linestyle="--",
            label=BASELINE_DISABLED_LABEL if env == "drone" else "_nolegend_",
        )
        ax.set_xlim(50_000, 3_000_000)
        ax.set_ylim(0, 110)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid()

    fig.supylabel(
        rf"Rolling {rolling_window}-episode"
        + "\n"
        + r"goal-reaching rate (\%)"
        + "\n"
        + "Median with interquantile range",
        multialignment="center",
    )
    _figure_legend(fig, axes, ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "sac_calf_hyperparameter_ablation_goal_reaching_rate.pdf", output_dir)


def _plot_return_band(
    ax,
    df: pd.DataFrame,
    label: str,
    color: str,
    linestyle: str = "-",
    rolling_window: int = 20,
):
    mean, std = _episode_return_band(df, rolling_window=rolling_window)
    ax.plot(mean.index, mean, label=label, color=color, linestyle=linestyle)
    ax.fill_between(mean.index, mean - std, mean + std, alpha=0.14, color=color)


def _plot_goal_band(
    ax,
    df: pd.DataFrame,
    key: str,
    label: str,
    color: str,
    linestyle: str = "-",
    rolling_window: int = 75,
):
    metric = _select_metrics(df, key, deduplicate=True)[["run_id", "value", "step"]]
    median, low, high = _rolling_median_band(
        metric.copy(),
        rolling_window=rolling_window,
        grid_step=1000,
        grid_start=50_000,
        q_low=0.25,
        q_high=0.75,
    )
    ax.plot(median.index, median * 100, label=label, color=color, linestyle=linestyle)
    ax.fill_between(median.index, low * 100, high * 100, alpha=0.14, color=color)


def generate_backbone_comparison_drone(output_dir: Path | None = None) -> Path:
    configs = [
        ("TD3", _dataset("drone", "td3"), SCIENCE_COLORS[0], "-"),
        ("Residual TD3", _dataset("drone", "residual_td3"), SCIENCE_COLORS[1], "-"),
        (OURS_TD3_LABEL, _dataset("drone", "calf"), SCIENCE_COLORS[2], "-"),
        ("SAC", _sac_dataset("drone", "sac"), SCIENCE_COLORS[0], "--"),
        ("Residual SAC", _sac_dataset("drone", "sac_residual"), SCIENCE_COLORS[1], "--"),
        (OURS_SAC_LABEL, _sac_dataset("drone", "sac_calfq"), SCIENCE_COLORS[2], "--"),
    ]

    fig, (ax_return, ax_goal) = plt.subplots(1, 2, figsize=(6.8, 3.0))
    for label, data, color, linestyle in configs:
        _plot_return_band(ax_return, data, label, color, linestyle)
        _plot_goal_band(
            ax_goal,
            data,
            "episode_stats/is_in_hole",
            label,
            color,
            linestyle,
        )

    for ax in (ax_return, ax_goal):
        ax.axvline(
            x=TRANSITION_STEP,
            color="black",
            linestyle="--",
            label=BASELINE_DISABLED_LABEL if ax is ax_return else "_nolegend_",
        )
        ax.set_xlim(50_000, 3_000_000)
        ax.set_xlabel("Step")
        ax.grid()

    ax_return.set_title("Episode return")
    ax_return.set_ylabel("Rolling 20-Episode Return")
    ax_return.set_ylim(-4000, 0)
    ax_goal.set_title("Goal-reaching rate")
    ax_goal.set_ylabel(
        "Rolling 75-episode\ngoal-reaching rate (\\%)",
        multialignment="center",
    )
    ax_goal.set_ylim(0, 110)

    _figure_legend(fig, (ax_return, ax_goal), ncol=4, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return _save(fig, "backbone_comparison_drone.pdf", output_dir)


def generate_backbone_comparison_robot(output_dir: Path | None = None) -> Path:
    configs = [
        ("TD3", _dataset("robot", "td3"), SCIENCE_COLORS[0], "-"),
        ("Residual TD3", _dataset("robot", "residual_td3"), SCIENCE_COLORS[1], "-"),
        (OURS_TD3_LABEL, _dataset("robot", "calf"), SCIENCE_COLORS[2], "-"),
        ("SAC", _sac_dataset("robot", "sac"), SCIENCE_COLORS[0], "--"),
        ("Residual SAC", _sac_dataset("robot", "sac_residual"), SCIENCE_COLORS[1], "--"),
        (OURS_SAC_LABEL, _sac_dataset("robot", "sac_calfq"), SCIENCE_COLORS[2], "--"),
    ]

    fig, (ax_return, ax_goal) = plt.subplots(1, 2, figsize=(6.8, 3.0))
    for label, data, color, linestyle in configs:
        _plot_return_band(ax_return, data, label, color, linestyle)
        _plot_goal_band(
            ax_goal,
            data,
            "episode_stats/goal_reached",
            label,
            color,
            linestyle,
        )

    for ax in (ax_return, ax_goal):
        ax.axvline(
            x=TRANSITION_STEP,
            color="black",
            linestyle="--",
            label=BASELINE_DISABLED_LABEL if ax is ax_return else "_nolegend_",
        )
        ax.set_xlim(50_000, 3_000_000)
        ax.set_xlabel("Step")
        ax.grid()

    ax_return.set_title("Episode return")
    ax_return.set_ylabel("Rolling 20-Episode Return")
    ax_return.set_ylim(-60, 50)
    ax_goal.set_title("Goal-reaching rate")
    ax_goal.set_ylabel(
        "Rolling 75-episode\ngoal-reaching rate (\\%)",
        multialignment="center",
    )
    ax_goal.set_ylim(0, 110)

    _figure_legend(fig, (ax_return, ax_goal), ncol=4, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return _save(fig, "backbone_comparison_robot.pdf", output_dir)


def _final_window_seed_values(df: pd.DataFrame, key: str) -> pd.Series:
    metric = _select_metrics(df, key, deduplicate=True).copy()
    metric["value"] = pd.to_numeric(metric["value"], errors="coerce")
    metric = metric.dropna(subset=["value"])
    metric = metric[metric["step"].between(TRANSITION_STEP, 3_000_000)]
    return metric.groupby("run_id")["value"].mean()


def _mean_pm(values: pd.Series, multiplier: float = 1.0, decimals: int = 2) -> str:
    values = values.dropna() * multiplier
    mean = values.mean()
    std = values.std()
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def _sac_final_metric_rows(environment: str) -> list[tuple[str, ...]]:
    methods = [
        (OURS_SAC_LABEL, "sac_calfq"),
        ("Residual SAC", "sac_residual"),
        ("SAC", "sac"),
    ]

    rows = []
    for label, dataset_name in methods:
        data = _sac_dataset(environment, dataset_name)
        if environment == "drone":
            goal = _mean_pm(_final_window_seed_values(data, "episode_stats/is_in_hole"), 100)
            avoidance = _mean_pm(
                _final_window_seed_values(data, "episode_stats/avoidance_score"),
                decimals=4,
            )
            rows.append((label, goal, avoidance))
        else:
            goal_values = _final_window_seed_values(data, "episode_stats/goal_reached")
            object_values = _final_window_seed_values(data, "episode_stats/object_collected")
            combined_values = (goal_values + object_values) / 2
            rows.append(
                (
                    label,
                    _mean_pm(goal_values, 100),
                    _mean_pm(object_values, 100),
                    _mean_pm(combined_values, 100),
                )
            )
    return rows


def _robot_collection_key(data: pd.DataFrame) -> str:
    if (data["key"] == "episode_stats/object_collected").any():
        return "episode_stats/object_collected"
    return "episode_stats/targets_captured"


def _td3_final_metric_rows(environment: str) -> list[tuple[str, ...]]:
    methods = [
        (OURS_TD3_LABEL, "calf"),
        ("Residual TD3", "residual_td3"),
        ("TD3", "td3"),
    ]

    rows = []
    for label, dataset_name in methods:
        data = _dataset(environment, dataset_name)
        if environment == "drone":
            goal = _mean_pm(_final_window_seed_values(data, "episode_stats/is_in_hole"), 100)
            avoidance = _mean_pm(
                _final_window_seed_values(data, "episode_stats/avoidance_score"),
                decimals=4,
            )
            rows.append((label, goal, avoidance))
        else:
            goal_values = _final_window_seed_values(data, "episode_stats/goal_reached")
            object_values = _final_window_seed_values(data, _robot_collection_key(data))
            combined_values = (goal_values + object_values) / 2
            rows.append(
                (
                    label,
                    _mean_pm(goal_values, 100),
                    _mean_pm(object_values, 100),
                    _mean_pm(combined_values, 100),
                )
            )
    return rows


def _robot_baseline_metric_row() -> tuple[str, str, str, str]:
    path = DATA_ROOT / "robot" / "robot_nominal_eval_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing robot nominal evaluation summary: {path}")
    row = pd.read_csv(path).iloc[0]
    return (
        "Baseline policy",
        f"{row['goal_reached_mean_pct']:.0f} $\\pm$ {row['goal_reached_std_pct']:.0f}",
        f"{row['object_collected_mean_pct']:.2f} $\\pm$ {row['object_collected_std_pct']:.2f}",
        f"{row['combined_mean_pct']:.2f} $\\pm$ {row['combined_std_pct']:.2f}",
    )


def _article_final_metric_rows() -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    drone_rows = [
        *_td3_final_metric_rows("drone"),
        *_sac_final_metric_rows("drone"),
        ("Baseline policy", "100 $\\pm$ 0", "0.54 $\\pm$ 0.18"),
    ]
    robot_rows = [
        *_td3_final_metric_rows("robot"),
        *_sac_final_metric_rows("robot"),
        _robot_baseline_metric_row(),
    ]
    return drone_rows, robot_rows


def generate_article_final_metrics_tables(output_dir: Path | None = None) -> Path:
    out_dir = output_dir or OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "final_metrics_tables.tex"

    drone_rows, robot_rows = _article_final_metric_rows()

    drone_body = "\n".join(
        f"      {method} & {goal} & {avoidance} \\\\"
        for method, goal, avoidance in drone_rows[:3]
    )
    drone_body += "\n      \\midrule\n"
    drone_body += "\n".join(
        f"      {method} & {goal} & {avoidance} \\\\"
        for method, goal, avoidance in drone_rows[3:6]
    )
    drone_body += "\n      \\midrule\n"
    drone_body += (
        f"      {drone_rows[6][0]} & {drone_rows[6][1]} & {drone_rows[6][2]} \\\\"
    )

    robot_body = "\n".join(
        f"      {method} & {goal} & {collected} & {combined} \\\\"
        for method, goal, collected, combined in robot_rows[:3]
    )
    robot_body += "\n      \\midrule\n"
    robot_body += "\n".join(
        f"      {method} & {goal} & {collected} & {combined} \\\\"
        for method, goal, collected, combined in robot_rows[3:6]
    )
    robot_body += "\n      \\midrule\n"
    robot_body += (
        f"      {robot_rows[6][0]} & {robot_rows[6][1]} & {robot_rows[6][2]} "
        f"& {robot_rows[6][3]} \\\\"
    )

    path.write_text(
        r"""\begin{table*}[!t]
  \centering

  \begin{minipage}{0.48\textwidth}
    \centering \scriptsize \setlength{\tabcolsep}{2pt}
    \caption{
      \textbf{Final-stage metrics in the Contaminated-Zone AUV
      Navigation environment.} TD3- and SAC-based methods are
      reported in the same table.
      Results correspond to episodes occurring between 2.7M and 3.0M
      environment steps and are aggregated over ten independent random seeds.
      \textit{Goal reached (\%)} denotes the percentage of episodes
      in which the agent successfully reaches the goal set $\G$
      within the episode horizon.
      \textit{Avoidance score} is defined as the maximum penetration
      depth into contaminated region $\mathcal{C}$ during an episode,
      $\max_{t \in \text{episode}} d((x_t,y_t),\bar{\mathcal{C}})$,
      where $d(\cdot,\cdot)$ is the Euclidean distance and
      $\bar{\mathcal{C}}$ is the closed complement of $\mathcal{C}$.
      Lower avoidance score indicates safer behavior (less intrusion
      into $\mathcal{C}$), while higher goal-reaching rate indicates
      better task success.
      All values are shown as \textit{mean $\pm$ standard deviation}.
    }\label{tab:metrics}
    \begin{tabular}{lcc}
      \toprule Method & Goal reached (\%) & Avoidance score \\
      \midrule
"""
        + drone_body
        + r"""
      \bottomrule
    \end{tabular}
  \end{minipage}\hfill
  \begin{minipage}{0.48\textwidth}
    \centering \scriptsize \setlength{\tabcolsep}{1pt}
    \caption{
      \textbf{Final-stage metrics in the Treasure-Collecting Robot
      environment.} TD3- and SAC-based methods are reported in the same table.
      Results correspond to episodes occurring between 2.7M and 3.0M
      environment steps and are aggregated over ten independent random seeds.
      \textit{Goal reached (\%)} denotes the percentage of episodes
      in which the robot reaches the goal set $\G$ within the episode horizon.
      \textit{Treasure collected (\%)} denotes the percentage of
      episodes in which the robot successfully collects the required
      treasure at least once during the episode.
      Since the task requires achieving both sub-goals, the
      \textit{Combined} metric is defined as
      $\tfrac{1}{2}(\textit{Goal reached} + \textit{Treasure collected})$.
      Higher values indicate better performance.
      All values are shown as \textit{mean $\pm$ standard deviation}.
    }\label{tab:metrics_kin_robot}
    \begin{tabular}{lccc}
      \toprule Method & Goal reached (\%) &
      \makecell{Treasure\\collected (\%)} & Combined \\
      \midrule
"""
        + robot_body
        + r"""
      \bottomrule
    \end{tabular}
  \end{minipage}
\end{table*}
""",
        encoding="utf-8",
    )
    return path


def generate_sac_final_metrics_tables(output_dir: Path | None = None) -> Path:
    out_dir = output_dir or OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sac_final_metrics_tables.tex"

    drone_rows = _sac_final_metric_rows("drone")
    robot_rows = _sac_final_metric_rows("robot")

    drone_body = "\n".join(
        f"      {method} & {goal} & {avoidance} \\\\"
        for method, goal, avoidance in drone_rows
    )
    robot_body = "\n".join(
        f"      {method} & {goal} & {collected} & {combined} \\\\"
        for method, goal, collected, combined in robot_rows
    )

    path.write_text(
        r"""\begin{table*}[!t]
  \centering

  \begin{minipage}{0.48\textwidth}
    \centering \small \setlength{\tabcolsep}{3pt}
    \caption{
      \textbf{Final-stage SAC metrics in the Contaminated-Zone AUV
      Navigation environment.} Results correspond to episodes occurring
      between 2.7M and 3.0M environment steps and are aggregated across
      ten independent seeds. \textit{Goal reached (\%)} is computed from
      the environment's goal-indicator metric; lower avoidance score
      indicates less intrusion into the contaminated region.
      Values are shown as \textit{mean $\pm$ standard deviation}.
    }\label{tab:sac_metrics_drone}
    \begin{tabular}{lcc}
      \toprule Method & Goal reached (\%) & Avoidance score \\ \midrule
"""
        + drone_body
        + r"""
      \bottomrule
    \end{tabular}
  \end{minipage}\hfill
  \begin{minipage}{0.48\textwidth}
    \centering \small \setlength{\tabcolsep}{1pt}
    \caption{
      \textbf{Final-stage SAC metrics in the Treasure-Collecting Robot
      environment.} Results correspond to episodes occurring between
      2.7M and 3.0M environment steps and are aggregated across ten
      independent seeds. The \textit{Combined} metric is
      $\tfrac{1}{2}(\textit{Goal reached} + \textit{Treasure collected})$.
      Values are shown as \textit{mean $\pm$ standard deviation}.
    }\label{tab:sac_metrics_kin_robot}
    \begin{tabular}{lccc}
      \toprule Method & Goal reached (\%) & Treasure collected (\%) & Combined \\
      \midrule
"""
        + robot_body
        + r"""
      \bottomrule
    \end{tabular}
  \end{minipage}
\end{table*}
""",
        encoding="utf-8",
    )
    return path


def _gradient_fill(ax, x, y_low, y_high, color_low, color_high, zorder=1):
    from matplotlib.colors import LinearSegmentedColormap

    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack([gradient, gradient])
    cmap = LinearSegmentedColormap.from_list("_agency_grad", [color_low, color_high])
    y_lo = float(np.nanmin(y_low))
    y_hi = float(np.nanmax(y_high))
    im = ax.imshow(
        gradient,
        aspect="auto",
        origin="lower",
        extent=[float(x.min()), float(x.max()), y_lo, y_hi],
        cmap=cmap,
        zorder=zorder,
    )
    poly = ax.fill_between(x, y_low, y_high, facecolor="none", edgecolor="none")
    paths = poly.get_paths()
    if paths:
        im.set_clip_path(paths[0], transform=ax.transData)
    return im


def generate_graphical_abstract_agency(output_dir: Path | None = None) -> Path:
    drone_baseline_data = _dataset("drone", "calf")
    baseline_mean, _ = _baseline_policy_band(drone_baseline_data)
    learning_mean = 100 - baseline_mean
    steps = baseline_mean.index.to_numpy()

    policyteal = "#2F7997"
    baselinegreen = "#88B9A8"
    softorange = "#F8E0B8"
    learnorange = "#E8893B"

    fig, ax = plt.subplots(figsize=(6.8, 2.5))
    baseline_top = np.full_like(learning_mean.to_numpy(), 100.0)
    _gradient_fill(
        ax, steps, learning_mean.to_numpy(), baseline_top,
        color_low=baselinegreen, color_high=policyteal,
    )
    _gradient_fill(
        ax, steps, np.zeros_like(learning_mean.to_numpy()), learning_mean.to_numpy(),
        color_low=learnorange, color_high=softorange,
    )
    ax.plot(steps, learning_mean, color="white", linewidth=1.2, zorder=3)

    ax.text(
        0.04, 0.68, r"\sffamily\bfseries \% actions from baseline policy",
        transform=ax.transAxes, fontsize=12, color="white",
    )
    ax.text(
        0.49, 0.20, r"\sffamily\bfseries \% actions from learning policy",
        transform=ax.transAxes, fontsize=12, color="white",
    )

    ax.vlines(
        TRANSITION_STEP, 0, 100,
        color="white", linestyle="--", linewidth=1.0, alpha=0.9,
    )

    # Curly brace over the post-transition "full autonomy" region
    from matplotlib.path import Path as _Path
    from matplotlib.patches import PathPatch as _PathPatch
    brace_x1 = float(TRANSITION_STEP)
    brace_x2 = float(steps.max())
    brace_mid = (brace_x1 + brace_x2) / 2
    brace_y = 102.0
    brace_depth = 4.5
    t_off = (brace_x2 - brace_x1) * 0.04
    brace_verts = [
        (brace_x1, brace_y),
        (brace_x1, brace_y + brace_depth * 0.5),
        (brace_mid - t_off, brace_y + brace_depth * 0.5),
        (brace_mid, brace_y + brace_depth * 0.5),
        (brace_mid, brace_y + brace_depth),
        (brace_mid, brace_y + brace_depth * 0.5),
        (brace_mid + t_off, brace_y + brace_depth * 0.5),
        (brace_x2, brace_y + brace_depth * 0.5),
        (brace_x2, brace_y),
    ]
    brace_codes = [_Path.MOVETO] + [_Path.CURVE3] * 8
    ax.add_patch(
        _PathPatch(
            _Path(brace_verts, brace_codes),
            fill=False, edgecolor="#1A3C47", lw=1.2, clip_on=True,
        )
    )
    ax.text(
        brace_mid, brace_y + brace_depth + 1.0,
        "\n".join([
            r"\sffamily\bfseries Full autonomy of",
            r"\sffamily\bfseries the learning policy",
        ]),
        ha="center", va="bottom", fontsize=10, color="#1A3C47",
        linespacing=1.1,
    )

    ax.set_xlim(steps.min(), steps.max())
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 100])
    ax.set_yticklabels([r"\sffamily 0\%", r"\sffamily 100\%"])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlabel(r"\sffamily Training step", fontsize=13)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(False)

    xmin_step, xmax_step = float(steps.min()), float(steps.max())
    learning_arr = learning_mean.to_numpy()

    def _axfrac_to_data(frac: float) -> float:
        return xmin_step + frac * (xmax_step - xmin_step)

    arrow_props = dict(
        arrowstyle="<->", color="white", lw=1.8,
        mutation_scale=14, shrinkA=0, shrinkB=0,
    )
    # Zone-span arrows — both at the same x, stacked at the baseline/learning split
    x_span = _axfrac_to_data(0.46)
    y_split = float(np.interp(x_span, steps, learning_arr))
    ax.annotate(
        "", xy=(x_span, 100), xytext=(x_span, y_split),
        arrowprops=arrow_props, zorder=4,
    )
    ax.annotate(
        "", xy=(x_span, 0), xytext=(x_span, y_split),
        arrowprops=arrow_props, zorder=4,
    )

    fig.tight_layout(pad=0.3)
    fig.canvas.draw()

    mid_idx = int(len(steps) * 0.38)
    neighborhood = max(1, len(steps) // 10)
    idx_a = max(0, mid_idx - neighborhood)
    idx_b = min(len(steps) - 1, mid_idx + neighborhood)
    point_a = ax.transData.transform((steps[idx_a], learning_mean.iloc[idx_a]))
    point_b = ax.transData.transform((steps[idx_b], learning_mean.iloc[idx_b]))
    diagonal_angle = float(
        np.degrees(np.arctan2(point_b[1] - point_a[1], point_b[0] - point_a[0]))
    )
    idx_part1 = int(len(steps) * 0.23)
    idx_part2 = int(len(steps) * 0.64)
    ax.text(
        steps[idx_part1], float(learning_mean.iloc[idx_part1]) - 5.0,
        r"\sffamily\itshape Agency transfers from baseline",
        rotation=diagonal_angle, rotation_mode="anchor",
        ha="center", va="top", color="#1A3C47", fontsize=11, zorder=5,
    )
    ax.text(
        steps[idx_part2], float(learning_mean.iloc[idx_part2]) - 5.0,
        r"\sffamily\itshape to learning policy",
        rotation=diagonal_angle, rotation_mode="anchor",
        ha="center", va="top", color="#1A3C47", fontsize=11, zorder=5,
    )
    return _save(fig, "graphical_abstract_agency.pdf", output_dir)


def generate_all_figures(output_dir: Path | None = None) -> list[Path]:
    return [
        generate_episode_return_comparison(output_dir),
        generate_goal_reaching_rate_comparison(output_dir),
        generate_sac_episode_return_comparison(output_dir),
        generate_sac_goal_reaching_rate_comparison(output_dir),
        generate_baseline_policy_calls_percent(output_dir),
        generate_schedule_parameters(output_dir),
        generate_ablation_episode_return_drone_robot(output_dir),
        generate_calf_td3_drone_ablation_returns(output_dir),
        generate_calf_sac_robot_hyperparameter_sensitivity(output_dir),
        generate_calf_td3_drone_ablation_anneal_is_in_hole(output_dir),
        generate_drone_checkpoint_lambda_distance(output_dir),
        generate_ablation_baseline_calls_robot(output_dir),
        generate_graphical_abstract_agency(output_dir),
    ]
