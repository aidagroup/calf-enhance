#!/usr/bin/env python3
"""Verify copied robot nominal-eval CSV files and recompute summary statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import math


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "code" / "calf-enhance-repo" / "run" / "expdata"
TARGET_DIR = REPO_ROOT / "code" / "plotting" / "expdata" / "cleared" / "robot"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _std_error(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return _std(values) / math.sqrt(len(values))


@dataclass
class SummaryMetrics:
    env_id: str
    num_episodes: int
    start_seed: int
    end_seed: int
    goal_reached_mean_pct: float
    goal_reached_std_pct: float
    goal_reached_se_pct: float
    object_collected_mean_pct: float
    object_collected_std_pct: float
    object_collected_se_pct: float
    combined_mean_pct: float
    combined_std_pct: float
    combined_se_pct: float
    avg_targets_captured: float
    avg_episode_steps: float


def _compute_summary(episode_rows: list[dict[str, str]], env_id: str) -> SummaryMetrics:
    seeds = [int(row["seed"]) for row in episode_rows]
    goal_values = [float(row["goal_reached"]) for row in episode_rows]
    object_values = [float(row["object_collected"]) for row in episode_rows]
    combined_values = [float(row["combined"]) for row in episode_rows]
    capture_counts = [float(row["targets_captured_total"]) for row in episode_rows]
    episode_lengths = [float(row["episode_steps"]) for row in episode_rows]

    return SummaryMetrics(
        env_id=env_id,
        num_episodes=len(episode_rows),
        start_seed=min(seeds),
        end_seed=max(seeds),
        goal_reached_mean_pct=100.0 * _mean(goal_values),
        goal_reached_std_pct=100.0 * _std(goal_values),
        goal_reached_se_pct=100.0 * _std_error(goal_values),
        object_collected_mean_pct=100.0 * _mean(object_values),
        object_collected_std_pct=100.0 * _std(object_values),
        object_collected_se_pct=100.0 * _std_error(object_values),
        combined_mean_pct=100.0 * _mean(combined_values),
        combined_std_pct=100.0 * _std(combined_values),
        combined_se_pct=100.0 * _std_error(combined_values),
        avg_targets_captured=_mean(capture_counts),
        avg_episode_steps=_mean(episode_lengths),
    )


def _assert_close(name: str, actual: float, expected: float, tol: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{name} mismatch: actual={actual}, expected={expected}")


def main() -> None:
    source_episode_path = SOURCE_DIR / "robot_nominal_eval_episodes.csv"
    source_summary_path = SOURCE_DIR / "robot_nominal_eval_summary.csv"
    target_episode_path = TARGET_DIR / "robot_nominal_eval_episodes.csv"
    target_summary_path = TARGET_DIR / "robot_nominal_eval_summary.csv"

    for path in [
        source_episode_path,
        source_summary_path,
        target_episode_path,
        target_summary_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    source_episode_hash = _sha256(source_episode_path)
    target_episode_hash = _sha256(target_episode_path)
    source_summary_hash = _sha256(source_summary_path)
    target_summary_hash = _sha256(target_summary_path)

    if source_episode_hash != target_episode_hash:
        raise AssertionError("Per-episode CSV copy does not match source.")
    if source_summary_hash != target_summary_hash:
        raise AssertionError("Summary CSV copy does not match source.")

    source_episode_rows = _load_csv_rows(source_episode_path)
    target_episode_rows = _load_csv_rows(target_episode_path)
    if source_episode_rows != target_episode_rows:
        raise AssertionError("Per-episode CSV rows differ despite matching file hashes.")

    summary_row = _load_csv_rows(target_summary_path)[0]
    computed = _compute_summary(
        target_episode_rows,
        env_id=summary_row["env_id"],
    )

    if computed.num_episodes != int(summary_row["num_episodes"]):
        raise AssertionError("num_episodes mismatch")
    if computed.start_seed != int(summary_row["start_seed"]):
        raise AssertionError("start_seed mismatch")
    if computed.end_seed != int(summary_row["end_seed"]):
        raise AssertionError("end_seed mismatch")

    _assert_close(
        "goal_reached_mean_pct",
        computed.goal_reached_mean_pct,
        float(summary_row["goal_reached_mean_pct"]),
    )
    _assert_close(
        "goal_reached_std_pct",
        computed.goal_reached_std_pct,
        float(summary_row["goal_reached_std_pct"]),
    )
    _assert_close(
        "goal_reached_se_pct",
        computed.goal_reached_se_pct,
        float(summary_row["goal_reached_se_pct"]),
    )
    _assert_close(
        "object_collected_mean_pct",
        computed.object_collected_mean_pct,
        float(summary_row["object_collected_mean_pct"]),
    )
    _assert_close(
        "object_collected_std_pct",
        computed.object_collected_std_pct,
        float(summary_row["object_collected_std_pct"]),
    )
    _assert_close(
        "object_collected_se_pct",
        computed.object_collected_se_pct,
        float(summary_row["object_collected_se_pct"]),
    )
    _assert_close(
        "combined_mean_pct",
        computed.combined_mean_pct,
        float(summary_row["combined_mean_pct"]),
    )
    _assert_close(
        "combined_std_pct",
        computed.combined_std_pct,
        float(summary_row["combined_std_pct"]),
    )
    _assert_close(
        "combined_se_pct",
        computed.combined_se_pct,
        float(summary_row["combined_se_pct"]),
    )
    _assert_close(
        "avg_targets_captured",
        computed.avg_targets_captured,
        float(summary_row["avg_targets_captured"]),
    )
    _assert_close(
        "avg_episode_steps",
        computed.avg_episode_steps,
        float(summary_row["avg_episode_steps"]),
    )

    print("Verified robot nominal evaluation CSV files.")
    print(f"Source per-episode CSV: {source_episode_path}")
    print(f"Copied per-episode CSV: {target_episode_path}")
    print(f"Source summary CSV: {source_summary_path}")
    print(f"Copied summary CSV: {target_summary_path}")
    print(f"SHA256 per-episode CSV: {target_episode_hash}")
    print(f"SHA256 summary CSV: {target_summary_hash}")
    print(
        "Recomputed metrics: "
        f"goal={computed.goal_reached_mean_pct:.2f}%, "
        f"object={computed.object_collected_mean_pct:.2f}%, "
        f"combined={computed.combined_mean_pct:.2f}%"
    )


if __name__ == "__main__":
    main()
