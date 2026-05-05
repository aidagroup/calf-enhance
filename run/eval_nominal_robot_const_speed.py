#!/usr/bin/env python3
"""Evaluate the nominal policy in RobotNavigationConstSpeedCatch-v0 and save CSV outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math

import gymnasium as gym
import numpy as np
import src  # noqa: F401 - registers gym environments
import tyro
from src import RUN_PATH
from src.envs.robot_navigation_const_speed import ConstSpeedGoalController


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


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
class Args:
    env_id: str = "RobotNavigationConstSpeedCatch-v0"
    """Environment id to evaluate."""
    num_episodes: int = 1000
    """Number of independent episodes to evaluate."""
    start_seed: int = 0
    """Seed of the first episode; episodes use consecutive seeds."""
    per_episode_csv: Path = Path(RUN_PATH) / "expdata" / "robot_nominal_eval_episodes.csv"
    """Output CSV with one row per evaluated episode."""
    summary_csv: Path = Path(RUN_PATH) / "expdata" / "robot_nominal_eval_summary.csv"
    """Output CSV with aggregate statistics."""


def main(args: Args) -> None:
    args.per_episode_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env_id)
    controller = ConstSpeedGoalController()

    rows: list[dict[str, float | int]] = []
    goal_values: list[float] = []
    object_values: list[float] = []
    combined_values: list[float] = []
    capture_counts: list[float] = []
    episode_lengths: list[float] = []

    try:
        for seed in range(args.start_seed, args.start_seed + args.num_episodes):
            obs, info = env.reset(seed=seed)
            terminated = False
            truncated = False
            episode_steps = 0

            while not (terminated or truncated):
                action = controller.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_steps += 1

            goal_reached = float(bool(info.get("goal_reached", False)))
            targets_captured_total = int(info.get("targets_captured_total", 0))
            object_collected = float(targets_captured_total > 0)
            combined = 0.5 * (goal_reached + object_collected)

            row = {
                "seed": seed,
                "episode_steps": episode_steps,
                "goal_reached": int(goal_reached),
                "targets_captured_total": targets_captured_total,
                "object_collected": int(object_collected),
                "combined": combined,
                "distance_to_goal": float(info.get("distance_to_goal", float("nan"))),
            }
            rows.append(row)
            goal_values.append(goal_reached)
            object_values.append(object_collected)
            combined_values.append(combined)
            capture_counts.append(float(targets_captured_total))
            episode_lengths.append(float(episode_steps))
    finally:
        env.close()

    with args.per_episode_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "episode_steps",
                "goal_reached",
                "targets_captured_total",
                "object_collected",
                "combined",
                "distance_to_goal",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_row = {
        "env_id": args.env_id,
        "num_episodes": args.num_episodes,
        "start_seed": args.start_seed,
        "end_seed": args.start_seed + args.num_episodes - 1,
        "goal_reached_mean_pct": 100.0 * _mean(goal_values),
        "goal_reached_std_pct": 100.0 * _std(goal_values),
        "goal_reached_se_pct": 100.0 * _std_error(goal_values),
        "object_collected_mean_pct": 100.0 * _mean(object_values),
        "object_collected_std_pct": 100.0 * _std(object_values),
        "object_collected_se_pct": 100.0 * _std_error(object_values),
        "combined_mean_pct": 100.0 * _mean(combined_values),
        "combined_std_pct": 100.0 * _std(combined_values),
        "combined_se_pct": 100.0 * _std_error(combined_values),
        "avg_targets_captured": _mean(capture_counts),
        "avg_episode_steps": _mean(episode_lengths),
    }

    with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    print(f"Saved per-episode CSV: {args.per_episode_csv}")
    print(f"Saved summary CSV: {args.summary_csv}")
    print(summary_row)


if __name__ == "__main__":
    main(tyro.cli(Args))
