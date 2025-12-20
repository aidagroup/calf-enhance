#!/usr/bin/env python3
import gymnasium as gym
import mlflow
import numpy as np
import tyro
from dataclasses import dataclass, field
from typing import List, Optional

from src import RUN_PATH
from src.controller import RobotNavigationGoalController
from src.utils.robot_nav_logging import log_robot_nav_trajectory


def _rollout_episode(
    env_id: str,
    seed: int,
    max_steps: int,
    render_mode: Optional[str],
    controller: RobotNavigationGoalController,
):
    env = gym.make(env_id, seed=seed, render_mode=render_mode)
    obs, info = env.reset(seed=seed)
    episode_trajectory = []
    total_reward = 0.0
    final_info = info

    for _ in range(max_steps):
        action = controller.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_trajectory.append(
            {
                "obs": obs.copy(),
                "actions": np.array(action).copy(),
                "reward": np.array(reward).copy(),
            }
        )
        total_reward += float(reward)
        obs = next_obs
        final_info = info

        if terminated or truncated:
            break

    env.close()
    goal_reached = bool(final_info.get("goal_reached", False))
    distance_to_goal = float(final_info.get("distance_to_goal", 0.0))

    return {
        "trajectory": episode_trajectory,
        "total_reward": total_reward,
        "goal_reached": goal_reached,
        "distance_to_goal": distance_to_goal,
        "steps": len(episode_trajectory),
        "seed": seed,
        "targets_captured_total": final_info.get("targets_captured_total", 0),
    }


@dataclass
class Args:
    env_id: str = "RobotNavigation-v0"
    """Environment id to evaluate."""
    seeds: List[int] = field(default_factory=lambda: list(range(200)))
    """List of seeds / initial conditions to roll out."""
    max_steps: int = 300
    """Maximum steps per rollout."""
    render_mode: Optional[str] = None
    """Optional render mode passed to the environment."""
    log_mlflow: bool = True
    """Whether to log metrics and trajectory plots to MLflow."""
    experiment_name: str = "robot_nav_goal_controller"
    """MLflow experiment name (when logging is enabled)."""
    run_name: Optional[str] = None
    """Optional MLflow run name (default derives from experiment and seeds)."""


def main(args: Args):
    controller = RobotNavigationGoalController()
    results = []
    rewards = []
    targets_captured = []

    mlflow_run = None
    if args.log_mlflow:
        mlflow.set_tracking_uri(f"file://{RUN_PATH}/mlruns")
        mlflow.set_experiment(args.experiment_name)
        mlflow_run = mlflow.start_run(
            run_name=args.run_name
            or f"{args.experiment_name}__{','.join(map(str, args.seeds))}"
        )

    try:
        for idx, seed in enumerate(args.seeds):
            rollout = _rollout_episode(
                env_id=args.env_id,
                seed=seed,
                max_steps=args.max_steps,
                render_mode=args.render_mode,
                controller=controller,
            )
            results.append(rollout)
            rewards.append(rollout["total_reward"])
            targets_captured.append(rollout["targets_captured_total"])
            print(
                f"Seed {seed}: return={rollout['total_reward']:.3f}, "
                f"steps={rollout['steps']}, goal_reached={rollout['goal_reached']}, "
                f"distance_to_goal={rollout['distance_to_goal']:.3f}, "
                f"targets_captured={rollout['targets_captured_total']}"
            )
            print(f"Avg reward: {np.mean(rewards)}")
            print(f"Avg targets captured: {np.mean(targets_captured)}")
            if args.log_mlflow and mlflow_run is not None:
                mlflow.log_metric("return", rollout["total_reward"], step=idx)
                mlflow.log_metric(
                    "distance_to_goal", rollout["distance_to_goal"], step=idx
                )
                mlflow.log_metric(
                    "goal_reached", float(rollout["goal_reached"]), step=idx
                )
                log_robot_nav_trajectory(
                    rollout["trajectory"],
                    global_step=seed,
                    total_reward=rollout["total_reward"],
                    goal_reached=rollout["goal_reached"],
                    artifact_subdir="robot_navigation/controller_rollouts",
                )
    finally:
        if args.log_mlflow and mlflow_run is not None:
            mlflow.end_run()

    return results


if __name__ == "__main__":
    main(tyro.cli(Args))
