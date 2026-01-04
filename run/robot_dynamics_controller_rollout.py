#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import tyro

from src.controller import RobotDynamicsCollectController, RobotDynamicsGoalController


@dataclass
class Args:
    env_id: str = "RobotDynamics-v0"
    """Environment id to evaluate."""
    seed: int = 0
    """Seed for the initial condition."""
    episodes: int = 20
    """Number of episodes to roll out."""
    max_steps: int = 400
    """Maximum steps per episode."""
    render_mode: Optional[str] = "human"
    """Optional render mode passed to the environment."""
    controller: str = "goal"
    """Controller type: 'goal' or 'collect'."""


def _make_controller(kind: str):
    if kind == "goal":
        return RobotDynamicsGoalController()
    if kind == "collect":
        return RobotDynamicsCollectController()
    raise ValueError(f"Unknown controller '{kind}'. Use 'goal' or 'collect'.")


def main(args: Args):
    controller = _make_controller(args.controller)
    env = gym.make(args.env_id, render_mode=args.render_mode)

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        total_reward = 0.0
        final_info = info
        steps = 0

        for _ in range(args.max_steps):
            action = controller.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            final_info = info
            steps += 1
            if terminated or truncated:
                break

        print(
            f"Episode {episode}: return={total_reward:.3f}, "
            f"steps={steps}, "
            f"goal_reached={final_info.get('goal_reached', False)}, "
            f"distance_to_target={final_info.get('distance_to_target', 0.0):.3f}, "
            f"collectable_captured={final_info.get('collectable_captured', 0)}"
        )

    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
