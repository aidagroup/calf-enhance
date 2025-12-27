from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from src.utils.metrics_controller import MetricsCollector


@dataclass
class RobotDynamicsConfig:
    """Configuration values for the robot dynamics environment."""

    max_speed: float = 0.15
    control_dt: float = 0.05
    max_angular_velocity: float = math.pi
    world_low: float = 0.0
    world_high: float = 1.0
    start_position_distribution: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (0.5, 0.1),
        (0.9, 0.9),
    )
    start_angle_distribution: Tuple[float, float] = (0.0, 2 * math.pi)
    target_position: Tuple[float, float] = (0, 0.5)
    target_radius: float = 0.05
    n_collectables: int = 3
    collectable_radius: float = 0.1
    collectable_reward: float = 50.0
    terminal_reward: float = 10.0


class RobotDynamicsEnv(gym.Env[np.ndarray, np.ndarray]):
    """Simple 2D robot dynamics environment without obstacles or rendering."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        config: Optional[RobotDynamicsConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if render_mode is not None:
            raise ValueError("RobotDynamicsEnv does not support rendering.")
        self.render_mode = render_mode
        self.config = config or RobotDynamicsConfig()
        if self.config.control_dt <= 0.0:
            raise ValueError("RobotDynamicsEnv control_dt must be positive.")

        self._rng: np.random.Generator
        self._steps = 0
        self._last_action = np.zeros(2, dtype=np.float32)
        self.robot_position = np.zeros(2, dtype=np.float32)
        self.robot_angle = 0.0
        self.target_position = np.array(self.config.target_position, dtype=np.float32)
        self.target_radius = self.config.target_radius
        self.collectable_positions = np.zeros(
            (self.config.n_collectables, 2), dtype=np.float32
        )
        self.collectable_captured_mask = [False] * self.config.n_collectables

        self.action_space = spaces.Box(
            low=np.array(
                [-self.config.max_speed, -self.config.max_angular_velocity],
                dtype=np.float32,
            ),
            high=np.array(
                [self.config.max_speed, self.config.max_angular_velocity],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        # Observation: [robot_x, robot_y, cos(angle), sin(angle), collectable_x, collectable_y, collectable_captured]
        obs_low = np.array(
            [
                self.config.world_low - self.config.world_high,
                self.config.world_low - self.config.world_high,
                -1.0,
                -1.0,
            ]
            + [
                self.config.world_low - self.config.world_high,
                self.config.world_low - self.config.world_high,
                0.0,
            ]
            * self.config.n_collectables,
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                self.config.world_high - self.config.world_low,
                self.config.world_high - self.config.world_low,
                1.0,
                1.0,
            ]
            + [
                self.config.world_high - self.config.world_low,
                self.config.world_high - self.config.world_low,
                1.0,
            ]
            * self.config.n_collectables,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        if seed is None:
            self.np_random, _ = seeding.np_random(None)
        else:
            self.np_random, _ = seeding.np_random(seed)
        self._rng = self.np_random

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        self._rng = self.np_random

        position = self._rng.uniform(
            self.config.start_position_distribution[0],
            self.config.start_position_distribution[1],
            2,
        )
        angle = self._rng.uniform(
            self.config.start_angle_distribution[0],
            self.config.start_angle_distribution[1],
        )

        self.robot_position = position
        self.robot_angle = self._wrap_angle(angle)

        # Spawn collectable at random position within world bounds
        self.collectable_positions = self._rng.uniform(
            self.config.world_low,
            self.config.world_high,
            (self.config.n_collectables, 2),
        ).astype(np.float32)
        self.collectable_captured = [False] * self.config.n_collectables

        observation = self._get_observation()
        info = {
            "distance_to_target": np.linalg.norm(
                self.robot_position - self.target_position
            ),
            "collectable_captured": sum(self.collectable_captured),
        }
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        speed = float(action[0])
        angular_velocity = float(action[1])

        delta = np.array(
            [math.cos(self.robot_angle), math.sin(self.robot_angle)],
            dtype=np.float32,
        )
        self.robot_position = self.robot_position + delta * (
            speed * self.config.control_dt
        )
        self.robot_position = np.clip(
            self.robot_position,
            self.config.world_low,
            self.config.world_high,
        )
        self.robot_angle = self._wrap_angle(
            self.robot_angle + angular_velocity * self.config.control_dt
        )

        # Check if collectable is captured
        collectable_reward = 0.0
        for i in range(self.config.n_collectables):
            if not self.collectable_captured[i]:
                distance_to_collectable = np.linalg.norm(
                    self.robot_position - self.collectable_positions[i]
                )
                if distance_to_collectable < self.config.collectable_radius:
                    collectable_reward = self.config.collectable_reward
                    self.collectable_captured[i] = True

        observation = self._get_observation()
        distance_to_target = np.linalg.norm(self.robot_position - self.target_position)
        reward = -distance_to_target + collectable_reward
        terminated = distance_to_target < self.target_radius
        reward += terminated * self.config.terminal_reward

        info = {
            "distance_to_target": distance_to_target,
            "collectable_captured": sum(self.collectable_captured),
            "goal_reached": terminated,
        }
        return observation, reward, terminated, False, info

    def _get_observation(self) -> np.ndarray:
        return np.array(
            [
                self.robot_position[0] - self.target_position[0],
                self.robot_position[1] - self.target_position[1],
                np.cos(self.robot_angle),
                np.sin(self.robot_angle),
            ]
            + sum(
                [
                    [
                        (
                            self.robot_position[0] - self.target_position[0]
                            if self.collectable_captured[i]
                            else self.robot_position[0]
                            - self.collectable_positions[i][0]
                        ),
                        (
                            self.robot_position[1] - self.target_position[1]
                            if self.collectable_captured[i]
                            else self.robot_position[1]
                            - self.collectable_positions[i][1]
                        ),
                        self.collectable_captured[i],
                    ]
                    for i in range(len(self.collectable_captured))
                ],
                [],
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi


class RobotDynamicsMetricsCollector(MetricsCollector):
    def __init__(self, rolling_window_size: int = 20):
        super().__init__()
        self.rolling_window_size = rolling_window_size

    def collect_metrics_from_final_episode_info(self, info: dict, step: int):
        super().collect_metrics_from_final_episode_info(info, step)
        self.append_metric(
            "episode_stats/distance_to_target", info["distance_to_target"], step=step
        )
        self.append_metric(
            "episode_stats/collectable_captured",
            info["collectable_captured"],
            step=step,
        )
        self.append_metric(
            "episode_stats/goal_reached", float(info["goal_reached"]), step=step
        )
        self.rolling_window["collectable_captured"].append(info["collectable_captured"])
        self.rolling_window["goal_reached"].append(float(info["goal_reached"]))
        self.append_metric(
            f"episode_stats/goal_reached_rolling_{self.rolling_window_size}",
            np.mean(self.rolling_window["goal_reached"]),
            step=step,
        )
        self.append_metric(
            f"episode_stats/collectable_captured_rolling_{self.rolling_window_size}",
            np.mean(self.rolling_window["collectable_captured"]),
            step=step,
        )
