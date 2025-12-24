from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding


@dataclass
class RobotDynamicsConfig:
    """Configuration values for the robot dynamics environment."""

    max_steps: int = 400
    max_speed: float = 0.15
    control_dt: float = 0.05
    max_angular_velocity: float = math.pi
    world_low: float = 0.0
    world_high: float = 1.0
    start_position_distribution: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.5, 0.1), (0.9, 0.9))
    start_angle: float = 0.0
    clip_position: bool = True
    target_position: Tuple[float, float] = (0, 0.5)
    target_radius: float = 0.05


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
    
        self.action_space = spaces.Box(
            low=np.array(
                [0.0, -self.config.max_angular_velocity],
                dtype=np.float32,
            ),
            high=np.array(
                [self.config.max_speed, self.config.max_angular_velocity],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        obs_low = np.array(
            [self.config.world_low, self.config.world_low, -1.0, -1.0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [self.config.world_high, self.config.world_high, 1.0, 1.0],
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

        self._steps = 0
        self._last_action = np.zeros(2, dtype=np.float32)

        position = self._rng.uniform(self.config.start_position_distribution[0], self.config.start_position_distribution[1], 2)
        angle = self._rng.uniform(0, 2 * math.pi)
        if options:
            if "position" in options:
                position = np.asarray(options["position"], dtype=np.float32)
            if "angle" in options:
                angle = float(options["angle"])

        self.robot_position = position
        self.robot_angle = self._wrap_angle(angle)

        observation = self._get_observation()
        info = {
            "robot_angle": self.robot_angle,
            "last_action": self._last_action.copy(),
        }
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._last_action = action
        self._steps += 1

        speed = float(action[0])
        angular_velocity = float(action[1])

        delta = np.array(
            [math.cos(self.robot_angle), math.sin(self.robot_angle)],
            dtype=np.float32,
        )
        self.robot_position = self.robot_position + delta * (
            speed * self.config.control_dt
        )
        if self.config.clip_position:                
            self.robot_position = np.clip(
                self.robot_position,
                self.config.world_low,
                self.config.world_high,
            )
        self.robot_angle = self._wrap_angle(
            self.robot_angle + angular_velocity * self.config.control_dt
        )

        observation = self._get_observation()
        distance_to_target = np.linalg.norm(self.robot_position - self.target_position)
        reward = -distance_to_target * 0.001
        if self._steps % 100 == 0:
            print(f"Distance to target: {distance_to_target}")
        terminated = distance_to_target < self.target_radius

        truncated = self._steps >= self.config.max_steps
        info = {
            "robot_angle": self.robot_angle,
            "last_action": self._last_action.copy(),
            "step": self._steps,
        }
        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        return np.array(
            [self.robot_position[0], self.robot_position[1], np.cos(self.robot_angle), np.sin(self.robot_angle)],
            dtype=np.float32,
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
