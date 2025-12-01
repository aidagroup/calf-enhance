from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding


@dataclass
class RobotNavigationConfig:
    """Configuration values for the robot navigation environment."""

    max_steps: int = 300
    obstacle_count: int = 4
    obstacle_radius_range: Tuple[float, float] = (0.05, 0.12)
    max_speed: float = 0.012
    success_radius: float = 0.05
    window_size: Tuple[int, int] = (800, 600)
    obstacle_padding: float = 0.005
    obstacle_placement_attempts: int = 40
    obstacle_radius_shrink_factor: float = 0.85
    obstacle_radius_shrink_steps: int = 5
    obstacle_collision_penalty: float = 5.0
    moving_obstacle_count: int = 0
    moving_speed_range: Tuple[float, float] = (0.008, 0.024)
    moving_direction_change_prob: float = 0.01
    moving_obstacle_radius: Optional[float] = None
    moving_obstacle_speed: Optional[float] = None
    moving_obstacle_x_range: Optional[Tuple[float, float]] = (0.1, 0.9)


class RobotNavigationEnv(gym.Env[np.ndarray, np.ndarray]):
    """Simple 2D robot navigation environment with a top-down pygame renderer."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        config: Optional[RobotNavigationConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render_mode '{render_mode}'. Use None, 'human', or 'rgb_array'.")
        self.render_mode = render_mode
        self.config = config or RobotNavigationConfig()

        self._rng: np.random.Generator
        self._steps = 0
        self._last_action = np.zeros(2, dtype=np.float32)

        self.robot_position = np.zeros(2, dtype=np.float32)
        self.robot_angle = 0.0
        self.goal_position = np.array([0.0, 0.5], dtype=np.float32)
        self._obstacles = np.zeros((self.config.obstacle_count, 3), dtype=np.float32)
        self._num_obstacles = 0
        self._moving_indices: list[int] = []
        self._moving_velocities = np.zeros((0, 2), dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([0.0, -math.pi], dtype=np.float32),
            high=np.array([1.0, math.pi], dtype=np.float32),
            dtype=np.float32,
        )

        obs_low = np.zeros(2 + 1 + 2 + 3 * self.config.obstacle_count, dtype=np.float32)
        obs_high = np.ones_like(obs_low)
        obs_low[2] = -math.pi
        obs_high[2] = math.pi
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        self._pygame = None
        self._window = None
        self._clock = None
        self._surface = None

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

        # Robot starts on the right side of the room
        self.robot_position = np.array(
            [
                self._rng.uniform(0.7, 0.9),
                self._rng.uniform(0.1, 0.9),
            ],
            dtype=np.float32,
        )
        self.robot_angle = math.pi  # start roughly facing the goal

        # Sample random obstacles on the left side without overlaps
        self._obstacles.fill(0.0)
        target_obstacles = self.config.obstacle_count
        placed = 0
        attempts_per_obstacle = self.config.obstacle_placement_attempts

        moving_count = min(self.config.moving_obstacle_count, target_obstacles)
        if moving_count > 0:
            selection = self._rng.choice(
                target_obstacles, size=moving_count, replace=False
            )
            selection = np.atleast_1d(selection)
            self._moving_indices = sorted(int(x) for x in selection)
        else:
            self._moving_indices = []
        moving_indices_set = set(self._moving_indices)

        radius_low, radius_high = self.config.obstacle_radius_range
        shrink_factor = self.config.obstacle_radius_shrink_factor
        shrink_steps = max(1, self.config.obstacle_radius_shrink_steps)

        while placed < target_obstacles:
            success = False
            current_high = radius_high
            is_moving = placed in moving_indices_set

            for _ in range(shrink_steps):
                for _ in range(attempts_per_obstacle):
                    if is_moving and self.config.moving_obstacle_radius is not None:
                        radius = float(self.config.moving_obstacle_radius)
                    else:
                        radius = float(self._rng.uniform(radius_low, current_high))

                    if is_moving and self.config.moving_obstacle_x_range is not None:
                        range_low, range_high = self.config.moving_obstacle_x_range
                        x_low = max(range_low + radius, radius)
                        x_high = min(range_high - radius, 1.0 - radius)
                    else:
                        x_low = 0.1 + radius
                        x_high = 0.45
                    if x_high <= x_low:
                        x_high = x_low + 1e-3

                    x = float(self._rng.uniform(x_low, x_high))
                    y = float(self._rng.uniform(0.1 + radius, 0.9 - radius))
                    candidate = np.array([x, y, radius], dtype=np.float32)

                    if self._is_valid_obstacle(candidate, placed):
                        self._obstacles[placed] = candidate
                        placed += 1
                        success = True
                        break

                if success:
                    break
                current_high = max(radius_low, current_high * shrink_factor)

            if not success:
                raise RuntimeError(
                    "Failed to place non-overlapping obstacles; consider reducing obstacle_count or padding."
                )

        self._num_obstacles = self.config.obstacle_count
        self._initialize_moving_obstacles()

        observation = self._get_observation()
        info = {
            "distance_to_goal": self._distance_to_goal(),
            "robot_angle": self.robot_angle,
        }
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._last_action = action
        self._steps += 1

        self._update_moving_obstacles()

        speed = float(action[0])
        angle = float(action[1])
        if speed > 1e-4:
            self.robot_angle = angle

        delta = np.array(
            [math.cos(self.robot_angle), math.sin(self.robot_angle)],
            dtype=np.float32,
        )
        self.robot_position = np.clip(
            self.robot_position + delta * (speed * self.config.max_speed),
            0.0,
            1.0,
        )

        distance = self._distance_to_goal()
        terminated = distance <= self.config.success_radius
        truncated = self._steps >= self.config.max_steps

        in_obstacle = self._is_in_obstacle(self.robot_position)

        # Reward encourages closing the distance, penalizes obstacle incursions, bonus on success
        reward = -distance
        if in_obstacle:
            reward -= self.config.obstacle_collision_penalty
        if terminated:
            reward += 1.0

        observation = self._get_observation()
        info = {
            "distance_to_goal": distance,
            "num_obstacles": self._num_obstacles,
            "last_action": self._last_action.copy(),
            "robot_angle": self.robot_angle,
            "in_obstacle": in_obstacle,
            "goal_reached": terminated,
            "moving_obstacles": len(self._moving_indices),
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            raise RuntimeError("Render mode must be 'human' or 'rgb_array' to draw the simulation.")

        self._ensure_pygame()
        pygame = self._pygame
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
        else:
            pygame.event.pump()

        width, height = self.config.window_size
        if self._surface is None:
            self._surface = pygame.Surface((width, height))

        self._surface.fill((240, 240, 240))
        self._draw_room()
        self._draw_goal()
        self._draw_obstacles()
        center = self._world_to_screen(self.robot_position)
        self._draw_robot_body(center, min(width, height))

        if self.render_mode == "human":
            self._window.blit(self._surface, (0, 0))
            pygame.display.flip()
            if self._clock is not None:
                self._clock.tick(self.metadata["render_fps"])
            return None

        frame = pygame.surfarray.array3d(self._surface)
        frame = np.transpose(frame, (1, 0, 2))
        if self._clock is not None:
            self._clock.tick(self.metadata["render_fps"])
        return frame

    def close(self) -> None:
        if self._pygame is None:
            return
        pygame = self._pygame
        if self._window is not None:
            pygame.display.quit()
            self._window = None
        if self._surface is not None:
            self._surface = None
        if self._clock is not None:
            self._clock = None
        pygame.quit()
        self._pygame = None

    def _get_observation(self) -> np.ndarray:
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        observation[0:2] = self.robot_position
        observation[2] = self.robot_angle
        observation[3:5] = self.goal_position
        obstacle_slice = 5 + 3 * self._num_obstacles
        observation[5:obstacle_slice] = self._obstacles[: self._num_obstacles].reshape(-1)
        return observation

    def _initialize_moving_obstacles(self) -> None:
        count = len(self._moving_indices)
        if count <= 0:
            self._moving_indices = []
            self._moving_velocities = np.zeros((0, 2), dtype=np.float32)
            return

        self._moving_velocities = np.zeros((count, 2), dtype=np.float32)
        for idx in range(count):
            self._moving_velocities[idx] = self._sample_moving_velocity()

    def _sample_moving_velocity(self) -> np.ndarray:
        if self.config.moving_obstacle_speed is not None:
            speed = float(self.config.moving_obstacle_speed)
        else:
            speed_low, speed_high = self.config.moving_speed_range
            speed = float(self._rng.uniform(speed_low, speed_high))
        angle = float(self._rng.uniform(0.0, 2 * math.pi))
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        return direction * speed

    def _update_moving_obstacles(self) -> None:
        if not self._moving_indices:
            return

        for vel_idx, obstacle_idx in enumerate(self._moving_indices):
            if obstacle_idx >= self._num_obstacles:
                continue

            radius = self._obstacles[obstacle_idx, 2]
            if radius <= 0:
                continue

            if (
                self.config.moving_direction_change_prob > 0.0
                and self._rng.random() < self.config.moving_direction_change_prob
            ):
                self._moving_velocities[vel_idx] = self._sample_moving_velocity()

            velocity = self._moving_velocities[vel_idx]
            position = self._obstacles[obstacle_idx, 0:2]
            new_pos = position + velocity

            if new_pos[0] - radius < 0.0 or new_pos[0] + radius > 1.0:
                velocity[0] *= -1.0
                new_pos[0] = np.clip(new_pos[0], radius, 1.0 - radius)

            if new_pos[1] - radius < 0.0 or new_pos[1] + radius > 1.0:
                velocity[1] *= -1.0
                new_pos[1] = np.clip(new_pos[1], radius, 1.0 - radius)

            overlap = False
            for other_idx in range(self._num_obstacles):
                if other_idx == obstacle_idx:
                    continue
                other = self._obstacles[other_idx]
                other_radius = other[2]
                if other_radius <= 0:
                    continue
                distance = np.linalg.norm(new_pos - other[:2])
                if distance < radius + other_radius:
                    overlap = True
                    break

            if overlap:
                velocity *= -1.0
                self._moving_velocities[vel_idx] = velocity
            else:
                self._obstacles[obstacle_idx, 0:2] = new_pos
                self._moving_velocities[vel_idx] = velocity

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal_position - self.robot_position))

    def _is_valid_obstacle(self, candidate: np.ndarray, count: int) -> bool:
        padding = self.config.obstacle_padding
        for other in self._obstacles[:count]:
            if not other.any():
                continue
            distance = np.linalg.norm(candidate[:2] - other[:2])
            if distance < candidate[2] + other[2] + padding:
                return False
        return True

    def _is_in_obstacle(self, position: np.ndarray) -> bool:
        for obstacle in self._obstacles[: self._num_obstacles]:
            if not obstacle.any():
                continue
            center = obstacle[:2]
            radius = obstacle[2]
            if np.linalg.norm(position - center) <= radius:
                return True
        return False

    def _ensure_pygame(self) -> None:
        if self.render_mode is None:
            return
        if self._pygame is not None:
            return
        import pygame  # Lazy import to keep dependency optional at runtime

        pygame.init()
        self._pygame = pygame
        if self.render_mode == "human":
            self._window = pygame.display.set_mode(self.config.window_size)
            pygame.display.set_caption("Robot Navigation")
        else:
            self._window = None
        self._surface = pygame.Surface(self.config.window_size)
        self._clock = pygame.time.Clock()

    def _world_to_screen(self, position: np.ndarray) -> Tuple[int, int]:
        width, height = self.config.window_size
        x = int(position[0] * width)
        y = int((1.0 - position[1]) * height)
        return x, y

    def _draw_room(self) -> None:
        pygame = self._pygame
        width, height = self.config.window_size
        border_color = (50, 50, 50)
        pygame.draw.rect(self._surface, border_color, pygame.Rect(0, 0, width, height), width=4)

    def _draw_goal(self) -> None:
        pygame = self._pygame
        width, height = self.config.window_size
        goal_height = int(height * 0.2)
        goal_rect = pygame.Rect(0, int(height * 0.4), int(width * 0.02), goal_height)
        pygame.draw.rect(self._surface, (50, 200, 50), goal_rect)

    def _draw_obstacles(self) -> None:
        pygame = self._pygame
        width, height = self.config.window_size
        for idx in range(self._num_obstacles):
            x, y, radius = self._obstacles[idx]
            center = self._world_to_screen(np.array([x, y], dtype=np.float32))
            pygame.draw.circle(
                self._surface,
                (180, 60, 60),
                center,
                int(radius * min(width, height)),
            )

    def _draw_robot(self) -> None:
        width, height = self.config.window_size
        center = self._world_to_screen(self.robot_position)
        self._draw_robot_body(center, min(width, height))

    def _draw_robot_body(self, center: Tuple[int, int], scale: int) -> None:
        pygame = self._pygame
        body_radius = max(6, int(0.045 * scale))

        speed_ratio = float(np.clip(self._last_action[0], 0.0, 1.0))
        pulse_radius = body_radius + int(2 + 3 * math.sin(self._steps * 0.3 + speed_ratio * math.pi))
        pulse_radius = max(pulse_radius, body_radius + 1)

        # Draw shimmering aura
        pygame.draw.circle(self._surface, (180, 200, 255), center, pulse_radius, width=2)

        # Base body
        pygame.draw.circle(self._surface, (60, 120, 240), center, body_radius)
        pygame.draw.circle(self._surface, (20, 60, 160), center, body_radius, width=2)
        highlight = (center[0] + int(body_radius * 0.4), center[1] - int(body_radius * 0.4))
        pygame.draw.circle(self._surface, (220, 240, 255), highlight, max(1, body_radius // 3))

        heading_world = np.array(
            [math.cos(self.robot_angle), math.sin(self.robot_angle)],
            dtype=np.float32,
        )
        heading_screen = np.array([heading_world[0], -heading_world[1]], dtype=np.float32)
        perp_screen = np.array([-heading_screen[1], heading_screen[0]], dtype=np.float32)

        nose_offset = heading_screen * body_radius * 1.3
        tail_offset = heading_screen * body_radius * 0.7
        wing_offset = perp_screen * body_radius * 0.8

        nose = (
            int(center[0] + nose_offset[0]),
            int(center[1] + nose_offset[1]),
        )
        left_wing = (
            int(center[0] - tail_offset[0] + wing_offset[0]),
            int(center[1] - tail_offset[1] + wing_offset[1]),
        )
        right_wing = (
            int(center[0] - tail_offset[0] - wing_offset[0]),
            int(center[1] - tail_offset[1] - wing_offset[1]),
        )

        pygame.draw.polygon(self._surface, (250, 250, 255), [nose, left_wing, right_wing])
        pygame.draw.polygon(self._surface, (40, 80, 200), [nose, left_wing, right_wing], width=1)

        # Thruster animation
        thruster_wave = 0.7 + 0.3 * (math.sin(self._steps * 0.5 + speed_ratio * 2.0) + 1) / 2
        thruster_length = body_radius * (0.4 + 0.6 * speed_ratio) * thruster_wave
        thruster_vec = -heading_screen * thruster_length
        thruster_base = (
            int(center[0] - tail_offset[0]),
            int(center[1] - tail_offset[1]),
        )
        thruster_end = (
            int(thruster_base[0] + thruster_vec[0]),
            int(thruster_base[1] + thruster_vec[1]),
        )
        flame_color = (
            255,
            int(160 + 70 * speed_ratio),
            int(80 + 100 * speed_ratio),
        )
        pygame.draw.line(self._surface, flame_color, thruster_base, thruster_end, width=3)


class SimpleGoalController:
    """Greedy controller that moves toward the goal with speed-angle commands."""

    def __init__(self, *, max_speed: float) -> None:
        self.max_speed = float(max_speed)

    def act(self, observation: np.ndarray) -> np.ndarray:
        robot = observation[0:2]
        current_angle = float(observation[2])
        goal = observation[3:5]
        delta = goal - robot

        distance = np.linalg.norm(delta)
        if distance < 1e-8:
            return np.array([0.0, current_angle], dtype=np.float32)

        desired_angle = float(math.atan2(delta[1], delta[0]))
        # Saturate speed so the robot slows as it nears the goal
        if self.max_speed <= 0.0:
            speed_ratio = 0.0
        else:
            speed_ratio = min(1.0, distance / (self.max_speed * 8.0))
        speed_ratio = float(np.clip(speed_ratio, 0.0, 1.0))
        return np.array([speed_ratio, desired_angle], dtype=np.float32)
