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
class RobotNavigationConfig:
    """Configuration values for the robot navigation environment."""

    max_steps: int = 400
    obstacle_count: int = 4
    obstacle_radius_range: Tuple[float, float] = (0.05, 0.12)
    max_speed: float = 0.15
    control_dt: float = 0.05
    max_angular_velocity: float = math.pi
    success_radius: float = 0.05
    window_size: Tuple[int, int] = (800, 600)
    obstacle_padding: float = 0.005
    obstacle_placement_attempts: int = 40
    obstacle_radius_shrink_factor: float = 0.85
    obstacle_radius_shrink_steps: int = 5
    layout_max_retries: int = 5
    obstacle_collision_penalty: float = 5.0
    moving_obstacle_count: int = 10
    moving_speed_range: Tuple[float, float] = (0.08, 0.18)
    moving_direction_change_prob: float = 0.01
    moving_obstacle_radius: Optional[float] = None
    moving_obstacle_speed: Optional[float] = None
    moving_obstacle_x_range: Optional[Tuple[float, float]] = (0.1, 0.9)
    moving_noise_std: float = 0.25
    collect_targets: bool = False
    target_count: int = 10
    target_radius: float = 0.03
    target_capture_epsilon: float = 0.08
    target_reward: float = -20.0
    target_step_penalty: float = 0.001
    heading_penalty_scale: float = 0.2


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
            raise ValueError(
                f"Unsupported render_mode '{render_mode}'. Use None, 'human', or 'rgb_array'."
            )
        self.render_mode = render_mode
        self.config = config or RobotNavigationConfig()
        if self.config.control_dt <= 0.0:
            raise ValueError("RobotNavigationEnv control_dt must be positive.")

        self._rng: np.random.Generator
        self._steps = 0
        self._last_action = np.zeros(2, dtype=np.float32)

        self.robot_position = np.zeros(2, dtype=np.float32)
        self.robot_angle = 0.0
        self.goal_position = np.array([0.0, 0.5], dtype=np.float32)
        self._obstacles = np.zeros((self.config.obstacle_count, 3), dtype=np.float32)
        self._num_obstacles = 0
        self._moving_indices: list[int] = []
        self._moving_index_order: dict[int, int] = {}
        self._moving_velocities = np.zeros((0, 2), dtype=np.float32)
        self._moving_speed_scales = np.zeros(0, dtype=np.float32)
        self._targets_remaining = 0
        self._target_total = 0

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

        obs_dim = 2 + 2 + 2 + 3 * self.config.obstacle_count
        obs_low = np.zeros(obs_dim, dtype=np.float32)
        obs_high = np.ones_like(obs_low)
        obs_low[2:4] = -1.0
        obs_high[2:4] = 1.0
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

        layout_success = False
        if self.config.collect_targets:
            target_obstacles = max(1, int(self.config.target_count))
        else:
            target_obstacles = self.config.obstacle_count
        layout_retries = max(1, self.config.layout_max_retries)

        for _ in range(layout_retries):
            self._obstacles.fill(0.0)
            placed = 0
            blocker_placed = target_obstacles == 0

            if self.config.collect_targets:
                moving_count = target_obstacles
            else:
                moving_count = min(self.config.moving_obstacle_count, target_obstacles)
            if moving_count > 0:
                selection = self._rng.choice(
                    target_obstacles, size=moving_count, replace=False
                )
                selection = np.atleast_1d(selection)
                self._moving_indices = sorted(int(x) for x in selection)
                self._moving_index_order = {
                    idx: order for order, idx in enumerate(self._moving_indices)
                }
            else:
                self._moving_indices = []
                self._moving_index_order = {}
            moving_indices_set = set(self._moving_indices)

            radius_low, radius_high = self.config.obstacle_radius_range
            shrink_factor = self.config.obstacle_radius_shrink_factor
            shrink_steps = max(1, self.config.obstacle_radius_shrink_steps)
            attempts_per_obstacle = self.config.obstacle_placement_attempts

            placement_failed = False
            while placed < target_obstacles:
                success = False
                current_high = radius_high
                is_moving = placed in moving_indices_set
                force_blocker = not blocker_placed and not self.config.collect_targets

                for _ in range(shrink_steps):
                    for _ in range(attempts_per_obstacle):
                        if self.config.collect_targets:
                            radius = float(self.config.target_radius)
                        elif (
                            is_moving and self.config.moving_obstacle_radius is not None
                        ):
                            radius = float(self.config.moving_obstacle_radius)
                        else:
                            radius = float(self._rng.uniform(radius_low, current_high))

                        if force_blocker:
                            candidate = self._sample_goal_blocker_candidate(radius)
                            if candidate is None:
                                continue
                        elif is_moving:
                            order = self._moving_index_order.get(placed, 0)
                            x_low, x_high = self._moving_segment_bounds(order, radius)
                        elif self.config.collect_targets:
                            x_low = radius
                            x_high = 1.0 - radius
                        else:
                            x_low = 0.1 + radius
                            x_high = 0.45
                        if not force_blocker and x_high <= x_low:
                            x_high = x_low + 1e-3

                        if not force_blocker:
                            x = float(self._rng.uniform(x_low, x_high))
                            if self.config.collect_targets:
                                y = float(self._rng.uniform(radius, 1.0 - radius))
                            else:
                                y = float(self._rng.uniform(0.1 + radius, 0.9 - radius))
                            candidate = np.array([x, y, radius], dtype=np.float32)

                        if self._is_valid_obstacle(candidate, placed):
                            self._obstacles[placed] = candidate
                            placed += 1
                            success = True
                            if force_blocker:
                                blocker_placed = True
                            break

                    if success:
                        break
                    current_high = max(radius_low, current_high * shrink_factor)

                if not success:
                    placement_failed = True
                    break

            if placement_failed:
                continue

            self._num_obstacles = target_obstacles
            self._initialize_moving_obstacles()
            self._targets_remaining = int(
                np.count_nonzero(self._obstacles[: self._num_obstacles, 2] > 0)
            )
            self._target_total = (
                self._targets_remaining if self.config.collect_targets else 0
            )
            if not self._ensure_path_blocking_obstacle():
                placement_failed = True

            if not placement_failed:
                layout_success = True
                break

        if not layout_success:
            if self.config.collect_targets:
                self._place_targets_uniform(target_obstacles)
            else:
                raise RuntimeError(
                    "Failed to place non-overlapping obstacles; consider reducing obstacle_count or padding."
                )

        if not self.config.collect_targets:
            self._targets_remaining = 0
            self._target_total = 0

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

        speed = np.clip(float(action[0]), 0.0, self.config.max_speed)
        angular_velocity = np.clip(
            float(action[1]),
            -self.config.max_angular_velocity,
            self.config.max_angular_velocity,
        )

        delta = np.array(
            [math.cos(self.robot_angle), math.sin(self.robot_angle)],
            dtype=np.float32,
        )
        self.robot_position = np.clip(
            self.robot_position + delta * (speed * self.config.control_dt),
            0.0,
            1.0,
        )
        self.robot_angle = self._wrap_angle(
            self.robot_angle + angular_velocity * self.config.control_dt
        )

        if self.config.collect_targets:
            distance = self._distance_to_nearest_target()
        else:
            distance = self._distance_to_goal()
        distance_to_goal = self._distance_to_goal()
        heading_error = abs(self._heading_error())
        if self.config.collect_targets:
            goal_reached = distance_to_goal <= self.config.success_radius
            terminated = goal_reached
        else:
            goal_reached = distance <= self.config.success_radius
            terminated = goal_reached
        truncated = self._steps >= self.config.max_steps

        if self.config.collect_targets:
            in_obstacle = False
        else:
            in_obstacle = self._is_in_obstacle(self.robot_position)

        capture_reward = 0.0
        captured = 0
        if self.config.collect_targets:
            captured = self._handle_target_captures()
            capture_reward = captured * self.config.target_reward

        if self.config.collect_targets:
            angle_penalty = self.config.heading_penalty_scale * (
                heading_error / math.pi
            )
            reward = -distance - angle_penalty
            reward += capture_reward
        else:
            angle_penalty = self.config.heading_penalty_scale * (
                heading_error / math.pi
            )
            reward = -distance - angle_penalty
            if in_obstacle:
                reward -= self.config.obstacle_collision_penalty
            if goal_reached:
                reward += 1.0

        observation = self._get_observation()
        info = {
            "distance_to_goal": distance_to_goal,
            "num_obstacles": self._num_obstacles,
            "last_action": self._last_action.copy(),
            "robot_angle": self.robot_angle,
            "heading_error": heading_error,
            "in_obstacle": in_obstacle,
            "goal_reached": goal_reached,
            "moving_obstacles": len(self._moving_indices),
        }
        if self.config.collect_targets:
            info["distance_to_target"] = distance
            info["targets_remaining"] = self._targets_remaining
            info["captures"] = captured if capture_reward > 0 else 0
            info["targets_total"] = self._target_total
            info["targets_captured_total"] = max(
                0, self._target_total - self._targets_remaining
            )

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            raise RuntimeError(
                "Render mode must be 'human' or 'rgb_array' to draw the simulation."
            )

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
        observation[2] = math.cos(self.robot_angle)
        observation[3] = math.sin(self.robot_angle)
        observation[4:6] = self.goal_position
        obstacle_slice = 6 + 3 * self._num_obstacles
        observation[6:obstacle_slice] = self._obstacles[: self._num_obstacles].reshape(
            -1
        )
        return observation

    def _initialize_moving_obstacles(self) -> None:
        count = len(self._moving_indices)
        if count <= 0:
            self._moving_indices = []
            self._moving_index_order = {}
            self._moving_velocities = np.zeros((0, 2), dtype=np.float32)
            self._moving_speed_scales = np.zeros(0, dtype=np.float32)
            return

        self._moving_velocities = np.zeros((count, 2), dtype=np.float32)
        self._moving_speed_scales = np.zeros(count, dtype=np.float32)
        for idx in range(count):
            speed = self._sample_moving_speed()
            self._moving_speed_scales[idx] = speed
            angle = float(self._rng.uniform(0.0, 2 * math.pi))
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            self._moving_velocities[idx] = direction * speed

    def _moving_segment_bounds(self, order: int, radius: float) -> Tuple[float, float]:
        if self.config.moving_obstacle_x_range is None:
            low, high = 0.1, 0.9
        else:
            low, high = self.config.moving_obstacle_x_range

        width = max(high - low, 1e-3)
        count = max(1, len(self._moving_indices))
        segment = width / count
        start = low + segment * order
        end = min(high, start + segment)

        x_low = start + radius
        x_high = end - radius
        if x_high <= x_low:
            midpoint = (start + end) / 2.0
            x_low = max(low + radius, midpoint - 1e-3)
            x_high = min(high - radius, midpoint + 1e-3)
        return x_low, x_high

    def _reflect_on_bounds(
        self, position: np.ndarray, velocity: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        clipped = position.copy()
        for axis in range(2):
            min_bound = radius
            max_bound = 1.0 - radius
            if clipped[axis] < min_bound:
                clipped[axis] = min_bound
                velocity[axis] *= -1.0
            elif clipped[axis] > max_bound:
                clipped[axis] = max_bound
                velocity[axis] *= -1.0
        return clipped, velocity

    def _resolve_static_collisions(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        radius: float,
        obstacle_idx: int,
        moving_set: set[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        for other_idx in range(self._num_obstacles):
            if other_idx == obstacle_idx or other_idx in moving_set:
                continue
            other = self._obstacles[other_idx]
            other_radius = other[2]
            if other_radius <= 0:
                continue
            delta = position - other[:2]
            distance = np.linalg.norm(delta)
            min_dist = radius + other_radius
            if distance < min_dist:
                if distance < 1e-6:
                    normal = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    normal = delta / distance
                penetration = min_dist - distance + 1e-4
                position = position + normal * penetration
                velocity = velocity - 2.0 * np.dot(velocity, normal) * normal
                position, velocity = self._reflect_on_bounds(position, velocity, radius)
        return position, velocity

    def _resolve_moving_collisions(self, new_positions: np.ndarray) -> None:
        count = len(self._moving_indices)
        if count < 2:
            return
        for i in range(count):
            idx_i = self._moving_indices[i]
            if idx_i >= self._num_obstacles:
                continue
            radius_i = self._obstacles[idx_i, 2]
            if radius_i <= 0:
                continue
            for j in range(i + 1, count):
                idx_j = self._moving_indices[j]
                if idx_j >= self._num_obstacles:
                    continue
                radius_j = self._obstacles[idx_j, 2]
                if radius_j <= 0:
                    continue
                delta = new_positions[idx_i] - new_positions[idx_j]
                distance = np.linalg.norm(delta)
                min_dist = radius_i + radius_j
                if distance >= min_dist:
                    continue
                if distance < 1e-6:
                    normal = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    normal = delta / max(distance, 1e-6)
                vi = self._moving_velocities[i]
                vj = self._moving_velocities[j]
                vi_n = np.dot(vi, normal)
                vj_n = np.dot(vj, normal)
                vi = vi - normal * vi_n + normal * vj_n
                vj = vj - normal * vj_n + normal * vi_n
                penetration = min_dist - distance + 1e-4
                correction = normal * (penetration / 2.0)
                new_positions[idx_i] += correction
                new_positions[idx_j] -= correction
                new_positions[idx_i], vi = self._reflect_on_bounds(
                    new_positions[idx_i], vi, radius_i
                )
                new_positions[idx_j], vj = self._reflect_on_bounds(
                    new_positions[idx_j], vj, radius_j
                )
                self._moving_velocities[i] = vi
                self._moving_velocities[j] = vj

    def _sample_goal_blocker_candidate(self, radius: float) -> Optional[np.ndarray]:
        segment = self.goal_position - self.robot_position
        seg_len = float(np.linalg.norm(segment))
        if seg_len < 1e-6:
            return None
        direction = segment / seg_len
        perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)

        for _ in range(64):
            t = float(self._rng.uniform(0.2, 0.8))
            base = self.robot_position + direction * (seg_len * t)
            offset = float(self._rng.uniform(-radius * 2.0, radius * 2.0))
            center = base + perpendicular * offset
            center = np.clip(center, radius, 1.0 - radius)
            candidate = np.array([center[0], center[1], radius], dtype=np.float32)
            return candidate
        return None

    def _obstacle_blocks_path(self, center: np.ndarray, radius: float) -> bool:
        start = self.robot_position
        goal = self.goal_position
        segment = goal - start
        seg_len_sq = float(np.dot(segment, segment))
        if seg_len_sq < 1e-8:
            return False
        t = float(np.dot(center - start, segment) / seg_len_sq)
        if t <= 0.0 or t >= 1.0:
            return False
        closest = start + t * segment
        distance = np.linalg.norm(center - closest)
        return distance <= radius + self.config.obstacle_padding

    def _has_path_blocking_obstacle(self) -> bool:
        for obstacle in self._obstacles[: self._num_obstacles]:
            if obstacle[2] <= 0:
                continue
            if self._obstacle_blocks_path(obstacle[:2], obstacle[2]):
                return True
        return False

    def _ensure_path_blocking_obstacle(self) -> bool:
        if (
            self.config.collect_targets
            or self._num_obstacles <= 0
            or self._has_path_blocking_obstacle()
        ):
            return True
        for idx in range(self._num_obstacles):
            radius = self._obstacles[idx, 2]
            if radius <= 0:
                continue
            candidate = self._sample_goal_blocker_candidate(radius)
            if candidate is None:
                continue
            if self._is_valid_obstacle(
                candidate, self._num_obstacles, ignore_index=idx
            ):
                self._obstacles[idx] = candidate
                return True
        return False

    def _sample_moving_speed(self) -> float:
        if self.config.moving_obstacle_speed is not None:
            return float(self.config.moving_obstacle_speed)
        speed_low, speed_high = self.config.moving_speed_range
        return float(self._rng.uniform(speed_low, speed_high))

    def _update_moving_obstacles(self) -> None:
        if not self._moving_indices:
            return
        dt = self.config.control_dt
        moving_set = set(self._moving_indices)
        new_positions = self._obstacles[:, 0:2].copy()

        for vel_idx, obstacle_idx in enumerate(self._moving_indices):
            if obstacle_idx >= self._num_obstacles:
                continue

            radius = self._obstacles[obstacle_idx, 2]
            if radius <= 0:
                continue

            speed_scale = (
                self._moving_speed_scales[vel_idx]
                if vel_idx < len(self._moving_speed_scales)
                else self._sample_moving_speed()
            )

            if vel_idx < len(self._moving_velocities):
                velocity = self._moving_velocities[vel_idx]
            else:
                velocity = np.zeros(2, dtype=np.float32)

            if not np.any(velocity):
                angle = float(self._rng.uniform(0.0, 2 * math.pi))
                direction = np.array(
                    [math.cos(angle), math.sin(angle)], dtype=np.float32
                )
                velocity = direction * speed_scale

            noise_std = float(self.config.moving_noise_std) * speed_scale
            if noise_std > 0:
                noise = self._rng.normal(0.0, noise_std, size=2).astype(np.float32)
                velocity = velocity + noise

            speed = float(np.linalg.norm(velocity))
            if speed > speed_scale and speed > 1e-6:
                velocity = velocity / speed * speed_scale
            elif speed < 1e-6:
                angle = float(self._rng.uniform(0.0, 2 * math.pi))
                direction = np.array(
                    [math.cos(angle), math.sin(angle)], dtype=np.float32
                )
                velocity = direction * speed_scale
            position = new_positions[obstacle_idx]
            displacement = velocity * dt
            candidate_pos = position + displacement

            candidate_pos, velocity = self._reflect_on_bounds(
                candidate_pos, velocity, radius
            )
            candidate_pos, velocity = self._resolve_static_collisions(
                candidate_pos,
                velocity,
                radius,
                obstacle_idx,
                moving_set,
            )

            new_positions[obstacle_idx] = candidate_pos
            self._moving_velocities[vel_idx] = velocity

        self._resolve_moving_collisions(new_positions)

        for vel_idx, obstacle_idx in enumerate(self._moving_indices):
            if obstacle_idx >= self._num_obstacles:
                continue
            self._obstacles[obstacle_idx, 0:2] = new_positions[obstacle_idx]

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal_position - self.robot_position))

    def _nearest_target_vector(self) -> Optional[np.ndarray]:
        best_vec = None
        best_dist = float("inf")
        for obstacle in self._obstacles[: self._num_obstacles]:
            radius = obstacle[2]
            if radius <= 0:
                continue
            vec = obstacle[:2] - self.robot_position
            dist = np.linalg.norm(vec)
            if dist < best_dist:
                best_dist = dist
                best_vec = vec
        return best_vec

    def _distance_to_nearest_target(self) -> float:
        if not self.config.collect_targets:
            return self._distance_to_goal()
        vec = self._nearest_target_vector()
        if vec is None:
            return 0.0
        return float(np.linalg.norm(vec))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi) for stable orientation error calculations."""
        return float((angle + math.pi) % (2 * math.pi) - math.pi)

    def _heading_error(self) -> float:
        """Return signed angle between the robot heading and the goal direction."""
        if self.config.collect_targets:
            vec = self._nearest_target_vector()
            if vec is None or float(np.dot(vec, vec)) < 1e-12:
                return 0.0
            desired_angle = math.atan2(float(vec[1]), float(vec[0]))
        else:
            delta = self.goal_position - self.robot_position
            if float(np.dot(delta, delta)) < 1e-12:
                return 0.0
            desired_angle = math.atan2(float(delta[1]), float(delta[0]))
        return self._wrap_angle(desired_angle - self.robot_angle)

    def _is_valid_obstacle(
        self,
        candidate: np.ndarray,
        count: int,
        *,
        ignore_index: Optional[int] = None,
    ) -> bool:
        padding = self.config.obstacle_padding
        center = candidate[:2]
        radius = float(candidate[2])

        if np.any(center - radius < 0.0) or np.any(center + radius > 1.0):
            return False

        if np.linalg.norm(center - self.robot_position) < radius + padding:
            return False

        for idx, other in enumerate(self._obstacles[:count]):
            if idx == ignore_index:
                continue
            if not other.any():
                continue
            distance = np.linalg.norm(center - other[:2])
            if distance < radius + other[2] + padding:
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

    def _handle_target_captures(self) -> int:
        if not self.config.collect_targets or self._num_obstacles <= 0:
            return 0
        captured = 0
        eps = float(self.config.target_capture_epsilon)
        for idx in range(self._num_obstacles):
            radius = self._obstacles[idx, 2]
            if radius <= 0:
                continue
            center = self._obstacles[idx, 0:2]
            threshold = max(radius, eps)
            if np.linalg.norm(self.robot_position - center) <= threshold:
                self._obstacles[idx, 2] = 0.0
                self._obstacles[idx, 0:2] = center
                captured += 1
                order = self._moving_index_order.get(idx)
                if order is not None and order < len(self._moving_velocities):
                    self._moving_velocities[order] = 0.0
        if captured > 0:
            self._targets_remaining = max(0, self._targets_remaining - captured)
        return captured

    def _place_targets_uniform(self, count: int) -> None:
        radius = float(self.config.target_radius)
        centers = self._rng.uniform(radius, 1.0 - radius, size=(count, 2))
        self._obstacles[:count, 0:2] = centers
        self._obstacles[:count, 2] = radius
        self._num_obstacles = count
        self._moving_indices = list(range(count))
        self._moving_index_order = {idx: idx for idx in range(count)}
        self._initialize_moving_obstacles()
        self._targets_remaining = count
        self._target_total = count

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
        pygame.draw.rect(
            self._surface, border_color, pygame.Rect(0, 0, width, height), width=4
        )

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
        pulse_radius = body_radius + int(
            2 + 3 * math.sin(self._steps * 0.3 + speed_ratio * math.pi)
        )
        pulse_radius = max(pulse_radius, body_radius + 1)

        # Draw shimmering aura
        pygame.draw.circle(
            self._surface, (180, 200, 255), center, pulse_radius, width=2
        )

        # Base body
        pygame.draw.circle(self._surface, (60, 120, 240), center, body_radius)
        pygame.draw.circle(self._surface, (20, 60, 160), center, body_radius, width=2)
        highlight = (
            center[0] + int(body_radius * 0.4),
            center[1] - int(body_radius * 0.4),
        )
        pygame.draw.circle(
            self._surface, (220, 240, 255), highlight, max(1, body_radius // 3)
        )

        heading_world = np.array(
            [math.cos(self.robot_angle), math.sin(self.robot_angle)],
            dtype=np.float32,
        )
        heading_screen = np.array(
            [heading_world[0], -heading_world[1]], dtype=np.float32
        )
        perp_screen = np.array(
            [-heading_screen[1], heading_screen[0]], dtype=np.float32
        )

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

        pygame.draw.polygon(
            self._surface, (250, 250, 255), [nose, left_wing, right_wing]
        )
        pygame.draw.polygon(
            self._surface, (40, 80, 200), [nose, left_wing, right_wing], width=1
        )

        # Thruster animation
        thruster_wave = (
            0.7 + 0.3 * (math.sin(self._steps * 0.5 + speed_ratio * 2.0) + 1) / 2
        )
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
        pygame.draw.line(
            self._surface, flame_color, thruster_base, thruster_end, width=3
        )


class SimpleGoalController:
    """Greedy controller that moves toward the goal with speed/angular-velocity actions."""

    def __init__(
        self,
        *,
        max_speed: float,
        turn_gain: float = 1.0,
        max_turn_rate: float = math.pi,
    ) -> None:
        self.max_speed = float(max_speed)
        self.turn_gain = float(turn_gain)
        self.max_turn_rate = float(max_turn_rate)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + math.pi) % (2 * math.pi) - math.pi)

    def act(self, observation: np.ndarray) -> np.ndarray:
        robot = observation[0:2]
        heading_cos = float(observation[2])
        heading_sin = float(observation[3])
        current_angle = math.atan2(heading_sin, heading_cos)
        goal = observation[4:6]
        delta = goal - robot

        distance = np.linalg.norm(delta)
        if distance < 1e-8:
            return np.array([0.0, 0.0], dtype=np.float32)

        desired_angle = float(math.atan2(delta[1], delta[0]))
        angle_error = self._wrap_angle(desired_angle - current_angle)
        angular_velocity = np.clip(
            self.turn_gain * angle_error,
            -self.max_turn_rate,
            self.max_turn_rate,
        )

        # Saturate speed so the robot slows as it nears the goal
        if self.max_speed <= 0.0:
            speed_ratio = 0.0
        else:
            speed_ratio = min(1.0, distance / (self.max_speed * 8.0))
        speed_ratio = float(np.clip(speed_ratio, 0.0, 1.0))
        return np.array([speed_ratio, angular_velocity], dtype=np.float32)


class RobotNavigationMetricsCollector(MetricsCollector):
    def __init__(self, rolling_window_size: int = 20):
        super().__init__()
        self.rolling_window_size = rolling_window_size

    def collect_metrics_from_final_episode_info(self, info: dict, step: int) -> dict:
        super().collect_metrics_from_final_episode_info(info, step)
        self.append_metric(
            "episode_stats/distance_to_goal",
            info.get("distance_to_goal", 0.0),
            step=step,
        )
        self.rolling_window["distance_to_goal"].append(
            info.get("distance_to_goal", 0.0)
        )
        self.append_metric(
            f"episode_stats/distance_to_goal_rolling_{self.rolling_window_size}",
            np.mean(self.rolling_window["distance_to_goal"]),
            step=step,
        )
        self.append_metric(
            "episode_stats/in_obstacle",
            float(info.get("in_obstacle", False)),
            step=step,
        )
        self.rolling_window["in_obstacle"].append(float(info.get("in_obstacle", False)))
        self.append_metric(
            f"episode_stats/in_obstacle_rolling_{self.rolling_window_size}",
            np.mean(self.rolling_window["in_obstacle"]),
            step=step,
        )
        if "targets_captured_total" in info:
            captures = float(info.get("targets_captured_total", 0.0))
            self.append_metric("episode_stats/targets_captured", captures, step=step)
            self.rolling_window["targets_captured"].append(captures)
            self.append_metric(
                f"episode_stats/targets_captured_rolling_{self.rolling_window_size}",
                np.mean(self.rolling_window["targets_captured"]),
                step=step,
            )
        self.append_metric(
            "episode_stats/goal_reached",
            float(info.get("goal_reached", False)),
            step=step,
        )
        self.rolling_window["goal_reached"].append(float(info.get("goal_reached", False)))
        self.append_metric(
            f"episode_stats/goal_reached_rolling_{self.rolling_window_size}",
            np.mean(self.rolling_window["goal_reached"]),
            step=step,
        )
