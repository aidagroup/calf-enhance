#!/usr/bin/env python3
import numpy as np
import time
import gymnasium as gym
import os
from src import RUN_PATH
from pathlib import Path
from src.envs.underwaterdrone import (
    OFFSET_LAT,
    DRONE_INERTIA,
    TOP_Y,
    DRONE_MASS,
    GRAVITY,
    DRAG_COEFF,
    HOLE_WIDTH,
    MAX_F_LONG,
    MAX_F_LAT,
)


def make_env(env_id, seed, capture_video=True, run_name="underwater_drone_demo"):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            videos_dir = Path(RUN_PATH) / "videos" / run_name
            videos_dir.mkdir(parents=True, exist_ok=True)
            env = gym.wrappers.RecordVideo(env, f"{videos_dir}")
        else:
            env = gym.make(env_id, render_mode="human")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class Controller:
    def __init__(
        self,
        kp_theta: float = 20.0,
        kd_theta: float = 6.0,
        kvx: float = 4.0,
        kvy: float = 2.0,
        mass: float = 1.0,
        gravity: float = 1,
        offset: float = 0.2,
        max_f_long: float = 3.0,
        max_f_lat: float = 3.0,
        dt: float = 0.02,
    ) -> None:
        self.kp_theta = kp_theta
        self.kd_theta = kd_theta
        self.kvx = kvx
        self.kvy = kvy
        self.mass = mass
        self.gravity = gravity
        self.offset = offset
        self.max_f_long = max_f_long
        self.max_f_lat = max_f_lat
        self.dt = dt

    def reset(self) -> None:
        pass

    def get_action_lyapunov(self, obs):
        x, y, cos_theta, sin_theta, v_x, v_y, omega = obs
        alpha = 0.5
        theta = np.arctan2(sin_theta, cos_theta)

        norm_r = np.sqrt(x**2 + (TOP_Y - y) ** 2)
        norm_v = np.sqrt(v_x**2 + v_y**2)
        n_x = -x / norm_r
        n_y = (TOP_Y - y) / norm_r
        norm_r_dot = (x * v_x - (TOP_Y - y) * v_y) / norm_r
        n_x_dot = (-v_x - n_x * norm_r_dot) / norm_r
        n_y_dot = (-v_y - n_y * norm_r_dot) / norm_r

        v_x_diff = v_x - alpha * n_x
        v_y_diff = v_y - alpha * n_y
        a_long = v_x_diff * cos_theta + v_y_diff * sin_theta
        a_lat = -v_x_diff * sin_theta + v_y_diff * cos_theta + OFFSET_LAT * omega

        neg_b = (
            DRAG_COEFF * norm_v * (v_x_diff * cos_theta + v_y_diff * sin_theta)
            - DRONE_MASS * GRAVITY * v_y_diff
            - alpha * DRONE_MASS * (v_x_diff * n_x_dot - v_y_diff * n_y_dot)
        )

        target_luapunov = 10 * (v_x_diff**2 + v_y_diff**2) + 10 * omega**2

        eps = 1e-6
        F_long = a_long * (neg_b - target_luapunov) / (eps + a_long**2 + a_lat**2)
        F_lat = a_lat * (neg_b - target_luapunov) / (eps + a_long**2 + a_lat**2)

        return np.array([F_long, F_lat], dtype=np.float32)

    # ---------------------------------------------------------------------
    def get_action(self, state):
        self.kp_y = 2.0
        self.kd_y = 1.2
        self.kp_x = 1.5
        self.kd_x = 0.8
        self.target_y = TOP_Y
        self.hole_half = 0.5 * HOLE_WIDTH

        x, y, cos_theta, sin_theta, v_x, v_y, _ = state

        y_err = self.target_y - y
        Fy = GRAVITY * DRONE_MASS + self.kp_y * y_err - self.kd_y * v_y  # PD‑Anteil

        x_ref = 0.0
        x_err = x_ref - x
        Fx = self.kp_x * x_err - self.kd_x * v_x

        F_long = cos_theta * Fx + sin_theta * Fy
        F_lat = -sin_theta * Fx + cos_theta * Fy

        F_long = np.clip(F_long, -MAX_F_LONG, MAX_F_LONG)
        F_lat = np.clip(F_lat, -MAX_F_LAT, MAX_F_LAT)

        return np.array([F_long, F_lat], dtype=np.float32)


def main():
    # Create the environment with video recording
    seed = None
    env_id = "UnderwaterDrone-v0"
    capture_video = True
    env_fn = make_env(env_id, seed, capture_video)
    env = env_fn()

    # Reset the environment
    observation, info = env.reset(seed=seed)

    controller = Controller()

    # Run for 1000 steps or until terminated
    total_reward = 0
    for step in range(1000):
        # Get action from controller
        action = controller.get_action(observation)

        # Apply action
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Check if episode is done
        if terminated or truncated:
            x, y, cos_theta, sin_theta, v_x, v_y, omega = observation
            theta = np.arctan2(sin_theta, cos_theta)
            print(f"\nEpisode finished after {step+1} steps")
            print(f"Final position: x={info['x']:.2f}, y={info['y']:.2f}")
            print(f"Final velocity: v_x={v_x:.2f}, v_y={v_y:.2f}")
            print(
                f"Final orientation: theta={theta:.2f} (target={np.pi/2:.2f}), omega={omega:.2f}"
            )
            print(f"Total reward: {total_reward:.2f}")
            break

    # Clean up
    env.close()
    print(f"Video saved to {RUN_PATH}/videos/underwater_drone_demo")


if __name__ == "__main__":
    main()
