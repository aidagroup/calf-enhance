#!/usr/bin/env python3
import numpy as np
import time
import gymnasium as gym
import os
from src import RUN_PATH
from pathlib import Path


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
        kvx: float = 2.0,
        kvy: float = 2.0,
        mass: float = 1.0,
        gravity: float = 0.5,
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

    def get_action(self, obs):
        x, y, cos_theta, sin_theta, v_x, v_y, omega = obs
        theta = np.arctan2(sin_theta, cos_theta)

        tau = -self.kp_theta * (theta - np.pi / 2) - self.kd_theta * omega
        F_lat = np.clip(tau / self.offset, -self.max_f_lat, self.max_f_lat)

        ax_des = -self.kvx * v_x
        ay_des = -self.gravity - self.kvy * v_y

        Fx_lat = -sin_theta * F_lat
        Fy_lat = cos_theta * F_lat

        if abs(sin_theta) > 0.1:
            F_long = (ay_des * self.mass - Fy_lat) / sin_theta
        else:
            F_long = (
                (ax_des * self.mass - Fx_lat) / cos_theta
                if abs(cos_theta) > 0.1
                else 0.0
            )

        F_long = np.clip(F_long, -self.max_f_long, self.max_f_long)

        return np.array([F_long, F_lat], dtype=np.float32)


def main():
    # Create the environment with video recording
    seed = 21
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
