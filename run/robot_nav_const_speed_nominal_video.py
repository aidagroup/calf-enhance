#!/usr/bin/env python3
"""Roll out nominal policy in RobotNavigationConstSpeedCatch-v0 and save video."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import src  # noqa: F401 - registers gym environments
from src import RUN_PATH
from src.envs.robot_navigation_const_speed import ConstSpeedGoalController


def _ensure_headless_pygame() -> None:
    if os.environ.get("DISPLAY"):
        return
    if os.environ.get("SDL_VIDEODRIVER"):
        return
    os.environ["SDL_VIDEODRIVER"] = "dummy"


def _load_first_obs(json_path: Path) -> np.ndarray:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected non-empty episode list in {json_path}.")

    first = data[0]
    if "obs" not in first:
        raise KeyError(f"Missing 'obs' in first step of {json_path}.")

    obs = np.asarray(first["obs"], dtype=np.float32)
    if obs.ndim == 2:
        obs = obs[0]
    obs = obs.reshape(-1)
    if obs.size < 6:
        raise ValueError(f"Expected obs size >= 6, got {obs.size}.")
    if (obs.size - 6) % 3 != 0:
        raise ValueError(f"Expected obs size 6 + 3*k, got {obs.size}.")
    return obs


def _set_env_state_from_obs(env: gym.Env, obs: np.ndarray) -> None:
    base_env = env.unwrapped
    base_env.robot_position = obs[0:2].astype(np.float32).copy()
    base_env.robot_angle = float(math.atan2(float(obs[3]), float(obs[2])))
    base_env.goal_position = obs[4:6].astype(np.float32).copy()

    obstacles = obs[6:].reshape(-1, 3).astype(np.float32)
    if obstacles.shape[0] > base_env._obstacles.shape[0]:
        raise ValueError(
            f"Obs has {obstacles.shape[0]} obstacles but env supports only "
            f"{base_env._obstacles.shape[0]}."
        )
    base_env._obstacles.fill(0.0)
    if obstacles.size:
        base_env._obstacles[: obstacles.shape[0]] = obstacles
    base_env._num_obstacles = int(obstacles.shape[0])

    base_env._steps = 0
    base_env._last_action = np.zeros(2, dtype=np.float32)
    base_env.trajectory = [base_env.robot_position.copy()]
    base_env.captured_points = []
    if hasattr(base_env, "_target_total"):
        base_env._target_total = int(np.count_nonzero(obstacles[:, 2] > 0.0))
    if hasattr(base_env, "_targets_remaining"):
        base_env._targets_remaining = int(np.count_nonzero(obstacles[:, 2] > 0.0))


def _init_video_writer(output_file: Path, width: int, height: int, fps: int) -> cv2.VideoWriter:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create VideoWriter for {output_file}.")
    return writer


def rollout_nominal(
    *,
    env_id: str,
    init_json: Path,
    output_file: Path,
    fps: int,
    seed: int | None,
    max_steps: int | None,
) -> None:
    _ensure_headless_pygame()

    env = gym.make(env_id, render_mode="rgb_array")
    controller = ConstSpeedGoalController()

    init_obs = _load_first_obs(init_json)
    obs, _ = env.reset(seed=seed)
    _set_env_state_from_obs(env, init_obs)
    obs = env.unwrapped._get_observation()

    env.unwrapped.show_trajectory = True
    env.unwrapped.show_captured_points = True

    first_frame = env.render()
    if first_frame is None:
        raise RuntimeError("Render returned None on first frame.")
    height, width = first_frame.shape[:2]
    writer = _init_video_writer(output_file, width, height, fps)

    frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
    writer.write(frame_bgr)
    frame_count = 1

    terminated = False
    truncated = False
    step = 0
    total_reward = 0.0

    while not (terminated or truncated):
        if max_steps is not None and step >= max_steps:
            break
        action = controller.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step += 1

        frame = env.render()
        if frame is None:
            continue
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
        frame_count += 1

    writer.release()
    env.close()

    print(f"Saved: {output_file}")
    print(
        "Episode stats: "
        f"steps={step}, frames={frame_count}, total_reward={total_reward:.3f}, "
        f"goal_reached={bool(info.get('goal_reached', False))}, "
        f"targets_captured_total={int(info.get('targets_captured_total', 0))}, "
        f"distance_to_goal={float(info.get('distance_to_goal', float('nan'))):.4f}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run nominal const-speed policy from init-state in JSON and save a video."
        )
    )
    parser.add_argument(
        "--env-id",
        default="RobotNavigationConstSpeedCatch-v0",
        help="Gym env id to run.",
    )
    parser.add_argument(
        "--init-json",
        type=Path,
        default=Path(RUN_PATH) / "expdata" / "robot.json",
        help="Path to trajectory JSON; first obs is used as init-state.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(RUN_PATH) / "expdata" / "anim_nominal_const_speed.mp4",
        help="Output MP4 path.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for env reset before state override.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional hard limit for rollout steps.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rollout_nominal(
        env_id=args.env_id,
        init_json=args.init_json,
        output_file=args.output_file,
        fps=args.fps,
        seed=args.seed,
        max_steps=args.max_steps,
    )
