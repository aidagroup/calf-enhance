import argparse
import json
import os

import cv2
import numpy as np

from src import RUN_PATH


def _load_episode(json_file: str) -> list[dict]:
    with open(json_file, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(
            f"Expected JSON episode to be a list, got {type(data).__name__}."
        )
    return data


def _ensure_headless_pygame() -> None:
    if os.environ.get("DISPLAY"):
        return
    if os.environ.get("SDL_VIDEODRIVER"):
        return
    os.environ["SDL_VIDEODRIVER"] = "dummy"


def _init_video_writer(output_file: str, width: int, height: int, fps: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    if not video.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for '{output_file}'.")
    return video


def _infer_robot_obstacle_count(obs: np.ndarray) -> int:
    obs = np.asarray(obs).reshape(-1)
    if obs.size < 6:
        return 0
    tail = obs.size - 6
    if tail % 3 != 0:
        raise ValueError(f"RobotNavigation obs has {obs.size} dims; expected 6 + 3*k.")
    return int(tail // 3)


def json_to_video_underwaterdrone(json_file: str, output_file: str, fps: int = 50) -> None:
    data = _load_episode(json_file)

    from src.envs.underwaterdrone import UnderwaterDroneEnv

    env = UnderwaterDroneEnv(render_mode="rgb_array")
    env.reset()
    env.show_axes = True

    first_frame = env.render()
    if first_frame is None:
        raise RuntimeError("First frame is None. Check UnderwaterDrone rendering.")
    height, width = first_frame.shape[:2]
    print(f"Video dimensions: {width}x{height}")

    video = _init_video_writer(output_file, width, height, fps)
    frame_count = 0

    for i, step in enumerate(data):
        obs = np.asarray(step["obs"])
        if obs.ndim == 2:
            obs = obs[0]
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        x = float(obs[0])
        y = float(obs[1])
        cos_theta = float(obs[2])
        sin_theta = float(obs[3])
        v_x = float(obs[4])
        v_y = float(obs[5])
        omega = float(obs[6])
        theta = float(np.arctan2(sin_theta, cos_theta))

        env.drone.x = x
        env.drone.y = y
        env.drone.theta = theta
        env.drone.v_x = v_x
        env.drone.v_y = v_y
        env.drone.omega = omega
        env.last_action = np.asarray(step["actions"], dtype=np.float32).reshape(-1)
        env.trajectory.append(np.array([x, y], dtype=np.float32))

        frame = env.render()
        if frame is None:
            print(f"Warning: Frame {i} is None. Skipping.")
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)
        frame_count += 1
        if i % 100 == 0:
            print(f"Processed {i + 1} frames...")

    video.release()
    env.close()
    print(f"Done. Wrote {frame_count} frames to {output_file}")


def json_to_video_robot(json_file: str, output_file: str, fps: int = 30) -> None:
    _ensure_headless_pygame()
    from src.envs.robot_navigation import RobotNavigationConfig, RobotNavigationEnv

    data = _load_episode(json_file)
    if not data:
        raise ValueError("Episode JSON is empty.")

    first_obs = np.asarray(data[0]["obs"])
    if first_obs.ndim == 2:
        first_obs = first_obs[0]
    first_obs = np.asarray(first_obs, dtype=np.float32).reshape(-1)
    obstacle_count = _infer_robot_obstacle_count(first_obs)

    config = RobotNavigationConfig(obstacle_count=obstacle_count)
    env = RobotNavigationEnv(render_mode="rgb_array", config=config)
    env.reset()
    env.show_trajectory = True
    env.show_captured_points = True

    trajectory: list[np.ndarray] = []
    captured_points: list[np.ndarray] = []
    prev_obstacles: np.ndarray | None = None

    first_frame = env.render()
    if first_frame is None:
        raise RuntimeError("First frame is None. Check RobotNavigation rendering.")
    height, width = first_frame.shape[:2]
    print(f"Video dimensions: {width}x{height}")

    video = _init_video_writer(output_file, width, height, fps)
    frame_count = 0

    for i, step in enumerate(data):
        obs = np.asarray(step["obs"])
        if obs.ndim == 2:
            obs = obs[0]
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        x, y = float(obs[0]), float(obs[1])
        cos_theta, sin_theta = float(obs[2]), float(obs[3])
        theta = float(np.arctan2(sin_theta, cos_theta))
        goal = obs[4:6].astype(np.float32)

        obstacle_flat = obs[6:]
        obstacles = (
            obstacle_flat.reshape(-1, 3).astype(np.float32)
            if obstacle_flat.size
            else np.zeros((0, 3), dtype=np.float32)
        )

        action = np.asarray(step.get("actions", []), dtype=np.float32)
        if action.ndim == 2:
            action = action[0]
        action = action.reshape(-1)
        if action.size == 0:
            last_action = np.zeros(2, dtype=np.float32)
        elif action.size == 1:
            last_action = np.array(
                [config.max_speed, float(action[0])], dtype=np.float32
            )
        else:
            last_action = np.array(
                [float(action[0]), float(action[1])], dtype=np.float32
            )

        env._steps = i
        env.robot_position = np.array([x, y], dtype=np.float32)
        env.robot_angle = theta
        env.goal_position = goal
        env._last_action = last_action

        trajectory.append(env.robot_position.copy())
        if prev_obstacles is not None and prev_obstacles.shape == obstacles.shape:
            prev_r = prev_obstacles[:, 2]
            cur_r = obstacles[:, 2]
            dropped = np.where((prev_r > 0.0) & (cur_r <= 0.0))[0]
            for idx in dropped.tolist():
                captured_points.append(obstacles[idx, 0:2].copy())
        prev_obstacles = obstacles.copy()
        env.trajectory = trajectory
        env.captured_points = captured_points

        if obstacles.shape[0] > env._obstacles.shape[0]:
            raise ValueError(
                f"Episode has {obstacles.shape[0]} obstacles but env has obstacle_count={env._obstacles.shape[0]}."
            )
        env._obstacles.fill(0.0)
        if obstacles.size:
            env._obstacles[: obstacles.shape[0]] = obstacles
        env._num_obstacles = int(obstacles.shape[0])

        frame = env.render()
        if frame is None:
            print(f"Warning: Frame {i} is None. Skipping.")
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)
        frame_count += 1
        if i % 100 == 0:
            print(f"Processed {i + 1} frames...")

    video.release()
    env.close()
    print(f"Done. Wrote {frame_count} frames to {output_file}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a saved episode JSON to an MP4 video."
    )
    parser.add_argument(
        "--env",
        choices=["underwaterdrone", "robot"],
        default="underwaterdrone",
        help="Which environment renderer to use.",
    )
    parser.add_argument("--json-file", required=True, help="Path to episode JSON file.")
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output mp4 path. Defaults to RUN_PATH/<json_basename>.mp4",
    )
    parser.add_argument("--fps", type=int, default=None, help="Video FPS.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.output_file is None:
        base = os.path.basename(args.json_file).split(".")[0]
        args.output_file = f"{RUN_PATH}/{base}.mp4"

    if args.env == "underwaterdrone":
        fps = 50 if args.fps is None else int(args.fps)
        json_to_video_underwaterdrone(args.json_file, args.output_file, fps=fps)
    else:
        fps = 30 if args.fps is None else int(args.fps)
        json_to_video_robot(args.json_file, args.output_file, fps=fps)
