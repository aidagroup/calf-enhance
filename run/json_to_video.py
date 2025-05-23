import json
import numpy as np
import cv2
from src.envs.underwaterdrone import UnderwaterDroneEnv
from src import RUN_PATH


def json_to_video(json_file, output_file, fps=50):
    # Load JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    # Initialize environment
    env = UnderwaterDroneEnv(render_mode="rgb_array")
    env.reset()
    env.show_axes = True

    # Get video dimensions from first frame
    first_frame = env.render()
    if first_frame is None:
        print("Error: First frame is None. Check environment rendering.")
        return
    height, width = first_frame.shape[:2]
    print(f"Video dimensions: {width}x{height}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    frame_count = 0
    # Process each step
    for i, step in enumerate(data):
        obs = np.array(step["obs"][0])

        # Extract state components from observation
        x = obs[0]
        y = obs[1]
        cos_theta = obs[2]
        sin_theta = obs[3]
        v_x = obs[4]
        v_y = obs[5]
        omega = obs[6]

        # Calculate theta from cos and sin
        theta = np.arctan2(sin_theta, cos_theta)

        # Set drone state directly
        env.drone.x = x
        env.drone.y = y
        env.drone.theta = theta
        env.drone.v_x = v_x
        env.drone.v_y = v_y
        env.drone.omega = omega
        env.last_action = np.array(step["actions"][0])

        env.trajectory.append(np.array([x, y]))

        # Render frame
        frame = env.render()
        if frame is None:
            print(f"Warning: Frame {i} is None. Skipping.")
            continue

        # Convert RGB to BGR (OpenCV format)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Write frame
        video.write(frame)
        frame_count += 1
        if i % 100 == 0:
            print(f"Processed {i+1} frames...")

    # Release resources
    video.release()
    env.close()
    print(f"Done. Wrote {frame_count} frames to {output_file}")


if __name__ == "__main__":
    # Example usage
    json_file = f"/home/user/Repos/calfq-td3/run/mlruns/283352061287474666/104d90fa54b14952908033257381781d/artifacts/trajectories/0002993999.json"
    output_file = f"{RUN_PATH}/" + json_file.split("/")[-1].split(".")[0] + ".mp4"
    print(f"Input: {json_file}\nOutput: {output_file}")
    json_to_video(json_file, output_file)
