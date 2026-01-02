"""Demo rollout for RobotNavigationConstSpeedCatch-v0 with ConstSpeedGoalController."""

import gymnasium as gym
import src  # noqa: F401 - registers environments
from src.envs.robot_navigation_const_speed import ConstSpeedGoalController


def main(num_episodes: int = 5) -> None:
    env = gym.make("RobotNavigationConstSpeedCatch-v0", render_mode="human")
    controller = ConstSpeedGoalController()

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        step = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action = controller.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

        captures = info.get("targets_captured_total", 0)
        goal_reached = info.get("goal_reached", False)
        distance = info.get("distance_to_goal", -1)
        print(
            f"Episode {ep + 1}: steps={step}, reward={total_reward:.2f}, "
            f"captures={captures}, goal_reached={goal_reached}, dist={distance:.3f}"
        )

    env.close()


if __name__ == "__main__":
    main()

