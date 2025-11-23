from pathlib import Path
import gymnasium as gym

from src.envs.robot_navigation import RobotNavigationConfig

REPO_PATH = Path(__file__).parent.parent
SRC_PATH = REPO_PATH / "src"
RUN_PATH = REPO_PATH / "run"

gym.register(
    id="UnderwaterDrone-v0",
    entry_point="src.envs:UnderwaterDroneEnv",
    max_episode_steps=1500,
)

gym.register(
    id="LidarNav-v0",
    entry_point="src.envs:LidarNavEnv",
)

gym.register(
    id="RobotNavigation-v0",
    entry_point="src.envs:RobotNavigationEnv",
    max_episode_steps=300,
)

gym.register(
    id="RobotNavigationEmpty-v0",
    entry_point="src.envs:RobotNavigationEnv",
    kwargs={"config": RobotNavigationConfig(obstacle_count=0)},
    max_episode_steps=300,
)

gym.register(
    id="RobotNavigationSingle-v0",
    entry_point="src.envs:RobotNavigationEnv",
    kwargs={"config": RobotNavigationConfig(obstacle_count=1)},
    max_episode_steps=300,
)
