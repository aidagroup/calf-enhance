from pathlib import Path
import gymnasium as gym

REPO_PATH = Path(__file__).parent.parent
SRC_PATH = REPO_PATH / "src"
RUN_PATH = REPO_PATH / "run"

gym.register(
    id="UnderwaterDrone-v0",
    entry_point="src.envs:UnderwaterDroneEnv",
    max_episode_steps=1500,
)
