# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass, field

import gymnasium as gym
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer

from src import RUN_PATH
from src.config import config
from src.envs.robot_dynamics import RobotDynamicsMetricsCollector
from src.envs.robot_navigation import RobotNavigationMetricsCollector
from src.envs.underwaterdrone import UnderwaterDroneMetricsCollector
from src.controller import (
    RobotDynamicsGoalController,
    RobotNavigationConstSpeedGoalController,
    RobotNavigationGoalController,
    UnderwaterDroneNominalController,
)
from src.utils.metrics_controller import MetricsCollector
from src.utils.mlflow import MlflowConfig, log_json_artifact, mlflow_monitoring
from src.utils.robot_nav_logging import log_robot_nav_trajectory


@dataclass
class Args:
    mlflow: MlflowConfig = field(
        default_factory=lambda: MlflowConfig(
            tracking_uri=config.MLFLOW_TRACKING_URI,
            experiment_name=os.path.basename(__file__)[: -len(".py")],
        )
    )
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = "cuda:0"
    """device to run the experiment on"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    learning_starts: int = 5_000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """entropy regularization coefficient"""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    rolling_average_window: int = 20
    """the rolling average window for the metrics"""

    def __post_init__(self):
        default_experiment_name = os.path.basename(__file__)[: -len(".py")]
        auto_experiment_name = default_experiment_name + "__" + self.env_id
        if (
            not self.mlflow.experiment_name
            or self.mlflow.experiment_name == default_experiment_name
        ):
            self.mlflow.experiment_name = auto_experiment_name
        if not self.mlflow.run_name:
            timestamp = int(time.time())
            if "__" + self.env_id in self.mlflow.experiment_name:
                self.mlflow.run_name = (
                    self.mlflow.experiment_name + "__" + str(self.seed) + "__" + str(timestamp)
                )
            else:
                self.mlflow.run_name = (
                    self.mlflow.experiment_name
                    + "__"
                    + self.env_id
                    + "__"
                    + str(self.seed)
                    + "__"
                    + str(timestamp)
                )


def create_metrics_collector(env_id: str, rolling_window_size: int = 20):
    if env_id.startswith("RobotDynamics"):
        return RobotDynamicsMetricsCollector(rolling_window_size)
    if env_id.startswith("RobotNavigation"):
        return RobotNavigationMetricsCollector(rolling_window_size)
    if env_id.startswith("UnderwaterDrone"):
        return UnderwaterDroneMetricsCollector(rolling_window_size)
    return MetricsCollector(rolling_window_size)


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", seed=seed)
            env = gym.wrappers.RecordVideo(
                env,
                f"{RUN_PATH}/videos/{run_name}",
                episode_trigger=lambda e: e % 5 == 0,
            )
        else:
            env = gym.make(env_id, seed=seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def create_controller(env_id: str):
    if env_id.startswith("UnderwaterDrone"):
        return UnderwaterDroneNominalController()
    if env_id.startswith("RobotDynamics"):
        return RobotDynamicsGoalController()
    if env_id.startswith("RobotNavigationConstSpeed"):
        return RobotNavigationConstSpeedGoalController()
    if env_id.startswith("RobotNavigation"):
        return RobotNavigationGoalController()
    raise ValueError(f"Environment {env_id} not supported")


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


@mlflow_monitoring()
def main(args: Args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    mlflow.set_tag("videos_path", f"{RUN_PATH}/videos/{args.mlflow.run_name}")

    device = torch.device(args.device)
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed + i,
                i,
                args.capture_video,
                args.mlflow.run_name,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    metrics_collector = create_metrics_collector(
        args.env_id, args.rolling_average_window
    )
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
        log_alpha = None
        a_optimizer = None
        target_entropy = None

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    start_time = time.time()
    episode_trajectory = []
    controller = create_controller(args.env_id)
    obs, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions, _, _ = actor.get_action(torch.tensor(obs, dtype=torch.float32, device=device))
                actions = actions.cpu().numpy()

        actions_to_apply = (
            actions + controller.get_action(obs)
        ).clip(envs.single_action_space.low, envs.single_action_space.high)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions_to_apply)
        episode_trajectory.append(
            {"obs": obs.copy(), "actions": actions_to_apply.copy(), "reward": np.array(rewards).copy()}
        )

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is None:
                    continue
                metrics_collector.collect_metrics_from_final_episode_info(info, global_step)
                episode_return = float(np.asarray(info["episode"]["r"]).item())
                print(f"global_step={global_step}, episodic_return={episode_return}")
                metrics_collector.log_pending_metrics(synchronous=True)
                log_json_artifact(
                    episode_trajectory,
                    "trajectories",
                    json_name=f"{global_step:010d}.json",
                )
                if args.env_id.startswith("RobotNavigation"):
                    log_robot_nav_trajectory(
                        episode_trajectory,
                        global_step,
                        total_reward=episode_return,
                        goal_reached=bool(info.get("goal_reached", False)),
                    )
                episode_trajectory = []
                break

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * min_qf_next_target.view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                metrics_collector.append_metric("losses/qf1_values", qf1_a_values.mean().item(), step=global_step)
                metrics_collector.append_metric("losses/qf2_values", qf2_a_values.mean().item(), step=global_step)
                metrics_collector.append_metric("losses/qf1_loss", qf1_loss.item(), step=global_step)
                metrics_collector.append_metric("losses/qf2_loss", qf2_loss.item(), step=global_step)
                metrics_collector.append_metric("losses/qf_loss", qf_loss.item() / 2.0, step=global_step)
                metrics_collector.append_metric("losses/actor_loss", actor_loss.item(), step=global_step)
                metrics_collector.append_metric("losses/alpha", alpha, step=global_step)
                if args.autotune:
                    metrics_collector.append_metric("losses/alpha_loss", alpha_loss.item(), step=global_step)
                metrics_collector.append_metric(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    step=global_step,
                )
                print("SPS:", int(global_step / (time.time() - start_time)))
                metrics_collector.log_pending_metrics(synchronous=True)

    envs.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
