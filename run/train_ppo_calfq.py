import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from scipy.optimize import root_scalar
from torch.distributions.normal import Normal

from src import RUN_PATH
from src.config import config
from src.controller import (
    UnderwaterDroneNominalController,
    RobotDynamicsGoalController,
    RobotNavigationConstSpeedGoalController,
    RobotNavigationGoalController,
)
from src.envs.robot_dynamics import RobotDynamicsMetricsCollector
from src.envs.robot_navigation import RobotNavigationMetricsCollector
from src.envs.underwaterdrone import UnderwaterDroneMetricsCollector
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
    torch_deterministic: bool = True
    device: str = "cuda:0"
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: Optional[str] = None
    capture_video: bool = False

    env_id: str = "UnderwaterDrone-v0"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    rolling_average_window: int = 20

    calfq_critic_improvement_threshold: float = 0.01
    calfq_p_relax_init: float = 0.8
    calfq_p_relax_decay: float = 0.995
    calfq_anneal: bool = True
    calfq_anneal_frac: float = 0.9
    calfq_selective_buffer: bool = True

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

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
                    self.mlflow.experiment_name
                    + "__"
                    + str(self.seed)
                    + "__"
                    + str(timestamp)
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
        env = gym.wrappers.ClipAction(env)
        env.action_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


@mlflow_monitoring()
def main(args: Args):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = max(1, args.total_timesteps // args.batch_size)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    mlflow.set_tag("videos_path", f"{RUN_PATH}/videos/{args.mlflow.run_name}")

    device = torch.device(args.device)
    controller = create_controller(args.env_id)

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
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    metrics_collector = create_metrics_collector(
        args.env_id, args.rolling_average_window
    )
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        dtype=torch.float32,
        device=device,
    )
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape,
        dtype=torch.float32,
        device=device,
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    episode_trajectories = [[] for _ in range(args.num_envs)]
    best_values = np.full((args.num_envs, 1), fill_value=-np.inf)
    relax_probs = np.full((args.num_envs, 1), fill_value=args.calfq_p_relax_init)
    p_relax_decay = args.calfq_p_relax_decay
    n_safe_actions = np.zeros((args.num_envs, 1), dtype=np.float32)
    rolling_episode_lengths = deque(maxlen=args.rolling_average_window)
    action_low = envs.single_action_space.low
    action_high = envs.single_action_space.high

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                sampled_action, sampled_logprob, _, value = agent.get_action_and_value(
                    next_obs
                )
                values[step] = value.flatten()

            policy_action_np = sampled_action.cpu().numpy().clip(action_low, action_high)
            current_values = value.detach().cpu().numpy()
            is_values_improved = (
                current_values > best_values + args.calfq_critic_improvement_threshold
            )
            safe_actions = controller.get_action(next_obs_np)
            relax_draw = np.random.rand(*relax_probs.shape) < relax_probs
            use_policy_mask = np.logical_or(is_values_improved, relax_draw)
            current_actions_np = np.where(
                use_policy_mask,
                policy_action_np,
                safe_actions,
            ).astype(np.float32)
            relax_probs *= p_relax_decay
            n_safe_actions += (~use_policy_mask).astype(np.float32)

            actions[step] = sampled_action
            logprobs[step] = sampled_logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(
                current_actions_np
            )

            for env_idx in range(args.num_envs):
                episode_trajectories[env_idx].append(
                    {
                        "obs": obs[step, env_idx].detach().cpu().numpy().copy(),
                        "actions": current_actions_np[env_idx].copy(),
                        "reward": np.array([reward[env_idx]], dtype=np.float32),
                    }
                )

            best_values = np.where(
                is_values_improved,
                current_values,
                best_values,
            )

            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

            if "final_info" in infos:
                for env_idx, info in enumerate(infos["final_info"]):
                    if info is None:
                        continue

                    episode_return = float(np.asarray(info["episode"]["r"]).item())
                    episode_len = float(np.asarray(info["episode"]["l"]).item())
                    metrics_collector.collect_metrics_from_final_episode_info(
                        info, global_step
                    )
                    print(f"global_step={global_step}, episodic_return={episode_return}")

                    metrics_collector.append_metric(
                        "episode_stats/n_safe_actions",
                        float(n_safe_actions[env_idx, 0]) / max(episode_len, 1.0),
                        step=global_step,
                    )
                    metrics_collector.rolling_window["n_safe_actions"].append(
                        float(n_safe_actions[env_idx, 0])
                    )
                    metrics_collector.append_metric(
                        f"episode_stats/n_safe_actions_rolling_{args.rolling_average_window}",
                        float(np.mean(metrics_collector.rolling_window["n_safe_actions"])),
                        step=global_step,
                    )
                    rolling_episode_lengths.append(int(episode_len))
                    metrics_collector.log_pending_metrics(synchronous=True)

                    json_name = (
                        f"{global_step:010d}.json"
                        if args.num_envs == 1
                        else f"{global_step:010d}_{env_idx}.json"
                    )
                    episode_trajectory = episode_trajectories[env_idx]
                    log_json_artifact(
                        episode_trajectory,
                        "trajectories",
                        json_name=json_name,
                    )
                    if args.env_id.startswith("RobotNavigation"):
                        log_robot_nav_trajectory(
                            episode_trajectory,
                            global_step,
                            total_reward=episode_return,
                            goal_reached=bool(info.get("goal_reached", False)),
                        )
                    episode_trajectories[env_idx] = []

                for env_idx, done in enumerate(next_done_np):
                    if not done:
                        continue

                    best_values[env_idx, 0] = -np.inf
                    n_safe_actions[env_idx, 0] = 0.0

                    if args.calfq_anneal and len(rolling_episode_lengths) > 0:
                        frac = np.clip(
                            global_step / (args.total_timesteps * args.calfq_anneal_frac),
                            0.0,
                            1.0,
                        )
                        relax_probs[env_idx, 0] = (
                            args.calfq_p_relax_init
                            + (1.0 - args.calfq_p_relax_init) * frac
                        )
                        episode_len = int(
                            np.sum(rolling_episode_lengths) / len(rolling_episode_lengths)
                        )
                        reference_init = sum(
                            args.calfq_p_relax_init
                            * args.calfq_p_relax_decay ** np.arange(episode_len)
                        )
                        reference = reference_init + (episode_len - reference_init) * frac
                        p_relax_decay = root_scalar(
                            lambda x: sum(x**t for t in range(episode_len))
                            - reference / relax_probs[env_idx, 0],
                            bracket=[0.0, 1.0],
                            method="brentq",
                        ).root
                    else:
                        relax_probs[env_idx, 0] = args.calfq_p_relax_init
                        p_relax_decay = args.calfq_p_relax_decay

                    metrics_collector.append_metric(
                        "calfq/p_relax",
                        float(relax_probs[env_idx, 0]),
                        step=global_step,
                    )
                    metrics_collector.append_metric(
                        "calfq/p_relax_decay",
                        float(p_relax_decay),
                        step=global_step,
                    )
                    metrics_collector.log_pending_metrics(synchronous=True)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta
                    + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        # PPO learns from the learning-policy actions sampled during rollout.
        # The environment may execute a safe action after the CALF gate, but the
        # actor loss is still formed against the sampled policy action.
        selected_inds = np.arange(args.batch_size)

        clipfracs = []
        old_approx_kl = torch.tensor(0.0, device=device)
        approx_kl = torch.tensor(0.0, device=device)
        pg_loss = torch.tensor(0.0, device=device)
        v_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)

        if len(selected_inds) > 0:
            ppo_inds = selected_inds.copy()
            for _ in range(args.update_epochs):
                np.random.shuffle(ppo_inds)
                for start in range(0, len(ppo_inds), args.minibatch_size):
                    mb_inds = ppo_inds[start : start + args.minibatch_size]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        )

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(
                            v_loss_unclipped, v_loss_clipped
                        ).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss
                        - args.ent_coef * entropy_loss
                        + v_loss * args.vf_coef
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = int(global_step / max(time.time() - start_time, 1e-9))

        metrics_collector.append_metric(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], step=global_step
        )
        metrics_collector.append_metric(
            "losses/value_loss", v_loss.item(), step=global_step
        )
        metrics_collector.append_metric(
            "losses/policy_loss", pg_loss.item(), step=global_step
        )
        metrics_collector.append_metric(
            "losses/entropy", entropy_loss.item(), step=global_step
        )
        metrics_collector.append_metric(
            "losses/old_approx_kl", old_approx_kl.item(), step=global_step
        )
        metrics_collector.append_metric(
            "losses/approx_kl", approx_kl.item(), step=global_step
        )
        metrics_collector.append_metric(
            "losses/clipfrac",
            float(np.mean(clipfracs)) if clipfracs else 0.0,
            step=global_step,
        )
        metrics_collector.append_metric(
            "losses/explained_variance", explained_var, step=global_step
        )
        metrics_collector.append_metric("charts/SPS", sps, step=global_step)
        print("SPS:", sps)
        metrics_collector.log_pending_metrics(synchronous=False)

    envs.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
