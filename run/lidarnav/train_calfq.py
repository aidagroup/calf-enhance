# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass, field
from scipy.optimize import root_scalar
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from src.utils.mlflow import mlflow_monitoring, MlflowConfig, log_json_artifact
from src import RUN_PATH
import stable_baselines3 as sb3
import mlflow
from collections import defaultdict, deque
from src.controller import LidarNavController


@dataclass
class Args:
    mlflow: MlflowConfig = field(
        default_factory=lambda: MlflowConfig(
            tracking_uri=f"file://{RUN_PATH}/mlruns",
            experiment_name=os.path.basename(__file__)[: -len(".py")],
        )
    )
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "UnderwaterDrone-v0"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    rolling_average_window: int = 20
    """the rolling average window for the metrics"""

    # Calfq specific arguments
    calfq_critic_improvement_threshold: float = 0.01
    """the threshold for the critic improvement"""
    calfq_p_relax_init: float = 0.0
    """the initial value of the p_relax parameter"""
    calfq_p_relax_decay: float = 0.95
    """the decay rate of the p_relax parameter"""
    calfq_anneal: bool = True
    """if toggled, the p_relax and p_relax_decay parameters will be annealed"""
    calfq_anneal_frac: float = 0.9
    """the fraction of the total timesteps to anneal the p_relax and p_relax_decay parameters"""

    def __post_init__(self):
        self.mlflow.experiment_name = (
            os.path.basename(__file__)[: -len(".py")] + "__" + self.env_id
        )
        self.mlflow.run_name = (
            self.mlflow.experiment_name
            + "__"
            + str(self.seed)
            + "__"
            + str(int(time.time()))
        )


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


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
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
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
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
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


@mlflow_monitoring()
def main(args: Args):
    mlflow.set_tag("videos_path", f"{RUN_PATH}/videos/{args.mlflow.run_name}")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # env setup
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

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    rolling_window = defaultdict(lambda: deque(maxlen=args.rolling_average_window))

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    def q_values(obs, actions):
        torch_obs, torch_actions = (
            torch.tensor(obs, device=device),
            torch.tensor(actions, device=device),
        )
        with torch.no_grad():
            torch_q_values = torch.min(
                qf1_target(torch_obs, torch_actions),
                qf2_target(torch_obs, torch_actions),
            )
            return torch_q_values.cpu().numpy()

    if args.env_id.startswith("LidarNav"):
        controller = LidarNavController()
    else:
        raise ValueError(f"Environment {args.env_id} not supported")

    start_time = time.time()
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    best_q_values = np.full(shape=(envs.num_envs, 1), fill_value=-np.inf)
    relax_probs = np.full(shape=(envs.num_envs, 1), fill_value=args.calfq_p_relax_init)
    p_relax_decay = args.calfq_p_relax_decay
    n_safe_actions = np.zeros(shape=(envs.num_envs, 1))
    rolling_episode_lengths = deque(maxlen=args.rolling_average_window)
    episode_trajectory = []
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = (
                    actions.cpu()
                    .numpy()
                    .clip(envs.single_action_space.low, envs.single_action_space.high)
                )

        current_q_values = q_values(obs, actions)
        is_q_values_improved = (
            current_q_values > best_q_values + args.calfq_critic_improvement_threshold
        )
        safe_actions = controller.get_action(obs)
        current_actions = np.where(
            np.logical_or(
                is_q_values_improved,
                np.random.rand(*relax_probs.shape) < relax_probs,
            ),
            actions,
            safe_actions,
        )
        relax_probs *= p_relax_decay
        safe_actions_mask = ~np.logical_or(
            is_q_values_improved,
            np.random.rand(*relax_probs.shape) < relax_probs,
        )
        n_safe_actions += safe_actions_mask

        next_obs, rewards, terminations, truncations, infos = envs.step(
            np.array(current_actions, dtype=float)
        )

        episode_trajectory.append(
            {
                "obs": obs,
                "actions": current_actions,
            }
        )

        best_q_values = np.where(
            is_q_values_improved,
            current_q_values,
            best_q_values,
        )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:

                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}"
                    )
                    mlflow.log_metric(
                        "charts/episodic_return", info["episode"]["r"], global_step
                    )
                    mlflow.log_metric(
                        "charts/episodic_length", info["episode"]["l"], global_step
                    )

                    rolling_window["episodic_return"].append(info["episode"]["r"])
                    if (
                        len(rolling_window["episodic_return"])
                        > args.rolling_average_window
                    ):
                        rolling_window["episodic_return"].pop(0)
                    mlflow.log_metric(
                        f"charts/episodic_return_rolling_{args.rolling_average_window}",
                        sum(rolling_window["episodic_return"])
                        / len(rolling_window["episodic_return"]),
                        global_step,
                    )

                    # mlflow.log_metric(
                    #     "episode_stats/is_in_hole",
                    #     info["is_in_hole"],
                    #     global_step,
                    # )
                    # rolling_window["is_in_hole"].append(info["is_in_hole"])
                    # mlflow.log_metric(
                    #     f"episode_stats/is_in_hole_rolling_{args.rolling_average_window}",
                    #     np.mean(rolling_window["is_in_hole"]),
                    #     global_step,
                    # )

                    mlflow.log_metric(
                        "episode_stats/n_safe_actions",
                        np.mean(n_safe_actions) / info["episode"]["l"],
                        global_step,
                    )
                    rolling_window["n_safe_actions"].append(np.mean(n_safe_actions))

                    mlflow.log_metric(
                        f"episode_stats/n_safe_actions_rolling_{args.rolling_average_window}",
                        np.mean(rolling_window["n_safe_actions"]),
                        global_step,
                    )

                    # mlflow.log_metric(
                    #     "episode_stats/avoidance_score",
                    #     info["avoidance_score"],
                    #     global_step,
                    # )
                    # rolling_window["avoidance_score"].append(info["avoidance_score"])
                    # mlflow.log_metric(
                    #     f"episode_stats/avoidance_score_rolling_{args.rolling_average_window}",
                    #     np.mean(rolling_window["avoidance_score"]),
                    #     global_step,
                    # )
                    rolling_episode_lengths.append(info["episode"]["l"])

                    log_json_artifact(
                        episode_trajectory,
                        f"trajectories",
                        json_name=f"{global_step:010d}.json",
                    )
                    episode_trajectory = []
                    break
        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, (trunc, term) in enumerate(zip(truncations, terminations)):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

            if trunc or term:
                best_q_values[idx, 0] = -np.inf
                n_safe_actions[idx, 0] = 0

                if args.calfq_anneal:
                    frac = np.clip(
                        global_step / (args.total_timesteps * args.calfq_anneal_frac),
                        0.0,
                        1.0,
                    )
                    relax_probs[idx, 0] = (
                        args.calfq_p_relax_init + (1.0 - args.calfq_p_relax_init) * frac
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
                        - reference / relax_probs[idx, 0],
                        bracket=[0.0, 1.0],
                        method="brentq",
                    ).root
                else:
                    relax_probs[idx, 0] = args.calfq_p_relax_init
                    p_relax_decay = args.calfq_p_relax_decay

                mlflow.log_metric(
                    "calfq/p_relax",
                    relax_probs[idx, 0],
                    global_step,
                )
                mlflow.log_metric(
                    "calfq/p_relax_decay",
                    p_relax_decay,
                    global_step,
                )

        if np.any(~safe_actions_mask):
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (
                    torch.randn_like(data.actions, device=device) * args.policy_noise
                ).clamp(-args.noise_clip, args.noise_clip) * target_actor.action_scale

                next_state_actions = (
                    target_actor(data.next_observations) + clipped_noise
                ).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(
                    actor.parameters(), target_actor.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                mlflow.log_metric(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                mlflow.log_metric(
                    "losses/qf2_values", qf2_a_values.mean().item(), global_step
                )
                mlflow.log_metric("losses/qf1_loss", qf1_loss.item(), global_step)
                mlflow.log_metric("losses/qf2_loss", qf2_loss.item(), global_step)
                mlflow.log_metric("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                mlflow.log_metric("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                mlflow.log_metric(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

    # if args.save_model:
    #     model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
    #     torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
    #     print(f"model saved to {model_path}")
    #     from cleanrl_utils.evals.td3_eval import evaluate

    #     episodic_returns = evaluate(
    #         model_path,
    #         make_env,
    #         args.env_id,
    #         eval_episodes=10,
    #         run_name=f"{run_name}-eval",
    #         Model=(Actor, QNetwork),
    #         device=device,
    #         exploration_noise=args.exploration_noise,
    #     )
    #     for idx, episodic_return in enumerate(episodic_returns):
    #         mlflow.log_metric("eval/episodic_return", episodic_return, idx)

    #     if args.upload_model:
    #         from cleanrl_utils.huggingface import push_to_hub

    #         repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
    #         repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
    #         push_to_hub(
    #             args,
    #             episodic_returns,
    #             repo_id,
    #             "TD3",
    #             f"runs/{run_name}",
    #             f"videos/{run_name}-eval",
    #         )

    envs.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
