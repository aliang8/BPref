# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pickle as pkl

import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
import utils
from torch.optim.lr_scheduler import CosineAnnealingLR

# MetaWorld imports
import metaworld

# Import RewardModel from robot_pref
import sys
sys.path.append('/scr/aliang80/robot_pref')
from models.reward_models import RewardModel

# Usage example for reward model relabeling:
# python iql.py reward_type=reward_model reward_model_path=/path/to/trained/reward_model.pt

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "metaworld_sweep-into-v2"  # MetaWorld environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 50  # How many episodes run during evaluation
    max_timesteps: int = 250000
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # Dataset
    replay_buffer_path: str = "/scr/aliang80/BPref/exp/metaworld_sweep-into-v2/H256_L3_B512_tau0.005/unsup0_topk5_sac_lr0.0003_temp0.1_seed12345/replay_buffer_step_60000.pkl"  # Path to PEBBLE replay buffer pickle file
    reward_model_path: str = ""  # Path to trained reward model for reward_model reward type
    # IQL
    buffer_size: int = 1_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    # Wandb logging
    project: str = "robot_pref"
    entity: str = "clvr"
    group: str = "IQL-MetaWorld"
    name: str = "IQL"
    reward_type: str = "zero"  # Options: "zero", "negative", "uniform", "original", "reward_model"
    # "zero": All rewards set to 0 (sanity check)
    # "negative": Flip sign of original rewards  
    # "uniform": Random uniform rewards [0,1]
    # "original": Keep original rewards from dataset
    # "reward_model": Use trained reward model to relabel data (requires reward_model_path)

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


# def make_metaworld_env(env_name: str, seed: int = 0):
#     """Create a MetaWorld environment with the exact specified name.

#     Args:
#         env_name: Exact name of the MetaWorld environment to create
#         seed: Random seed for the environment

#     Returns:
#         MetaWorld environment instance
#     """
#     from metaworld.envs import (
#         ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
#         ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
#     )

#     # Try to find the environment in the goal observable environments
#     if env_name in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
#         env_constructor = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
#     # Otherwise, try to find it in the goal hidden environments
#     elif env_name in ALL_V2_ENVIRONMENTS_GOAL_HIDDEN:
#         env_constructor = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_name]
#     else:
#         # If not found, raise a clear error with available options
#         print("Available goal observable environments:")
#         for name in sorted(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys()):
#             print(f"  - {name}")
#         print("\nAvailable goal hidden environments:")
#         for name in sorted(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN.keys()):
#             print(f"  - {name}")
#         raise ValueError(
#             f"Environment '{env_name}' not found in MetaWorld environments. Please use one of the listed environments."
#         )

#     # Create the environment with the specified seed
#     env = env_constructor(seed=seed)
#     return env


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
        reward_type: str = "zero",
        reward_model: Optional[RewardModel] = None
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self.reward_type = reward_type
        self.reward_model = reward_model

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data from PEBBLE replay buffer format
    def load_pebble_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        
        # Handle rewards - ensure it's 1D, then add dimension
        rewards = data["rewards"]
        if rewards.ndim > 1:
            rewards = rewards.squeeze()
        self._rewards[:n_transitions] = self._to_tensor(rewards[..., None])

        # Process rewards based on reward_type
        if self.reward_type == "zero":
            self._rewards[:n_transitions] = 0.0
        elif self.reward_type == "negative":
            self._rewards[:n_transitions] = -self._rewards[:n_transitions]
        elif self.reward_type == "uniform":
            self._rewards[:n_transitions] = self._to_tensor(np.random.uniform(0, 1, size=(n_transitions, 1)))
        elif self.reward_type == "original":
            pass  # Keep original rewards
        elif self.reward_type == "reward_model":
            if self.reward_model is None:
                raise ValueError("reward_model must be provided when using reward_model reward_type")
            
            print("Relabeling data with reward model...")
            self.reward_model.eval()
            batch_size = 1024  # Process in batches to avoid memory issues
            
            with torch.no_grad():
                for start_idx in range(0, n_transitions, batch_size):
                    end_idx = min(start_idx + batch_size, n_transitions)
                    
                    # Get batch of states and actions
                    batch_states = self._states[start_idx:end_idx]
                    batch_actions = self._actions[start_idx:end_idx]
                    
                    # Predict rewards using the reward model
                    predicted_rewards = self.reward_model(batch_states, batch_actions)
                    
                    # Update rewards in buffer
                    self._rewards[start_idx:end_idx] = predicted_rewards.unsqueeze(-1)
            

            # apply min-max normalization to rewards so they bound between 0 and 1
            min_reward = self._rewards[:n_transitions].min()
            max_reward = self._rewards[:n_transitions].max()
            self._rewards[:n_transitions] = (self._rewards[:n_transitions] - min_reward) / (max_reward - min_reward)
        
            print(f"Relabeled {n_transitions} transitions with reward model")
            print(f"New reward stats - Max: {self._rewards[:n_transitions].max():.4f}, Min: {self._rewards[:n_transitions].min():.4f}, Mean: {self._rewards[:n_transitions].mean():.4f}")
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}. Supported types: zero, negative, uniform, original, reward_model")
        
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        
        # Handle dones - ensure it's 1D, then add dimension
        # Convert from not_dones to dones format
        dones = data["dones"]
        if dones.ndim > 1:
            dones = dones.squeeze()
        self._dones[:n_transitions] = self._to_tensor((1.0 - dones)[..., None])
        
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")
        print(f"Observations shape: {self._states.shape}, Max: {self._states.max()}, Min: {self._states.min()}")
        print(f"Actions shape: {self._actions.shape}, Max: {self._actions.max()}, Min: {self._actions.min()}")
        print(f"Rewards shape: {self._rewards.shape}, Max: {self._rewards.max()}, Min: {self._rewards.min()}")
        print(f"Reward type: {self.reward_type}")
        print(f"Final reward stats - Max: {self._rewards[:n_transitions].max():.4f}, Min: {self._rewards[:n_transitions].min():.4f}, Mean: {self._rewards[:n_transitions].mean():.4f}")
        print(f"Dones shape: {self._dones.shape}, Max: {self._dones.max()}, Min: {self._dones.min()}")

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        if hasattr(env, 'action_space'):
            env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        entity=config["entity"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate actor on MetaWorld environment"""
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    success_rates = []
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_success = 0
        
        while not done:
            action = actor.act(obs, device)
            obs, reward, truncated, terminated, info = env.step(action)
            done = truncated or terminated
            episode_reward += reward
            # Track success for MetaWorld
            if 'success' in info:
                episode_success = max(episode_success, info['success'])
                
        episode_rewards.append(episode_reward)
        success_rates.append(episode_success)

    actor.train()
    return np.asarray(episode_rewards), np.asarray(success_rates)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["dones"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    # MetaWorld specific reward modification if needed
    if "button-press" in env_name or "metaworld" in env_name:
        # MetaWorld rewards are typically in [0, 1] range for success
        # May not need modification, but can normalize if needed
        pass
    elif any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    # Create MetaWorld environment
    env = utils.make_metaworld_env(config)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Load PEBBLE replay buffer instead of D4RL dataset
    if not config.replay_buffer_path:
        raise ValueError("replay_buffer_path must be specified for PEBBLE dataset")
    
    print(f"Loading PEBBLE replay buffer from: {config.replay_buffer_path}")
    with open(config.replay_buffer_path, 'rb') as f:
        buffer_data = pkl.load(f)
    
    # Convert PEBBLE buffer format to standard format
    dataset = {
        "observations": buffer_data["observations"],
        "actions": buffer_data["actions"], 
        "rewards": buffer_data["rewards"],
        "next_observations": buffer_data["next_observations"],
        "dones": buffer_data["dones"]
    }

    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    
    # Load reward model if needed
    reward_model = None
    if config.reward_type == "reward_model":
        if not config.reward_model_path:
            raise ValueError("reward_model_path must be specified when using reward_model reward_type")
        
        print(f"Loading reward model from: {config.reward_model_path}")
        reward_model = RewardModel(state_dim, action_dim)
        reward_model.load_state_dict(torch.load(config.reward_model_path, map_location=config.device))
        reward_model = reward_model.to(config.device)
        print("Reward model loaded successfully")
    
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
        reward_type=config.reward_type,
        reward_model=reward_model
    )
    replay_buffer.load_pebble_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    success_rates = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores, eval_successes = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            success_rate = eval_successes.mean() * 100.0
            evaluations.append(eval_score)
            success_rates.append(success_rate)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"Reward: {eval_score:.3f}, Success Rate: {success_rate:.1f}%"
            )
            print("---------------------------------------")
            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            wandb.log(
                {
                    "eval_score": eval_score,
                    "success_rate": success_rate
                }, 
                step=trainer.total_it
            )


if __name__ == "__main__":
    train()