# Colab-ready experiment: imperfect BC + residual RL improvement.
#
# Copy this file into a Colab cell sequence, or upload it as a .py file and run:
#   !python imperfect_bc_residual_colab.py
#
# Main idea:
#   1. Train an intentionally imperfect BC controller from limited demonstrations.
#   2. Freeze BC.
#   3. Add actuator bias so BC-only degrades under distribution/execution shift.
#   4. Train PPO/SAC residual policies to correct BC.
#   5. Compare BC-only vs BC+residual on clean and disturbed environments.

# %% Install dependencies
# In Colab, uncomment this line:
# !pip install -q "stable-baselines3[extra]" gymnasium tensorboard

# %% Imports and config
import os
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.data import DataLoader, TensorDataset


SEED = 7
ENV_NAME = "Pendulum-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Deliberately limited BC data/training so residual RL has room to improve.
EXPERT_TRANSITIONS = 3_000
BC_EPOCHS = 20
BC_BATCH_SIZE = 128

# Residual RL settings.
PPO_TIMESTEPS = 200_000
SAC_TIMESTEPS = 200_000
EVAL_EPISODES = 20

# Residual action bound in normalized action coordinates.
# Pendulum physical torque range is [-2, 2], so epsilon=0.20 means up to 0.4 torque correction.
RESIDUAL_EPSILON = 0.20
RESIDUAL_LAMBDA = 0.001

# Disturbed setting: persistent actuator bias plus small noise.
# This is learnable, unlike purely random large per-step noise.
ACTION_BIAS = 0.30
ACTION_NOISE_STD = 0.02

BC_PATH = "weak_bc_expert.pth"
PPO_PATH = "ppo_residual_imperfect_bc"
SAC_PATH = "sac_residual_imperfect_bc"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed()
print("device:", DEVICE)


# %% Utilities
def normalized_to_physical(normalized_action, action_space):
    normalized_action = np.asarray(normalized_action, dtype=np.float32)
    low = action_space.low.astype(np.float32)
    high = action_space.high.astype(np.float32)
    midpoint = (high + low) / 2.0
    scale = (high - low) / 2.0
    return midpoint + np.clip(normalized_action, -1.0, 1.0) * scale


def physical_to_normalized(physical_action, action_space):
    physical_action = np.asarray(physical_action, dtype=np.float32)
    low = action_space.low.astype(np.float32)
    high = action_space.high.astype(np.float32)
    midpoint = (high + low) / 2.0
    scale = (high - low) / 2.0
    return np.clip((physical_action - midpoint) / (scale + 1e-8), -1.0, 1.0)


def analytic_pendulum_expert(env):
    """Simple stabilizing controller used only to generate demonstrations."""
    theta, theta_dot = env.unwrapped.state
    torque = -2.0 * theta - 0.5 * theta_dot
    torque = np.clip(torque, env.action_space.low[0], env.action_space.high[0])
    return np.asarray([torque], dtype=np.float32)


def collect_limited_demonstrations(num_transitions=EXPERT_TRANSITIONS):
    env = gym.make(ENV_NAME)
    states, actions = [], []
    obs, _ = env.reset(seed=SEED)

    while len(states) < num_transitions:
        physical_action = analytic_pendulum_expert(env)
        normalized_action = physical_to_normalized(physical_action, env.action_space)

        states.append(obs.astype(np.float32))
        actions.append(normalized_action.astype(np.float32))

        obs, _, terminated, truncated, _ = env.step(physical_action)
        if terminated or truncated:
            obs, _ = env.reset(seed=SEED + len(states))

    env.close()
    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32)


# %% Imperfect BC policy
class WeakBCPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(64, 64)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, obs):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
            squeeze = x.ndim == 1
            if squeeze:
                x = x.unsqueeze(0)
            action = self.forward(x).detach().cpu().numpy().astype(np.float32)
        return action[0] if squeeze else action

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        return self


def train_weak_bc():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    env.close()

    print("Collecting limited demonstrations...")
    states, actions = collect_limited_demonstrations()

    bc = WeakBCPolicy(state_dim, action_dim).to(DEVICE)
    dataset = TensorDataset(torch.from_numpy(states), torch.from_numpy(actions))
    loader = DataLoader(dataset, batch_size=BC_BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(bc.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("Training intentionally imperfect BC...")
    bc.train()
    for epoch in range(1, BC_EPOCHS + 1):
        total_loss = 0.0
        for batch_states, batch_actions in loader:
            batch_states = batch_states.to(DEVICE)
            batch_actions = batch_actions.to(DEVICE)
            pred = bc(batch_states)
            loss = loss_fn(pred, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_states.size(0)

        if epoch == 1 or epoch % 5 == 0 or epoch == BC_EPOCHS:
            print(f"BC epoch {epoch:03d}/{BC_EPOCHS}, mse={total_loss / len(dataset):.6f}")

    bc.freeze()
    torch.save(bc.state_dict(), BC_PATH)
    print(f"Saved weak BC to {BC_PATH}")
    return bc


# %% Wrappers
class ActionDisturbanceWrapper(gym.ActionWrapper):
    """Adds persistent actuator bias and small noise in physical torque space."""

    def __init__(self, env, action_bias=0.0, action_noise_std=0.0):
        super().__init__(env)
        self.action_bias = np.full(env.action_space.shape, action_bias, dtype=np.float32)
        self.action_noise_std = float(action_noise_std)

    def action(self, action):
        action = np.asarray(action, dtype=np.float32)
        noise = np.zeros(self.action_space.shape, dtype=np.float32)
        if self.action_noise_std > 0.0:
            noise = self.np_random.normal(
                loc=0.0,
                scale=self.action_noise_std,
                size=self.action_space.shape,
            ).astype(np.float32)
        return np.clip(action + self.action_bias + noise, self.action_space.low, self.action_space.high)


class ResidualControlWrapper(gym.Wrapper):
    """RL action is residual; executed action is BC(obs) + epsilon * residual."""

    def __init__(self, env, bc_policy, epsilon=RESIDUAL_EPSILON, lamda=RESIDUAL_LAMBDA):
        super().__init__(env)
        self.bc_policy = bc_policy
        self.epsilon = float(epsilon)
        self.lamda = float(lamda)
        self.current_obs = None
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs.astype(np.float32)
        return self.current_obs, info

    def step(self, residual_action):
        base_action = self.bc_policy.predict(self.current_obs)
        residual_action = np.asarray(residual_action, dtype=np.float32).reshape(self.action_space.shape)
        delta_action = np.clip(residual_action, -1.0, 1.0) * self.epsilon

        normalized_action = np.clip(base_action + delta_action, -1.0, 1.0)
        physical_action = normalized_to_physical(normalized_action, self.env.action_space)

        obs, raw_reward, terminated, truncated, info = self.env.step(physical_action.astype(np.float32))
        self.current_obs = obs.astype(np.float32)

        residual_penalty = self.lamda * float(
            np.sum(np.square(delta_action)) / (self.epsilon**2 + 1e-8)
        )
        shaped_reward = float(raw_reward) - residual_penalty
        info = dict(info)
        info["raw_reward"] = float(raw_reward)
        info["residual_penalty"] = residual_penalty
        return self.current_obs, shaped_reward, terminated, truncated, info


def make_env(bc_policy, disturbed=False, lamda=RESIDUAL_LAMBDA):
    env = gym.make(ENV_NAME)
    if disturbed:
        env = ActionDisturbanceWrapper(
            env,
            action_bias=ACTION_BIAS,
            action_noise_std=ACTION_NOISE_STD,
        )
    env = ResidualControlWrapper(
        env,
        bc_policy,
        epsilon=RESIDUAL_EPSILON,
        lamda=lamda,
    )
    return Monitor(env)


# %% Evaluation
def evaluate_residual_policy(model, bc_policy, disturbed=False, episodes=EVAL_EPISODES, name="policy"):
    env = make_env(bc_policy, disturbed=disturbed, lamda=0.0)
    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=SEED + 10_000 + ep)
        done = False
        total_reward = 0.0
        while not done:
            if model is None:
                residual_action = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                residual_action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(residual_action)
            total_reward += float(info.get("raw_reward", reward))
            done = terminated or truncated
        rewards.append(total_reward)

    env.close()
    mean, std = float(np.mean(rewards)), float(np.std(rewards))
    print(f"{name}: {mean:.2f} +/- {std:.2f}")
    return mean, std


def print_result_table(results):
    print("\n=== Summary: task reward, higher is better ===")
    print(f"{'Method':<30} {'Clean':>18} {'Disturbed':>18}")
    print("-" * 70)
    for name, vals in results.items():
        clean = vals.get("clean")
        disturbed = vals.get("disturbed")
        clean_s = f"{clean[0]:.2f} +/- {clean[1]:.2f}" if clean else "-"
        dist_s = f"{disturbed[0]:.2f} +/- {disturbed[1]:.2f}" if disturbed else "-"
        print(f"{name:<30} {clean_s:>18} {dist_s:>18}")


# %% Train BC and evaluate BC-only
bc_model = train_weak_bc()

results = {}
results["Weak BC only"] = {
    "clean": evaluate_residual_policy(None, bc_model, disturbed=False, name="Weak BC only clean"),
    "disturbed": evaluate_residual_policy(None, bc_model, disturbed=True, name="Weak BC only disturbed"),
}


# %% Train PPO residual
train_env = DummyVecEnv([lambda: make_env(bc_model, disturbed=True, lamda=RESIDUAL_LAMBDA)])
eval_env = DummyVecEnv([lambda: make_env(bc_model, disturbed=True, lamda=0.0)])

ppo_eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_ppo_imperfect_bc",
    log_path="./ppo_imperfect_logs",
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
)

ppo_model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=1e-4,
    n_steps=1024,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    policy_kwargs=dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        log_std_init=-1.0,
    ),
    verbose=0,
    seed=SEED,
    device=DEVICE,
)

print("\nTraining PPO residual on disturbed environment...")
ppo_model.learn(total_timesteps=PPO_TIMESTEPS, callback=ppo_eval_callback)
ppo_model.save(PPO_PATH)
best_ppo = PPO.load("./best_ppo_imperfect_bc/best_model", env=train_env, device=DEVICE)

results["Weak BC + PPO residual"] = {
    "clean": evaluate_residual_policy(best_ppo, bc_model, disturbed=False, name="PPO residual clean"),
    "disturbed": evaluate_residual_policy(best_ppo, bc_model, disturbed=True, name="PPO residual disturbed"),
}


# %% Train SAC residual
train_env = DummyVecEnv([lambda: make_env(bc_model, disturbed=True, lamda=RESIDUAL_LAMBDA)])
eval_env = DummyVecEnv([lambda: make_env(bc_model, disturbed=True, lamda=0.0)])

sac_eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_sac_imperfect_bc",
    log_path="./sac_imperfect_logs",
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
)

sac_model = SAC(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    buffer_size=100_000,
    learning_starts=1_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    ent_coef="auto",
    train_freq=1,
    gradient_steps=1,
    policy_kwargs=dict(activation_fn=nn.ReLU, net_arch=[256, 256]),
    verbose=0,
    seed=SEED,
    device=DEVICE,
)

print("\nTraining SAC residual on disturbed environment...")
sac_model.learn(total_timesteps=SAC_TIMESTEPS, callback=sac_eval_callback)
sac_model.save(SAC_PATH)
best_sac = SAC.load("./best_sac_imperfect_bc/best_model", env=train_env, device=DEVICE)

results["Weak BC + SAC residual"] = {
    "clean": evaluate_residual_policy(best_sac, bc_model, disturbed=False, name="SAC residual clean"),
    "disturbed": evaluate_residual_policy(best_sac, bc_model, disturbed=True, name="SAC residual disturbed"),
}

print_result_table(results)


# %% Optional: save results
np.savez(
    "imperfect_bc_residual_results.npz",
    method_names=np.asarray(list(results.keys())),
    clean_means=np.asarray([results[k]["clean"][0] for k in results]),
    clean_stds=np.asarray([results[k]["clean"][1] for k in results]),
    disturbed_means=np.asarray([results[k]["disturbed"][0] for k in results]),
    disturbed_stds=np.asarray([results[k]["disturbed"][1] for k in results]),
)

print("\nSaved:")
print(" ", BC_PATH)
print(" ", PPO_PATH + '.zip')
print(" ", SAC_PATH + '.zip')
print(" ", "imperfect_bc_residual_results.npz")
