# === scripts/evaluation/test_agent.py ===

import torch
import gym
import os
from ppo_agent.ppo_agent import PPOAgent

# === Settings ===
env_name = "CartPole-v1"
model_path = os.path.join("models", "ppo_cartpole.pth")  # Make sure this matches what you saved

# === Create Environment ===
env = gym.make(env_name)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# === Load agent ===
agent = PPOAgent(obs_dim, action_dim)
agent.policy.load_state_dict(torch.load(model_path))
agent.policy.eval()

# === Run test episodes ===
n_episodes = 10
max_timesteps = 300

for ep in range(1, n_episodes + 1):
    state = env.reset()
    total_reward = 0

    for t in range(max_timesteps):
        env.render()

        action, _, _ = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    print(f"Test Episode {ep} — Total Reward: {total_reward:.2f}")

env.close()
