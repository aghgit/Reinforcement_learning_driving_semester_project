import torch
import torch.optim as optim
import gym
from ppo_agent.buffer import RolloutBuffer, ppo_update
from ppo_agent.trainer import PPO

# === Hyperparameters ===
env_name = "CartPole-v1"
gamma = 0.99
eps_clip = 0.2
K_epochs = 4
lr = 3e-4

# === Create environment ===
env = gym.make(env_name)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# === Initialize PPO agent and optimizer ===
agent = PPO(obs_dim, action_dim)
optimizer = optim.Adam(agent.parameters(), lr=lr)
buffer = RolloutBuffer()

# === Training loop ===
n_episodes = 1000
max_timesteps = 300
log_interval = 20

for episode in range(1, n_episodes + 1):
    state = env.reset()
    episode_reward = 0

    for t in range(max_timesteps):
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # Sample action
        action, log_prob = agent.act(state_tensor)

        # Step environment
        next_state, reward, done, _ = env.step(action)

        # Store in buffer
        buffer.states.append(state_tensor)
        buffer.actions.append(torch.tensor(action))
        buffer.log_probs.append(log_prob)
        buffer.rewards.append(reward)
        buffer.is_terminals.append(done)

        state = next_state
        episode_reward += reward

        if done:
            break

    # PPO update after every episode
    ppo_update(agent, buffer, optimizer, eps_clip, K_epochs, gamma)

    if episode % log_interval == 0:
        print(f"Episode {episode}\tReward: {episode_reward:.2f}")

env.close()