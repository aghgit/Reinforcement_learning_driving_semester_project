# === memory_and_update.py ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]

def compute_returns(rewards, is_terminals, gamma=0.99):
    returns = []
    discounted_sum = 0
    for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
        if is_terminal:
            discounted_sum = 0
        discounted_sum = reward + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    return returns

def ppo_update(agent, buffer, optimizer, eps_clip=0.2, K_epochs=4, gamma=0.99):
    # Step 1: compute returns
    returns = compute_returns(buffer.rewards, buffer.is_terminals, gamma)
    returns = torch.tensor(returns, dtype=torch.float32)

    # Convert lists to tensors
    old_states = torch.stack(buffer.states).detach()
    old_actions = torch.stack(buffer.actions).detach()
    old_log_probs = torch.stack(buffer.log_probs).detach()

    # Step 2: Perform K epochs of updates
    for _ in range(K_epochs):
        logprobs, state_values, dist_entropy = agent.evaluate(old_states, old_actions)
        advantages = returns - state_values.detach()

        # Step 3: ratio for clipped objective
        ratios = torch.exp(logprobs - old_log_probs)

        # Step 4: clipped surrogate objective
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
        loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(state_values, returns) - 0.01 * dist_entropy.mean()

        # Step 5: update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Clear memory after update
    buffer.clear()