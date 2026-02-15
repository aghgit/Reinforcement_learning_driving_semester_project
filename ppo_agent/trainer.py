# === ppo_trainer.py ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Shared layers
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        # Actor (Policy)
        self.policy_head = nn.Linear(64, action_dim)

        # Critic (Value)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def act(self, state):
        x = self.forward(state)
        logits = self.policy_head(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, states, actions):
        x = self.forward(states)
        logits = self.policy_head(x)
        dist = Categorical(logits=logits)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.value_head(x).squeeze()

        return action_logprobs, state_values, dist_entropy
