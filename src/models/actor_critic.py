import torch as th
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

# Actor-Critic Network for only vector observations with separate policy and value networks
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, device=None):
        super().__init__()

        print(f"obs_dim: {obs_dim}")
        
        # Separate policy network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim)
        )
        
        # Separate value network
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights deterministically
        self._init_weights()
        
        # Move model to device if specified
        if device is not None:
            self.to(device)
    
    def _init_weights(self):
        """Initialize network weights deterministically"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def get_action(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze()

    def evaluate_actions(self, obs, actions):
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values.squeeze()