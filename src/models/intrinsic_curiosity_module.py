import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IntrinsicCuriosityModule(nn.Module):
  """
  Minimal Intrinsic Curiosity Module (ICM) implementation.
  
  ICM consists of:
  1. Feature network: Extracts meaningful features from observations
  2. Inverse model: Predicts action given current and next state features
  3. Forward model: Predicts next state features given current features and action
  
  The prediction error of the forward model serves as intrinsic reward.
  """
  
  def __init__(self, feature_dim=128, action_dim=4, eta=0.01, beta=0.2, device=None):
    super().__init__()
    self.device = device
    self.eta = eta  # Scaling factor for intrinsic reward
    self.beta = beta  # Weight for inverse loss vs forward loss
    
    # Feature network - reuses the encoding from the main model
    # We'll extract features from the actor-critic's shared encoding
    self.feature_dim = feature_dim
    
    # Inverse model: predicts action from state features
    self.inverse_model = nn.Sequential(
      nn.Linear(feature_dim * 2, 256),  # concat current and next features
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, action_dim)
    )
    
    # Forward model: predicts next state features from current features + action
    self.forward_model = nn.Sequential(
      nn.Linear(feature_dim + action_dim, 256),  # features + one-hot action
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, feature_dim)
    )
    
    if device is not None:
      self.to(device)
  
  def get_features(self, actor_critic_model, observations):
    """Extract features using the actor-critic model's shared encoding"""
    with th.no_grad():
      features = actor_critic_model._encode_observations(observations)
    return features
  
  def forward(self, current_features, next_features, actions):
    """
    Forward pass through ICM
    
    Args:
        current_features: Features from current state
        next_features: Features from next state  
        actions: Actions taken (as indices)
        
    Returns:
        inverse_loss: Loss from inverse model
        forward_loss: Loss from forward model
        intrinsic_reward: Curiosity-driven reward
    """
    batch_size = current_features.shape[0]
    
    # Convert actions to one-hot encoding
    actions_onehot = F.one_hot(actions.long(), num_classes=self.inverse_model[-1].out_features).float()
    
    # Inverse model: predict action from state transition
    concat_features = th.cat([current_features, next_features], dim=1)
    predicted_actions = self.inverse_model(concat_features)
    inverse_loss = F.cross_entropy(predicted_actions, actions.long())
    
    # Forward model: predict next state features
    forward_input = th.cat([current_features, actions_onehot], dim=1)
    predicted_next_features = self.forward_model(forward_input)
    forward_loss = F.mse_loss(predicted_next_features, next_features.detach())
    
    # Intrinsic reward is the prediction error (curiosity)
    intrinsic_reward = self.eta * forward_loss.detach()
    
    return inverse_loss, forward_loss, intrinsic_reward
  
  def compute_icm_loss(self, current_features, next_features, actions):
    """Compute the combined ICM loss"""
    inverse_loss, forward_loss, _ = self.forward(current_features, next_features, actions)
    
    # Combined loss: weighted sum of inverse and forward losses
    total_loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss
    
    return total_loss, inverse_loss, forward_loss

  def compute_intrinsic_reward(self, current_features, next_features, actions):
    """Compute intrinsic reward for curiosity-driven exploration"""
    _, _, intrinsic_reward = self.forward(current_features, next_features, actions)
    return intrinsic_reward


class ActorCriticWithICM(nn.Module):
  """
  Wrapper that combines the multimodal actor-critic with ICM
  """
  
  def __init__(self, actor_critic_model, icm_module: IntrinsicCuriosityModule):
    super().__init__()
    self.actor_critic = actor_critic_model
    self.icm = icm_module
    self.device = actor_critic_model.device

  def forward(self, observations):
    return self.actor_critic(observations)
  
  def get_action(self, obs, deterministic=False):
    return self.actor_critic.get_action(obs, deterministic)
  
  def evaluate_actions(self, obs, actions):
    return self.actor_critic.evaluate_actions(obs, actions)
  
  def compute_curiosity_reward(self, current_obs, next_obs, actions):
    """Compute intrinsic curiosity reward"""
    current_features = self.icm.get_features(self.actor_critic, current_obs)
    next_features = self.icm.get_features(self.actor_critic, next_obs)
    
    intrinsic_reward = self.icm.compute_intrinsic_reward(
      current_features, next_features, actions
    )
    
    return intrinsic_reward
  
  def compute_icm_loss(self, current_obs, next_obs, dones, actions):
    """Compute ICM training loss"""

    proper_actions = []

    if isinstance(current_obs, dict):
      proper_obs = {key: [] for key in current_obs.keys()}
      proper_next_obs = {key: [] for key in next_obs.keys()}
      for i in range(len(current_obs['vector'])):
        if dones[i] == False:
          for key in current_obs.keys():
            proper_obs[key].append(current_obs[key][i])
            proper_next_obs[key].append(next_obs[key][i])

          proper_actions.append(actions[i])

      # Convert lists back to tensors
      proper_obs = {key: th.stack(proper_obs[key]) for key in proper_obs.keys()}
      proper_next_obs = {key: th.stack(proper_next_obs[key]) for key in proper_next_obs.keys()}
      proper_actions = th.stack(proper_actions)
    else:
      proper_obs = []
      proper_next_obs = []
      for i in range(len(current_obs)):
        if dones[i] == False:
          proper_obs.append(current_obs[i])
          proper_next_obs.append(next_obs[i])
          proper_actions.append(actions[i])

      # Convert lists back to tensors
      proper_obs = th.stack(proper_obs)
      proper_next_obs = th.stack(proper_next_obs)
      proper_actions = th.stack(proper_actions)
    
    current_features = self.icm.get_features(self.actor_critic, proper_obs)
    next_features = self.icm.get_features(self.actor_critic, proper_next_obs)

    return self.icm.compute_icm_loss(current_features, next_features, proper_actions) 