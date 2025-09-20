import time
import gymnasium as gym
import numpy as np
import torch as th
import random

from src.algorithms.RewardsNormalizer import RewardNormalizer
from src.utils.wandb_logger import WandBLogger

import torch.optim as optim

# FIX - rewrite
def create_optimizer_and_lr_scheduler(param_groups, weight_decay=1e-5, scheduler_step_size=None, scheduler_gamma=None):
  
  optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
  
  scheduler = None
  if scheduler_step_size is not None and scheduler_gamma is not None:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
  
  return optimizer, scheduler
  
# Dev note, to whole content of this module: use numpy.ndarray where possible (storing statistics etc.) and torch.tensors only for pytorch networks computations
class GAE:
  def __init__(self, gamma, lambda_):
    
    self.gamma = gamma
    self.lambda_ = lambda_
    
  
  def compute_gae(self, rewards, values, dones):
    """
    Following the GAE paper: https://arxiv.org/pdf/1506.02438 (formulas 11 to 16)
    + it is masked to account for 'done' state (inspired by https://nn.labml.ai/rl/ppo/gae.html)

    Args:
      rewards: rewards of the batch
      values: value estimates
      dones: done states (1 if state was terminal, 0 otherwise)
    """
    
    advantages = np.zeros_like(rewards)
    last_advantage = 0
    last_value = values[-1]
    
    for t in reversed(range(len(rewards))):
      done_mask = 1 - dones[t] # To account for terminal states (mask effectively zeroes statistics) 
      
      if(t == len(rewards) - 1):
        last_value = 0
      else:
        last_value = values[t+1]
      
      td_residual = rewards[t] + self.gamma * last_value * done_mask - values[t]
      advantages[t] = td_residual + self.gamma * self.lambda_ * last_advantage * done_mask
      last_advantage = advantages[t]
      
    return advantages
  
class RolloutBuffer:
  def __init__(self, device):
    self.device = device
    self.data = {
      'obs': [],
      'acts': [],
      'rews': [],
      'logprobs': [],
      'vals': [],
      'dones': []
    }
  
  def get_data(self):
    """Get all data as tensors"""
    # FIX - make all data fields the same type, either tensor or np.ndarray
    result = {}
    for key in self.data.keys():
      if key == 'obs':
        if isinstance(self.data['obs'][0], dict):
          result['obs'] = {}
          for obs_key in self.data['obs'][0].keys():
            result['obs'][obs_key] = th.stack([th.tensor(elem[obs_key], dtype=th.float32, device=self.device) for elem in self.data['obs']] )
      else:
        result[key] = th.tensor(self.data[key], dtype=th.float32, device=self.device)

      # print(f'{key}: {result[key]}')
    return result
  
  def add(self, observation, action, reward, log_prob, value, done):
    self.data['obs'].append(observation)
    self.data['acts'].append(action)
    self.data['rews'].append(reward)
    self.data['logprobs'].append(log_prob)
    self.data['vals'].append(value)
    self.data['dones'].append(done)
    
  def clear(self):
    for key in self.data.keys():
      self.data[key] = []
    
  
class PPOAgent:
  def __init__(self, model, settings: dict, optimizer: th.optim.Optimizer, scheduler = None, start_iteration = 0):
    self.settings = settings
    self.device = settings.get('device', 'cpu')
    
    # Runtime variables
    self.iteration = start_iteration
    
    # Seeding
    self.seed = settings.get('seed', 0)
    self.apply_seed()
    
    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    
    self.gamma = settings.get('gamma', 0.99)
    self.gae = GAE(
      gamma=self.gamma,
      lambda_=settings.get('gae_lambda', 0.95)
    )
    
    # Reward normalizer
    self.reward_normalizer = RewardNormalizer(
      gamma=self.gamma,
      epsilon=settings.get('reward_norm_epsilon', 1e-8)
    )
    
    # PPO algorithm settings
    self.clip_eps = settings.get('clip_eps', 0.2)
    # value_clip_eps is an absolute margin (not relative as clip_eps) which regulates how far can critic's value predictions change from the previous iteration
    self.value_clip_eps = settings.get('value_clip_eps', 0.2)
    self.max_grad_norm = settings.get('max_grad_norm', 0.5)
    
    self.batch_size = settings.get('batch_size', 128)
    self.buffer_size = settings.get('buffer_size', 1024)
    self.epochs = settings.get('epochs', 4)
    
    self.loss_val_coef = settings.get('loss_val_coef', 0.5)
    self.loss_entr_coef = settings.get('loss_entr_coef', 0.01)
    
    # Wandb logger
    self.logger = None
    
    try:
      self.logger = WandBLogger(self.settings, self.seed)
    except Exception as e:
      print(f'Logger failed to initialize: {e} It will be DISABLED for this run.')
      
    # Log hyperparams
    if self.logger is not None:
      self.logger.log_hyperparameters(settings)
    
  # Seeding function https://docs.pytorch.org/docs/stable/notes/randomness.html
  def apply_seed(self):
    final_seed = self.seed + self.iteration
    random.seed(final_seed)
    np.random.seed(final_seed)

  # Inspired by https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py ()
  def calculate_loss(self, obs, actions, old_logprobs, returns, advantages, old_values=None):
    logps, entropy, values_pred = self.model.evaluate_actions(obs, actions)
    # Clipped policy loss
    
    ratios = th.exp(logps - old_logprobs)

    # Like in original PPO paper
    policy_loss = -th.min(ratios * advantages, th.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages).mean()
    
    # Clip value to reduce critic's training variability (introduced in OpenAI's baselines and SB3))
    value_loss_unclipped = th.mean(((values_pred - returns)**2))
    
    # th.max at the end to choose more pesimistic scenario (analogically to PPO objective/policy loss)
    if self.value_clip_eps is not None and old_values is not None:
      value_pred_clipped = old_values + th.clip(values_pred - old_values, -self.value_clip_eps, self.value_clip_eps)
      value_loss_clipped = th.mean(((value_pred_clipped - returns)**2))
      value_loss = th.max(value_loss_unclipped, value_loss_clipped)
    else:
      value_loss = value_loss_unclipped
      
    value_loss *= self.loss_val_coef
    
    entropy_loss = entropy.mean() * self.loss_entr_coef
    
    total = policy_loss + value_loss - entropy_loss
    
    return total, policy_loss, value_loss, entropy_loss
  
  def update(self, obs, actions, old_logprobs, returns, advantages, old_values=None) -> dict:
    """
    Performs update of model after buffer data is gathered.
    
      Args:
        obs: observations from buffer
        actions: actions
        old_logprobs: log probabilities of actions taken from buffer 'old' refers to them being used as reference even after partial update of batches
        returns: discounted returns from buffer
        advantages: calculated advantages
        old_values: critic's values predictions taken from buffer 'old' refers to them being used as reference even after partial update of batches
    
      Returns: losses dictionary
    """
    old_params = None
    if self.logger is not None:
      old_params = self.logger.capture_parameters(self.model) # FIX - rename function
    
    losses = {'total': [], 'policy': [], 'value': [], 'entropy': []}

    for epoch in range(self.epochs):
      # List to convert from dict_values type to subscriptable
      buffer_len = len(list(obs.values())[0]) if isinstance(obs, dict) else len(obs)
      
      rng = np.random.RandomState(self.seed + self.iteration * epoch)
      indices = rng.permutation(buffer_len)
      
      # Batched update
      for batch_i, start  in enumerate(range(0, buffer_len, self.batch_size)):
        end = start + self.batch_size
        batch_indices = indices[start:end]
        
        if isinstance(obs, dict):
          batch_obs = {key: obs[key][batch_indices] for key in obs.keys()}
        else:
          batch_obs = obs[batch_indices]
        
        batch_actions = actions[batch_indices]
        batch_old_logprobs = old_logprobs[batch_indices]
        batch_returns = returns[batch_indices]
        batch_advantages = advantages[batch_indices]
        batch_old_values = old_values[batch_indices]
        
        total_loss, policy_loss, value_loss, entropy_loss = self.calculate_loss(batch_obs, batch_actions, batch_old_logprobs, batch_returns, batch_advantages, batch_old_values)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        th.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        
        self.optimizer.step()
        
        losses['total'].append(total_loss.item())
        losses['policy'].append(policy_loss.item())
        losses['value'].append(value_loss.item())
        losses['entropy'].append(entropy_loss.item())
        
        # Logs only for first batch (later batches may not represent gradients correctly as parameters will already change)
        if self.logger is not None and batch_i == 0:
          self.logger.log_gradients(self.model, iteration=self.iteration)
      
    if self.logger is not None and old_params is not None:
      self.logger.log_parameter_changes(self.model, self.iteration, old_params)
        
    return losses
  
  def compute_returns(self, rewards, dones):
    """Calculates discounted returns"""
    
    returns = np.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
      done_mask = 1 - dones[t] # To account for terminal states (mask effectively zeroes statistics)
      
      if(t == len(rewards) - 1):
        last_return = 0
      else:
        last_return = returns[t+1]
      
      returns[t] = rewards[t] + self.gamma * last_return * done_mask
      
    return th.tensor(returns, dtype=th.float32, device=self.device)
  
  def train(self, env: gym.Env, iterations):
    
    buffer = RolloutBuffer(self.device)
    
    start_iteration = self.iteration
    
    for i in range(start_iteration, start_iteration + iterations):
      self.iteration = i
      
      self.apply_seed()
      
      time_start = time.time()
      
      obs, _ = env.reset()
      
      ep_return = 0 # returns of specific episode
      ep_returns = [] # returns through episodes
      ep_steps = [] # steps of episodes
      
      step = 0
      steps_episode = 0
      while True:
        if isinstance(obs, dict):
          obs_tensor = {obs_key: th.tensor(obs[obs_key], dtype=th.float32, device=self.device)
                        for obs_key in obs.keys()}
        else:
          obs_tensor = th.tensor(obs, dtype=th.float32, device=self.device)
        
        action, logprob, _, value = self.model.get_action(obs_tensor)
        next_obs, reward, truncated, terminated, _ = env.step(action.item())
        
        done = truncated or terminated
        # Remove batch dimention of 1 from obs to add to buffer
        obs = {key: obs[key].squeeze(0) for key in obs.keys()}
        
        buffer.add(obs, action, reward, logprob, value, done)
        obs = next_obs
        
        ep_return += reward
        
        if done:
          obs, _ = env.reset()
          
          ep_steps.append(steps_episode)
          ep_returns.append(ep_return)
          ep_return = 0
          steps_episode = 0
          if step >= self.buffer_size:
            break
        
        step += 1
        steps_episode += 1
    
      buffer_data = buffer.get_data()
      
      rewards = buffer_data['rews'].cpu().numpy()
      values = buffer_data['vals']
      dones = buffer_data['dones']
      actions = buffer_data['acts']
      old_logprobs = buffer_data['logprobs']
      
      normalized_rewards = self.reward_normalizer.normalize(rewards)
      
      returns = self.compute_returns(normalized_rewards, dones)
      
      advantages = self.gae.compute_gae(normalized_rewards, values, dones)
      advantages = th.tensor(advantages, dtype=th.float32, device=self.device)
      
      observations = buffer_data['obs']

      losses = self.update(observations, actions, old_logprobs, returns, advantages, old_values=values)
      
      # Update scheduler
      if self.scheduler is not None:
        self.scheduler.step()
      
      buffer.clear()
      
      # Logging
      time_end = time.time()
      time_delta = time_end - time_start
      
      np_ep_returns = np.array(ep_returns)
      np_ep_steps = np.array(ep_steps)
      metrics = {
        'mean_return': np_ep_returns.mean(),
        'std_returns': np_ep_returns.std(),
        'mean_steps': np_ep_steps.mean(),
        'std_steps': np_ep_steps.std(),
        'time_taken': time_delta,
        'episodes_count': len(ep_returns)
      }
      self.logger.log_training_metrics(i, metrics)
      
      mean_losses = {key: np.mean(losses[key]) for key in losses.keys()}
      self.logger.log_losses(i, mean_losses)
      self.logger.log_learning_rates(i, self.optimizer)
      self.logger.log_weight_distributions(self.model, i)
      
      learning_rates = [group['lr'] for group in self.optimizer.param_groups]
      self.logger.log_console_training_summary(i, np.array(ep_returns), time_delta, np.array(ep_steps), losses, learning_rates)
      
    if self.logger is not None:
      self.logger.close()