import time
import gymnasium as gym
import numpy as np
import torch as th
import random

from src.models.icm_utils import get_model_flattened_parameters
from src.algorithms.RewardsNormalizer import RewardNormalizer
from src.utils.wandb_logger import WandBLogger
from src.utils.seed_utils import set_all_seeds, set_training_iteration_seed
from src.models.intrinsic_curiosity_module import ActorCriticWithICM
from src.utils.evaluation import evaluate_policy

import torch.optim as optim

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
      
      if t == len(rewards) - 1:
          next_value = 0
      else:
          next_value = values[t+1]
      
      td_residual = rewards[t] + self.gamma * next_value * done_mask - values[t]
      advantages[t] = td_residual + self.gamma * self.lambda_ * last_advantage * done_mask
      last_advantage = advantages[t]
        
    return advantages
    
class RolloutBuffer:
  def __init__(self, device):
    self.device = device
    self.data = {
      'obs': [],
      'next_obs': [],
      'acts': [],
      'rews': [],
      'intrinsic_rews': [],
      'logprobs': [],
      'vals': [],
      'dones': []
    }
  
  def get_data(self):
    """Get all data as tensors"""
    # FIX - make all data fields the same type, either tensor or np.ndarray
    result = {}
    for key in self.data.keys():
      if key in ['obs', 'next_obs']:
        if isinstance(self.data[key][0], dict):
          result[key] = {}
          for obs_key in self.data[key][0].keys():
            result[key][obs_key] = th.stack([th.tensor(elem[obs_key], dtype=th.float32, device=self.device) for elem in self.data[key]] )
        else:
          result[key] = th.tensor(self.data[key], dtype=th.float32, device=self.device)
      else:
        result[key] = th.tensor(self.data[key], dtype=th.float32, device=self.device)
    return result
  
  def add(self, observation, next_observation, action, reward, intrinsic_reward, log_prob, value, done):
    self.data['obs'].append(observation)
    self.data['next_obs'].append(next_observation)
    self.data['acts'].append(action)
    self.data['rews'].append(reward)
    self.data['intrinsic_rews'].append(intrinsic_reward)
    self.data['logprobs'].append(log_prob)
    self.data['vals'].append(value)
    self.data['dones'].append(done)
    
  def clear(self):
    for key in self.data.keys():
      self.data[key] = []
    
  
class PPOAgent:
  def __init__(self, model, settings: dict, optimizer: th.optim.Optimizer, scheduler = None, start_iteration = 0, start_timestep = 0):
    self.settings = settings
    self.device = settings.get('device', 'cpu')
    
    # Runtime variables
    self.iteration = start_iteration
    self.total_timesteps = start_timestep
    
    # Seeding
    self.seed = settings.get('seed', 0)
    set_all_seeds(self.seed)
    
    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    
    self.gamma = settings.get('gamma', 0.99)
    self.icm_normalizer_gamma = settings.get('icm_normalizer_gamma', self.gamma)
    self.gae = GAE(
      gamma=self.gamma,
      lambda_=settings.get('gae_lambda', 0.95)
    )

    self.icm_loss_weight = settings.get('icm_loss_weight', None)
    
    # Reward normalizers
    self.extrinsic_normalizer = RewardNormalizer(
      gamma=self.gamma,
      epsilon=settings.get('reward_norm_epsilon', 1e-8)
    )
    self.intrinsic_normalizer = RewardNormalizer(
      gamma=self.icm_normalizer_gamma,
      epsilon=settings.get('reward_norm_epsilon', 1e-8)
    )
    
    # Scaling for intrinsic reward
    self.intrinsic_reward_scale = settings.get('intrinsic_reward_scale', 1.0)
    
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
    
    self.heatmap_logging_freq = settings.get('heatmap_logging_freq', 10)
    
    # Evaluation settings
    self.eval_freq = settings.get('eval_freq', None)  # Evaluate every K iterations (None = no evaluation during training)
    self.eval_episodes = settings.get('eval_episodes', 10)  # Number of episodes for evaluation
    self.eval_env_type = settings.get('eval_env_type', 'find')  # Environment type for outcome categorization ('find' or 'find_deliver')
    
    # Wandb logger
    self.logger = None
    
    try:
      self.logger = WandBLogger(self.settings, self.seed)
    except Exception as e:
      print(f'Logger failed to initialize: {e} It will be DISABLED for this run.')
      
    # Log hyperparams
    if self.logger is not None:
      self.logger.log_hyperparameters(settings)

  # Inspired by https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py#L226 SOURCE
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
  
  def update(self, obs, actions, old_logprobs, returns, advantages, old_values=None, dones=None, next_obs=None) -> dict:
    """
    Performs update of model after buffer data is gathered.
    
      Args:
        obs: observations from buffer
        actions: actions
        old_logprobs: log probabilities of actions taken from buffer 'old' refers to them being used as reference even after partial update of batches
        returns: discounted returns from buffer
        advantages: calculated advantages
        old_values: critic's values predictions taken from buffer 'old' refers to them being used as reference even after partial update of batches
        next_obs: next observations from buffer
    
      Returns: losses dictionary
    """
    # Put model into training mode
    self.model.train()
    
    old_params = None
    if self.logger is not None:
      named_model_params = get_model_flattened_parameters(self.model)
      old_params = self.logger.capture_parameters(named_model_params) # FIX - rename function
    
    losses = {'total': [], 'policy': [], 'value': [], 'entropy': []}
    if self.icm_loss_weight is not None and isinstance(self.model, ActorCriticWithICM):
      losses['inverse_loss'] = []
      losses['forward_loss'] = []
      losses['icm_loss'] = []

    for epoch in range(self.epochs):
      
      buffer_len = len(list(obs.values())[0]) if isinstance(obs, dict) else len(obs)
      
      rng = np.random.RandomState(self.seed + self.iteration * 1000 + epoch)
      indices = rng.permutation(buffer_len)
      
      # Batched update
      for batch_i, start  in enumerate(range(0, buffer_len, self.batch_size)):
        end = start + self.batch_size
        batch_indices = indices[start:end]
        
        if isinstance(obs, dict):
          batch_obs = {key: obs[key][batch_indices] for key in obs.keys()}
          batch_next_obs = {key: next_obs[key][batch_indices] for key in next_obs.keys()}
        else:
          batch_obs = obs[batch_indices]
          batch_next_obs = next_obs[batch_indices]
        
        batch_actions = actions[batch_indices]
        batch_old_logprobs = old_logprobs[batch_indices]
        batch_returns = returns[batch_indices]
        batch_advantages = advantages[batch_indices]
        batch_old_values = old_values[batch_indices]
        batch_dones = dones[batch_indices]
        
        total_loss, policy_loss, value_loss, entropy_loss = self.calculate_loss(batch_obs, batch_actions, batch_old_logprobs, batch_returns, batch_advantages, batch_old_values)
        
        icm_loss = 0.0
        inverse_loss = 0.0
        forward_loss = 0.0
        
        # Compute ICM loss if using ICM
        if self.icm_loss_weight is not None and isinstance(self.model, ActorCriticWithICM):
          # If model is an ActorCriticWithICM wrapper, use its compute_icm_loss
          icm_loss, inverse_loss, forward_loss = self.model.compute_icm_loss(batch_obs, batch_next_obs, dones=batch_dones, actions=batch_actions)
          total_loss += self.icm_loss_weight * icm_loss
          inverse_loss *= self.icm_loss_weight
          forward_loss *= self.icm_loss_weight
        
        self.optimizer.zero_grad()
        total_loss.backward()

        # Logs only for first batch (later batches may not represent gradients correctly as parameters will already change)
        if self.logger is not None and batch_i == 0 and epoch == 0:
          named_model_params = get_model_flattened_parameters(self.model)
          self.logger.log_gradients(named_model_params, iteration=self.iteration)
        
        # Gradient clipping
        th.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        
        self.optimizer.step()
        
        losses['total'].append(total_loss.item())
        losses['policy'].append(policy_loss.item())
        losses['value'].append(value_loss.item())
        losses['entropy'].append(entropy_loss.item())
        if self.icm_loss_weight is not None and isinstance(self.model, ActorCriticWithICM):
          losses['inverse_loss'].append(inverse_loss.item())
          losses['forward_loss'].append(forward_loss.item())
          losses['icm_loss'].append(icm_loss.item() * self.icm_loss_weight)
        

      
    if self.logger is not None and old_params is not None:
      named_model_params = get_model_flattened_parameters(self.model)
      self.logger.log_parameter_changes(named_model_params, self.iteration, old_params)
        
    return losses
  
  def compute_returns(self, rewards, dones):
    """Calculates discounted returns"""
    
    returns = np.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
      done_mask = 1 - dones[t] # To account for terminal states (mask effectively zeroes statistics)
      
      if(t == len(rewards) - 1):
        next_return = 0
      else:
        next_return = returns[t+1]
      
      returns[t] = rewards[t] + self.gamma * next_return * done_mask
      
    return th.tensor(returns, dtype=th.float32, device=self.device)
  
  def train(self, env: gym.Env, iterations, early_stopping_fn=None):
    """
    Train the PPO agent.
    
    Args:
      env: Gymnasium environment
      iterations: Number of training iterations
      early_stopping_fn: Optional callable that takes (iteration, metrics, losses) and returns bool.
                        If it returns True, training stops early.
                        Signature: fn(iteration: int, metrics: dict, losses: dict) -> bool
    """
    
    buffer = RolloutBuffer(self.device)
    
    start_iteration = self.iteration
    final_iteration = start_iteration + iterations - 1
    
    for i in range(start_iteration, start_iteration + iterations):
      self.iteration = i
      
      time_start = time.time()
      
      ep_return = 0 # returns of specific episode
      ep_intrinsic_return = 0 # intrinsic returns of specific episode
      ep_returns = [] # returns through episodes
      ep_steps = [] # steps of episodes
      ep_intrinsic_returns = [] # intrinsic returns through episodes
      obs, info = env.reset()
      
      step = 0
      steps_episode = 0

      log_heatmap_data = i % self.heatmap_logging_freq == 0 and self.logger is not None
      heatmap_data = None

      if log_heatmap_data and 'map_position' in info and 'forward_direction' in info:
        heatmap_data = {
          'map_position': [],
          'forward_direction': [],
        }
        heatmap_data['map_position'].append(info['map_position'])
        heatmap_data['forward_direction'].append(info['forward_direction'])

      while True:
        if isinstance(obs, dict):
          obs_tensor = {obs_key: th.tensor(obs[obs_key], dtype=th.float32, device=self.device)
                        for obs_key in obs.keys()}
        else:
          obs_tensor = th.tensor(obs, dtype=th.float32, device=self.device)
        
        action, logprob, _, value = self.model.get_action(obs_tensor)
        next_obs, reward, truncated, terminated, info = env.step(action.item())

        if heatmap_data is not None and 'map_position' in info and 'forward_direction' in info:
          heatmap_data['map_position'].append(info['map_position'])
          heatmap_data['forward_direction'].append(info['forward_direction'])

        done = truncated or terminated
        
        # Process observations for buffer storage
        if isinstance(obs, dict):
          # For dict observations (multimodal), remove batch dimension if present
          obs_for_buffer = {key: obs[key].squeeze(0) if obs[key].ndim > 1 and obs[key].shape[0] == 1 else obs[key] 
                           for key in obs.keys()}
          next_obs_for_buffer = {key: next_obs[key].squeeze(0) if next_obs[key].ndim > 1 and next_obs[key].shape[0] == 1 else next_obs[key] 
                                for key in next_obs.keys()}
        else:
          # For vector observations, remove batch dimension if present (Unity ML-Agents adds it, gymnasium doesn't)
          obs_for_buffer = obs.squeeze(0) if obs.ndim > 1 and obs.shape[0] == 1 else obs
          next_obs_for_buffer = next_obs.squeeze(0) if next_obs.ndim > 1 and next_obs.shape[0] == 1 else next_obs
        
        # Compute intrinsic reward if using ICM
        intrinsic_reward = 0.0
        if self.icm_loss_weight is not None and isinstance(self.model, ActorCriticWithICM):
          intrinsic_reward = self.model.compute_curiosity_reward(obs_tensor, 
            {key: th.tensor(next_obs[key], dtype=th.float32, device=self.device) for key in next_obs.keys()} if isinstance(next_obs, dict) else th.tensor(next_obs, dtype=th.float32, device=self.device),
            action)
          intrinsic_reward = intrinsic_reward.item()
        
        buffer.add(obs_for_buffer, next_obs_for_buffer, action, reward, intrinsic_reward, logprob, value, done)
        obs = next_obs
        
        ep_return += reward
        ep_intrinsic_return += intrinsic_reward
        
        if done:
          
          obs, _ = env.reset()
          
          ep_steps.append(steps_episode)
          ep_returns.append(ep_return)
          ep_intrinsic_returns.append(ep_intrinsic_return)
          ep_return = 0
          ep_intrinsic_return = 0
          steps_episode = 0
          if step >= self.buffer_size:
            break
        
        step += 1
        steps_episode += 1
    
      if heatmap_data is not None:
        self.logger.log_heatmap_data(i, np.array(heatmap_data['map_position']), "map_position",
                                     title=f"Agent Visit Frequency (iteration {i})",
                                     x_label="X position",
                                     y_label="Y position",
                                     bounds=(-5, 5),
                                     buckets=10)
        self.logger.log_heatmap_data(i, np.array(heatmap_data['forward_direction']), "forward_direction",
                                     title=f"Agent Direction Frequency (iteration {i})",
                                     x_label="X position component",
                                     y_label="Y direction component",
                                     bounds=(-1, 1),
                                     buckets=20)

      # Update total timesteps
      self.total_timesteps += step
      
      buffer_data = buffer.get_data()
      
      np_rewards = buffer_data['rews'].cpu().numpy()
      np_intrinsic_rewards = buffer_data['intrinsic_rews'].cpu().numpy()
      
      values = buffer_data['vals']
      dones = buffer_data['dones']
      
      actions = buffer_data['acts']
      old_logprobs = buffer_data['logprobs']
      
      norm_extrinsic = self.extrinsic_normalizer.normalize(np_rewards)
      norm_intrinsic = self.intrinsic_normalizer.normalize(np_intrinsic_rewards)

      # Scale intrinsic before combining
      normalized_rewards = norm_extrinsic + self.intrinsic_reward_scale * norm_intrinsic

      returns = self.compute_returns(normalized_rewards, dones)
      
      advantages = self.gae.compute_gae(normalized_rewards, values, dones)
      advantages = th.tensor(advantages, dtype=th.float32, device=self.device)
      
      # FIX: Add advantage normalization (if enabled in settings)
      if self.settings.get('normalize_advantages', False):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
      
      observations = buffer_data['obs']
      next_observations = buffer_data['next_obs']

      losses = self.update(observations, actions, old_logprobs, returns, advantages, old_values=values, dones=dones, next_obs=next_observations)
      
      # Update scheduler
      if self.scheduler is not None:
        self.scheduler.step()
      
      buffer.clear()
      
      # Logging
      time_end = time.time()
      time_delta = time_end - time_start
      
      np_ep_returns = np.array(ep_returns)
      np_ep_intrinsic_returns = np.array(ep_intrinsic_returns)
      np_ep_steps = np.array(ep_steps)
      metrics = {
        'mean_return': np_ep_returns.mean(),
        'mean_intrinsic_return': np_ep_intrinsic_returns.mean(),
        'std_return': np_ep_returns.std(),
        'std_intrinsic_return': np_ep_intrinsic_returns.std(),
        'mean_steps': np_ep_steps.mean(),
        'std_steps': np_ep_steps.std(),
        'time_taken': time_delta,
        'episodes_count': len(ep_returns),
        'total_timesteps': self.total_timesteps
      }
      if self.logger is not None:
        self.logger.log_training_metrics(i, metrics)
      
      mean_losses = {key: np.mean(losses[key]) for key in losses.keys()}
      if self.logger is not None:
        self.logger.log_losses(i, mean_losses)
        self.logger.log_learning_rates(i, self.optimizer)

        named_model_params = get_model_flattened_parameters(self.model)
        self.logger.log_weight_distributions(named_model_params, i)
      
      learning_rates = [group['lr'] for group in self.optimizer.param_groups]
      if self.logger is not None:
        self.logger.log_console_training_summary(i, np.array(ep_returns), time_delta, np.array(ep_steps), losses, learning_rates, np_ep_intrinsic_returns)
      
      # Periodic evaluation
      if self.eval_freq is not None and (i + 1) % self.eval_freq == 0:
        print(f"\n=== EVALUATING AT ITERATION {i} ===")
        eval_start = time.time()
        
        # Determine observation type from environment
        obs_type = "multimodal" if isinstance(obs, dict) else "vector"
        
        # Run deterministic evaluation
        eval_mean_return, eval_std_return, eval_mean_steps, eval_std_steps, eval_returns, eval_steps, eval_outcomes = evaluate_policy(
          self.model, env, self.device, 
          num_episodes=self.eval_episodes, 
          seed=self.seed,
          obs_type=obs_type,
          verbose=False,
          env_type=self.eval_env_type
        )
        
        eval_time = time.time() - eval_start
        
        # Log evaluation metrics
        eval_metrics = {
          'mean_return': eval_mean_return,
          'std_return': eval_std_return,
          'mean_steps': eval_mean_steps,
          'std_steps': eval_std_steps,
          'time_taken': eval_time,
          'num_episodes': self.eval_episodes,
          'total_timesteps': self.total_timesteps
        }
        
        if self.logger is not None:
          self.logger.log_evaluation_metrics(i, eval_metrics)
          self.logger.log_evaluation_outcomes(i, eval_outcomes, self.eval_episodes)
        
        print(f"Evaluation: Mean return = {eval_mean_return:.2f} +- {eval_std_return:.2f}, Mean steps = {eval_mean_steps:.2f} +- {eval_std_steps:.2f}, Time = {eval_time:.2f}s")
      
        # Check early stopping condition
        if early_stopping_fn is not None:
          should_stop = early_stopping_fn('mean_evaluation_return', eval_mean_return)
          if should_stop:
            print(f"\n=== EARLY STOPPING TRIGGERED AT ITERATION {i} ===")
            if self.logger is not None:
              self.logger.log_event(i, "early_stopping_triggered")
            break
    
    # Final evaluation after training completes (if not already done)
    if self.eval_freq is not None:
      # Check if final iteration would trigger a periodic eval
      # Periodic eval triggers when (i + 1) % eval_freq == 0
      # So for final_iteration, it triggers when (final_iteration + 1) % eval_freq == 0
      if (final_iteration + 1) % self.eval_freq != 0:
        print(f"\n=== FINAL EVALUATION AT ITERATION {final_iteration} ===")
        eval_start = time.time()
        
        # Determine observation type from environment
        obs_type = "multimodal" if isinstance(obs, dict) else "vector"
        
        # Run deterministic evaluation
        eval_mean_return, eval_std_return, eval_mean_steps, eval_std_steps, eval_returns, eval_steps, eval_outcomes = evaluate_policy(
          self.model, env, self.device, 
          num_episodes=self.eval_episodes, 
          seed=self.seed,
          obs_type=obs_type,
          verbose=False,
          env_type=self.eval_env_type
        )
        
        eval_time = time.time() - eval_start
        
        # Log evaluation metrics
        eval_metrics = {
          'mean_return': eval_mean_return,
          'std_return': eval_std_return,
          'mean_steps': eval_mean_steps,
          'std_steps': eval_std_steps,
          'time_taken': eval_time,
          'num_episodes': self.eval_episodes,
          'total_timesteps': self.total_timesteps
        }
        
        if self.logger is not None:
          self.logger.log_evaluation_metrics(final_iteration, eval_metrics)
          self.logger.log_evaluation_outcomes(final_iteration, eval_outcomes, self.eval_episodes)
        
        print(f"Final Evaluation: Mean return = {eval_mean_return:.2f} +- {eval_std_return:.2f}, Mean steps = {eval_mean_steps:.2f} +- {eval_std_steps:.2f}, Time = {eval_time:.2f}s")