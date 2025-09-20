import time
import torch as th
import torch.optim as optim
import numpy as np
import random
import os
import sys

# FIX - remove file after rewriten by hand

# Add root directory to path to find config module
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.algorithms.RewardsNormalizer import RewardNormalizer
from src.utils.wandb_logger import WandBLogger

def create_optimizer_and_scheduler(param_groups, settings):
    
    weight_decay = settings.get('weight_decay', 1e-5)
    
    optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
    
    scheduler_step_size = settings.get('scheduler_step_size', 100)
    scheduler_gamma = settings.get('scheduler_gamma', 0.95)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    
    return optimizer, scheduler

# Generalized Advantage Estimation
class GAE:
    def __init__(self, gamma: float, lam: float):
        self.lam = lam
        self.gamma = gamma

    def compute_gae(self, rewards, values, dones) -> np.ndarray:
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            td_residual = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = td_residual + self.gamma * self.lam * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        return advantages

# Rollout Buffer
class RolloutBuffer:
    def __init__(self, device):
        self.device = device
        self.buffer = {
            'obs': [],
            'acts': [],
            'logps': [],
            'rews': [],
            'vals': [],
            'dones': []
        }

    def add(self, obs, act, logp, rew, val, done):
        self.buffer['obs'].append(obs)
        self.buffer['acts'].append(act)
        self.buffer['logps'].append(logp)
        self.buffer['rews'].append(rew)
        self.buffer['vals'].append(val)
        self.buffer['dones'].append(done)
    
    def get_data(self):
        """Extract all data from buffer as tensors"""
        data = {}
        for key in self.buffer.keys():
            if key == 'obs':
                # Handle dictionary observations
                if isinstance(self.buffer[key][0], dict):
                    data[key] = {}
                    for obs_key in self.buffer[key][0].keys():
                        data[key][obs_key] = th.stack([item[obs_key] for item in self.buffer[key]])
                else:
                    data[key] = th.stack(self.buffer[key])
            else:
                data[key] = th.tensor(self.buffer[key], dtype=th.float32, device=self.device)
        return data
    
    def clear(self):
        """Clear the buffer"""
        for key in self.buffer.keys():
            self.buffer[key] = []

# PPO Agent
class PPOAgent:
    def __init__(self, model_net, optimizer, settings, scheduler=None):
        self.settings = settings
        self.seed = settings.get('seed', 0)
        self.apply_seed(self.seed)

        self.device = settings['device']
        self.model = model_net.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Initialize GAE
        self.gae = GAE(
            gamma=settings.get('gamma', 0.99),
            lam=settings.get('lambda', 0.95)
        )

        # Initialize reward normalizer
        self.reward_normalizer = RewardNormalizer(
            gamma=settings.get('gamma', 0.99),
            epsilon=settings.get('reward_norm_epsilon', 1e-8)
        )

        # PPO specific settings
        self.clip_eps = settings.get('clip_eps', 0.2)
        self.value_clip_eps = settings.get('value_clip_eps', None)
        self.max_grad_norm = settings.get('max_grad_norm', 5.0)
        
        # Store other settings as instance variables for logging
        self.gamma = settings.get('gamma', 0.99)
        self.lambda_val = settings.get('lambda', 0.95)
        self.ppo_epochs = settings.get('ppo_epochs', 4)
        self.batch_size = settings.get('batch_size', 64)
        self.update_timesteps = settings.get('update_timesteps', 1024)
        self.val_loss_coef = settings.get('val_loss_coef', 0.5)
        self.ent_loss_coef = settings.get('ent_loss_coef', 0.01)
        
        # Initialize WandB Logger
        self.ppo_logger = None
        try:
            self.ppo_logger = WandBLogger(settings, self.seed)
        except Exception as e:
            print(f"Failed to initialize WandB Logger: {e}")
            raise e
          
        # Log hyperparameters (combine settings with seed)
        hyperparams = settings.copy()

        if self.ppo_logger is not None:
            self.ppo_logger.log_hyperparameters(hyperparams)

    def apply_seed(self, seed):
        """Set seeds for all random components to ensure reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

    def apply_seed_iteration(self, iteration):
        """Reset seed for each training iteration to ensure reproducibility"""
        seed = self.seed + iteration
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)

    # Returns: loss, policy_loss, value_loss, entropy_bonus
    def calculate_loss(self, mb_obs, mb_acts, mb_old_logps, mb_returns, mb_advantages, mb_old_values=None):
        logps, entropy, values = self.model.evaluate_actions(mb_obs, mb_acts)
        ratios = th.exp(logps - mb_old_logps)

        # Policy loss with clipping
        surr1 = ratios * mb_advantages
        surr2 = th.clamp(ratios, 1 - self.settings['clip_eps'], 1 + self.settings['clip_eps']) * mb_advantages
        policy_loss = -th.min(surr1, surr2).mean()

        # Value loss with optional clipping for stability
        if self.value_clip_eps is not None and mb_old_values is not None:
            # Clip new values relative to old values (SB3 approach)
            value_pred_clipped = mb_old_values + th.clamp(
                values - mb_old_values,
                -self.value_clip_eps,
                self.value_clip_eps
            )
            value_losses = (values - mb_returns) ** 2
            value_losses_clipped = (value_pred_clipped - mb_returns) ** 2
            value_loss = th.max(value_losses, value_losses_clipped).mean()
        else:
            # Standard MSE value loss
            value_loss = ((values - mb_returns)**2).mean()

        # Entropy loss
        entropy_bonus = entropy.mean()

        # Apply coefficients
        value_loss *= self.settings['val_loss_coef']
        entropy_bonus *= self.settings['ent_loss_coef']

        # Return combined loss for single optimizer
        total_loss = policy_loss + value_loss - entropy_bonus
        return total_loss, policy_loss, value_loss, entropy_bonus
    
    # Returns average loss of the batch
    def update(self, obs, acts, old_logps, returns, advantages, old_values=None, iteration=None):
        losses = {"total_loss": [], "policy_loss": [], "value_loss": [], "entropy_loss": []}
        
        # Capture parameters before optimization for change tracking
        if self.ppo_logger is not None:
            old_params = self.ppo_logger.capture_parameters(self.model)
      
        for epoch in range(self.settings['ppo_epochs']):
            # Get batch size based on observation type
            buffer_len = len(obs) if not isinstance(obs, dict) else len(list(obs.values())[0])
            
            # Use deterministic random state for batch sampling
            rng = np.random.RandomState(self.seed + epoch)
            idxs = rng.permutation(buffer_len)
            
            epoch_losses = {"total_loss": [], "policy_loss": [], "value_loss": [], "entropy_loss": []}
            
            print(f"Batch length: {buffer_len}, {self.settings['batch_size']}")
            for start in range(0, buffer_len, self.settings['batch_size']):
                end = start + self.settings['batch_size']
                print(f"Start: {start}, End: {end}")
                mb_idx = idxs[start:end]

                # Handle dictionary or tensor observations
                if isinstance(obs, dict):
                    mb_obs = {key: obs[key][mb_idx] for key in obs.keys()}
                else:
                    mb_obs = obs[mb_idx]
                    
                mb_acts = acts[mb_idx]
                mb_old_logps = old_logps[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_old_values = old_values[mb_idx] if old_values is not None else None

                # Calculate combined loss for both networks
                total_loss, policy_loss, value_loss, entropy_bonus = self.calculate_loss(
                    mb_obs, mb_acts, mb_old_logps, mb_returns, mb_advantages, mb_old_values
                )
                
                # Update both networks with single optimizer
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Log gradients before clipping (only on first epoch and first batch for interpretability)
                if self.ppo_logger is not None:
                  self.ppo_logger.log_gradients(self.model, iteration)
                
                # Add gradient clipping
                th.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                
                self.optimizer.step()
                
                epoch_losses["total_loss"].append(total_loss.item())
                epoch_losses["policy_loss"].append(policy_loss.item())
                epoch_losses["value_loss"].append(value_loss.item())
                epoch_losses["entropy_loss"].append(entropy_bonus.item())
                
            # Accumulate losses
            for key in losses.keys():
                losses[key].extend(epoch_losses[key])
        
        # Log parameter changes after optimization (only if iteration is provided)
        if iteration is not None and self.ppo_logger is not None:
            self.ppo_logger.log_parameter_changes(self.model, iteration, old_params)

        return losses
    
    def train(self, env, iterations, start_iteration=0):
        for iteration in range(start_iteration, start_iteration + iterations):
            time_start = time.time()
          
            # Reset seeds for this iteration to ensure reproducibility
            self.apply_seed_iteration(iteration)
            
            # Seed the environment for reproducibility
            obs, _ = env.reset(seed=self.seed + iteration)
            buffer = RolloutBuffer(self.device)

            # Don't reset reward normalizer - let it maintain running statistics
            ep_return = 0
            ep_returns = []
            ep_steps = []

            t = 0
            ep_t = 0
            while True:
                # Handle dictionary or tensor observations
                if isinstance(obs, dict):
                    # Convert each observation to tensor and add batch dimension
                    obs_tensor = {key: th.tensor(obs[key], dtype=th.float32, device=self.device) for key in obs.keys()}
                else:
                    obs_tensor = th.tensor(obs, dtype=th.float32, device=self.device)
                    
                action, logp, _, value = self.model.get_action(obs_tensor)
                
                next_obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated

                # Remove batch dimension from observation tensors for buffer storage
                if isinstance(obs, dict):
                    # Convert each observation to tensor and add batch dimension
                    obs_tensor = {key: obs_tensor[key].squeeze(0).clone().detach() for key in obs.keys()}
                else:
                    obs_tensor = obs_tensor.squeeze(0).clone().detach()

                # Store observations in buffer
                buffer.add(obs_tensor, action.item(), logp.item(), reward, value.item(), done)
                ep_return += reward
                obs = next_obs

                if done:
                    ep_returns.append(ep_return)
                    ep_return = 0
                    ep_steps.append(ep_t)
                    ep_t = 0
                    
                    obs, _ = env.reset(seed=self.seed + iteration)
                    if t >= self.settings['update_timesteps']:
                        break
                t += 1
                ep_t += 1

            # Extract data from buffer
            buffer_data = buffer.get_data()
            
            # Normalize rewards for better training stability
            rewards = buffer_data['rews'].cpu().numpy()
            print(rewards, type(rewards))
            normalized_rewards = self.reward_normalizer.normalize(rewards)
            
            values = buffer_data['vals'].cpu().numpy()
            dones = buffer_data['dones'].cpu().numpy()
            
            # Compute advantages using GAE with normalized rewards
            advantages = self.gae.compute_gae(normalized_rewards, values, dones)
            advantages = th.tensor(advantages, dtype=th.float32, device=self.device)
            
            # Compute returns: discounted cumulative rewards using normalized rewards
            returns = np.zeros_like(normalized_rewards)
            gamma = self.settings.get('gamma', 0.99)
            
            for t in reversed(range(len(normalized_rewards))):
                if t == len(normalized_rewards) - 1:
                    next_return = 0
                else:
                    next_return = returns[t + 1]
                returns[t] = normalized_rewards[t] + gamma * next_return * (1 - dones[t])
            
            returns = th.tensor(returns, dtype=th.float32, device=self.device)

            # Training step
            losses = self.update(
                buffer_data['obs'], 
                buffer_data['acts'], 
                buffer_data['logps'], 
                returns, 
                advantages,
                old_values=buffer_data['vals'],  # Pass old values for clipping
                iteration=iteration
            )
            
            # Clear buffer for next iteration
            buffer.clear()
            
            mean_losses = {key: np.mean(losses[key]) for key in losses}

            # Stats per real episode
            ep_steps_np = np.array(ep_steps)
            mean_steps = ep_steps_np.mean() if len(ep_steps_np) > 0 else 0.0
            std_steps = ep_steps_np.std(ddof=0) if len(ep_steps_np) > 0 else 0.0
            
            ep_returns_np = np.array(ep_returns)
            mean_return = ep_returns_np.mean() if len(ep_returns_np) > 0 else 0.0
            std_return = ep_returns_np.std(ddof=0) if len(ep_returns_np) > 0 else 0.0

            time_end = time.time()
            time_taken = time_end - time_start

            # Log weight distributions
            self.ppo_logger.log_weight_distributions(self.model, iteration)
            
            # Prepare metrics for logging
            training_metrics = {
                'mean_return': mean_return,
                'std_return': std_return,
                'mean_steps': mean_steps,
                'std_steps': std_steps,
                'time_taken': time_taken,
                'episodes_count': len(ep_returns)
            }
            
            # Log metrics
            self.ppo_logger.log_training_metrics(iteration, training_metrics)
            self.ppo_logger.log_losses(iteration, mean_losses)
            self.ppo_logger.log_learning_rates(iteration, self.optimizer)

            # Log current learning rates for console output
            current_lrs = [group['lr'] for group in self.optimizer.param_groups]
            
            # Console logging
            self.ppo_logger.log_console_training_summary(
                iteration, ep_returns, time_taken, std_return,
                mean_steps, std_steps, mean_losses, current_lrs
            )
            
            # Step the learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Close logger
        self.ppo_logger.close()