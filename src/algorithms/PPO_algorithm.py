import time
import torch as th
import torch.optim as optim
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from collections import defaultdict

# Add root directory to path to find config module
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.algorithms.RewardsNormalizer import RewardNormalizer

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
    def __init__(self, model_net, settings, seed=0):
        self.settings = settings
        self.device = settings['device']
        self.model = model_net.to(self.device)
        self.weight_decay = settings.get('weight_decay', 1e-5)
        
        # Create parameter groups with different learning rates
        general_lr = settings['lr']
        visual_lr = settings.get('visual_lr', general_lr)
        vector_lr = settings.get('vector_lr', general_lr)
        
        # Group parameters by component
        visual_params = list(self.model.visual_encoder_cnn.parameters()) + list(self.model.visual_encoder_mlp.parameters())
        vector_params = list(self.model.vector_encoder.parameters())
        general_params = list(self.model.policy_net.parameters()) + list(self.model.value_net.parameters())
        
        # Create optimizer with parameter groups
        param_groups = [
            {'params': visual_params, 'lr': visual_lr, 'name': 'visual_encoder'},
            {'params': vector_params, 'lr': vector_lr, 'name': 'vector_encoder'},
            {'params': general_params, 'lr': general_lr, 'name': 'policy_value'}
        ]
        
        self.optimizer = optim.Adam(param_groups, weight_decay=self.weight_decay)
        
        # Add learning rate scheduler for gradual decay
        scheduler_step_size = settings.get('scheduler_step_size', 100)
        scheduler_gamma = settings.get('scheduler_gamma', 0.95)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        # Set seeds for reproducibility
        self.set_seed(seed)

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
        self.max_grad_norm = settings.get('max_grad_norm', 5.0)  # Updated default to match new clipping
        
        # Store other settings as instance variables for logging
        self.learning_rate = settings.get('lr', 3e-4)
        self.gamma = settings.get('gamma', 0.99)
        self.lambda_val = settings.get('lambda', 0.95)
        self.ppo_epochs = settings.get('ppo_epochs', 4)
        self.batch_size = settings.get('batch_size', 64)
        self.update_timesteps = settings.get('update_timesteps', 1024)
        self.val_loss_coef = settings.get('val_loss_coef', 0.5)
        self.ent_loss_coef = settings.get('ent_loss_coef', 0.01)
        self.gate_loss_coef = settings.get('gate_loss_coef', 0.0)
        
        # Store the seed for later use
        self.seed = seed
        
        # Initialize TensorBoard logger
        self.logger = None
        if settings.get('use_tensorboard', False):
            log_dir = settings.get('tensorboard_log_dir', 'runs/ppo_training')
            experiment_name = settings.get('experiment_name', f'ppo_seed_{seed}')
            full_log_dir = os.path.join(log_dir, experiment_name)
            self.logger = SummaryWriter(full_log_dir)
            
            # Log hyperparameters
            self._log_hyperparameters()

    def _log_hyperparameters(self):
        """Log hyperparameters to TensorBoard"""
        if self.logger is None:
            return
            
        # Create a dictionary of hyperparameters to log
        hparams = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'lambda': self.lambda_val,
            'clip_eps': self.clip_eps,
            'ppo_epochs': self.ppo_epochs,
            'batch_size': self.batch_size,
            'update_timesteps': self.update_timesteps,
            'val_loss_coef': self.val_loss_coef,
            'ent_loss_coef': self.ent_loss_coef,
            'max_grad_norm': self.max_grad_norm,
            'seed': self.seed
        }
        
        # Log hyperparameters
        self.logger.add_hparams(hparams, {})

    def _log_weight_distributions(self, iteration):
        """Log weight distributions for different parts of the model"""
            
        # Define model parts to log
        model_parts = {
            'visual_encoder_cnn': self.model.visual_encoder_cnn,
            'visual_encoder_mlp': self.model.visual_encoder_mlp,
            'vector_encoder': self.model.vector_encoder,
            'policy_net': self.model.policy_net,
            'value_net': self.model.value_net
        }
        
        print(f"  Weight distributions:")
        for part_name, part_module in model_parts.items():
            weights = []
            for param in part_module.parameters():
                if param.requires_grad:
                    weights.extend(param.data.cpu().numpy().flatten())
            
            if weights:
                weights = np.array(weights)
                if self.logger is not None:
                    self.logger.add_histogram(f'Weights/{part_name}', weights, iteration)
                    self.logger.add_scalar(f'Weights/{part_name}_norm', np.linalg.norm(weights), iteration)
                    self.logger.add_scalar(f'Weights/{part_name}_min', np.min(weights), iteration)
                    self.logger.add_scalar(f'Weights/{part_name}_max', np.max(weights), iteration)
                
                # Print weight summary
                print(f"    {part_name}: norm={np.linalg.norm(weights):.6f}, range=[{np.min(weights):.4f}, {np.max(weights):.4f}]")

    def _log_gradient_distributions(self, iteration):
        """Log gradient distributions for different parts of the model"""
        
        # Define model parts to log
        model_parts = {
            'visual_encoder_cnn': self.model.visual_encoder_cnn,
            'visual_encoder_mlp': self.model.visual_encoder_mlp,
            'vector_encoder': self.model.vector_encoder,
            'policy_net': self.model.policy_net,
            'value_net': self.model.value_net
        }
        
        print(f"  Gradient distributions:")
        for part_name, part_module in model_parts.items():
            gradients = []
            for param in part_module.parameters():
                if param.requires_grad and param.grad is not None:
                    gradients.extend(param.grad.data.cpu().numpy().flatten())
            
            if gradients:
                gradients = np.array(gradients)
                if self.logger is not None:
                    self.logger.add_histogram(f'Gradients/{part_name}', gradients, iteration)
                    self.logger.add_scalar(f'Gradients/{part_name}_norm', np.linalg.norm(gradients), iteration)
                    self.logger.add_scalar(f'Gradients/{part_name}_min', np.min(gradients), iteration)
                    self.logger.add_scalar(f'Gradients/{part_name}_max', np.max(gradients), iteration)
                
                # Print gradient summary
                print(f"    {part_name}: norm={np.linalg.norm(gradients):.6f}, range=[{np.min(gradients):.6f}, {np.max(gradients):.6f}]")
            else:
                print(f"    {part_name}: NO GRADIENTS (not being updated!)")

    def _log_parameter_changes(self, iteration, old_params):
        """Log parameter changes after optimization step"""
        
        # Define model parts to log
        model_parts = {
            'visual_encoder_cnn': self.model.visual_encoder_cnn,
            'visual_encoder_mlp': self.model.visual_encoder_mlp,
            'vector_encoder': self.model.vector_encoder,
            'policy_net': self.model.policy_net,
            'value_net': self.model.value_net
        }
        
        print(f"  Parameter changes:")
        for part_name, part_module in model_parts.items():
            changes = []
            param_idx = 0
            for param in part_module.parameters():
                if param.requires_grad:
                    old_param = old_params[part_name][param_idx]
                    change = (param.data.cpu().numpy() - old_param).flatten()
                    changes.extend(change)
                    param_idx += 1
            
            if changes:
                changes = np.array(changes)
                if self.logger is not None:
                    self.logger.add_histogram(f'ParameterChanges/{part_name}', changes, iteration)
                    self.logger.add_scalar(f'ParameterChanges/{part_name}_avg', np.mean(np.abs(changes)), iteration)
                    self.logger.add_scalar(f'ParameterChanges/{part_name}_min', np.min(changes), iteration)
                    self.logger.add_scalar(f'ParameterChanges/{part_name}_max', np.max(changes), iteration)
                
                # Print parameter change summary
                print(f"    {part_name}: avg={np.mean(np.abs(changes)):.6f}, range=[{np.min(changes):.6f}, {np.max(changes):.6f}]")
            else:
                print(f"    {part_name}: NO CHANGES (not being updated!)")

    def _capture_parameters(self):
        """Capture current parameter values for change tracking"""
        model_parts = {
            'visual_encoder_cnn': self.model.visual_encoder_cnn,
            'visual_encoder_mlp': self.model.visual_encoder_mlp,
            'vector_encoder': self.model.vector_encoder,
            'policy_net': self.model.policy_net,
            'value_net': self.model.value_net
        }
        
        old_params = {}
        for part_name, part_module in model_parts.items():
            old_params[part_name] = []
            for param in part_module.parameters():
                if param.requires_grad:
                    old_params[part_name].append(param.data.cpu().numpy().copy())
        
        return old_params

    def _log_action_distribution_buffer(self, actions, iteration):
        """Log action distribution statistics for the whole buffer"""
        
        # Count action frequencies
        action_counts = defaultdict(int)
        for action in actions:
            action_counts[action] += 1
            
        if self.logger is not None:
            # Log action distribution as histogram
            action_array = np.array(actions)
            self.logger.add_histogram(f'Actions/Buffer_Distribution', action_array, iteration)
            
            # Log action frequencies
            for action, count in action_counts.items():
                self.logger.add_scalar(f'Actions/Buffer_Action_{action}_Count', count, iteration)
                self.logger.add_scalar(f'Actions/Buffer_Action_{action}_Frequency', count/len(actions), iteration)
        
        # Print action distribution summary
        action_summary = ", ".join([f"{count/len(actions)*100:.1f}%" for action, count in sorted(action_counts.items())])
        print(f"  Actions distribution: {action_summary}")

    def set_seed(self, seed):
        """Set seeds for all random components to ensure reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

    def reset_seed_for_iteration(self, iteration):
        """Reset seed for each training iteration to ensure reproducibility"""
        seed = self.settings.get('seed', 0) + iteration
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)

    # Returns: loss, policy_loss, value_loss, entropy_bonus
    def calculate_loss(self, mb_obs, mb_acts, mb_old_logps, mb_returns, mb_advantages):
        logps, entropy, values = self.model.evaluate_actions(mb_obs, mb_acts)
        ratios = th.exp(logps - mb_old_logps)

        # Policy loss with clipping
        surr1 = ratios * mb_advantages
        surr2 = th.clamp(ratios, 1 - self.settings['clip_eps'], 1 + self.settings['clip_eps']) * mb_advantages
        policy_loss = -th.min(surr1, surr2).mean()

        # Value loss
        value_loss = ((values - mb_returns)**2).mean()
        # print("value loss mean: ", value_loss.mean())

        # Entropy loss
        entropy_bonus = entropy.mean()

        gate_loss = th.tensor(0.0, device=self.device, requires_grad=True)
        # Gate loss: penalize gate values too close to 0 or 1 (encourage values near 0.5)
        if self.gate_loss_coef > 0.0:
            gate = self.model.get_current_gate()
            if gate is not None:
                # gate should be in [0, 1], penalize if too close to 0 or 1
                # Use (gate - 0.5)^2, mean over batch
                gate_loss = self.gate_loss_coef * ((gate - 0.5) ** 2).mean()

        # Apply coefficients
        value_loss *= self.settings['val_loss_coef']
        entropy_bonus *= self.settings['ent_loss_coef']

        # Return combined loss for single optimizer
        total_loss = policy_loss + value_loss - entropy_bonus + gate_loss
        return total_loss, policy_loss, value_loss, entropy_bonus, gate_loss
    
    # Returns average loss of the batch
    def update(self, obs, acts, old_logps, returns, advantages, iteration=None):
        losses = {"total_loss": [], "policy_loss": [], "value_loss": [], "entropy_loss": [], "gate_loss": []}
        
        # Capture parameters before optimization for change tracking
        old_params = self._capture_parameters()
      
        for epoch in range(self.settings['ppo_epochs']):
            # Get batch size based on observation type
            batch_len = len(obs) if not isinstance(obs, dict) else len(list(obs.values())[0])
            
            # Use deterministic random state for batch sampling
            rng = np.random.RandomState(self.seed + epoch)
            idxs = rng.permutation(batch_len)
            
            epoch_losses = {"total_loss": [], "policy_loss": [], "value_loss": [], "entropy_loss": [], "gate_loss": []}
            
            for start in range(0, batch_len, self.settings['batch_size']):
                end = start + self.settings['batch_size']
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

                # Calculate combined loss for both networks
                total_loss, policy_loss, value_loss, entropy_bonus, gate_loss = self.calculate_loss(
                    mb_obs, mb_acts, mb_old_logps, mb_returns, mb_advantages
                )
                
                # Update both networks with single optimizer
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Log gradients before clipping (only on first epoch and first batch for efficiency)
                if epoch == 0 and start == 0 and iteration is not None:
                    self._log_gradient_distributions(iteration)
                
                # Add gradient clipping
                th.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                
                self.optimizer.step()
                
                epoch_losses["total_loss"].append(total_loss.item())
                epoch_losses["policy_loss"].append(policy_loss.item())
                epoch_losses["value_loss"].append(value_loss.item())
                epoch_losses["entropy_loss"].append(entropy_bonus.item())
                epoch_losses["gate_loss"].append(gate_loss.item())
                
            # Accumulate losses
            for key in losses.keys():
                losses[key].extend(epoch_losses[key])
        
        # Log parameter changes after optimization (only if iteration is provided)
        if iteration is not None:
            self._log_parameter_changes(iteration, old_params)

        return losses
    
    def train(self, env, iterations):
        for iteration in range(iterations):
            time_start = time.time()
          
            # Reset seeds for this iteration to ensure reproducibility
            self.reset_seed_for_iteration(iteration)
            
            # Seed the environment for reproducibility
            obs, _ = env.reset(seed=self.settings.get('seed', 0) + iteration)
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
                    
                    obs, _ = env.reset(seed=self.settings.get('seed', 0) + iteration)
                    if t >= self.settings['update_timesteps']:
                        break
                t += 1
                ep_t += 1

            # Extract data from buffer
            buffer_data = buffer.get_data()
            
            # Normalize rewards for better training stability
            rewards = buffer_data['rews'].cpu().numpy()
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

            # Advantages are not normalized, because all the rewards are already
            # normalized. Additional normalization would make policy loss too small.

            # Training step
            losses = self.update(
                buffer_data['obs'], 
                buffer_data['acts'], 
                buffer_data['logps'], 
                returns, 
                advantages,
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
            self._log_weight_distributions(iteration)
            
            # Log action distribution for the whole buffer
            all_actions = buffer_data['acts'].cpu().numpy()
            self._log_action_distribution_buffer(all_actions, iteration)
            
            # Print logging summary
            if self.logger is not None:
                print(f"  Logged action distribution for buffer ({len(all_actions)} actions)")
                print(f"  Logged gradients and parameter changes for all model parts")

            # Get gate coefficient for logging
            gate_coeff = self.model.get_latest_gate_coeff() if self.gate_loss_coef > 0.0 else None
            
            # Log metrics to TensorBoard
            if self.logger is not None:
                self.logger.add_scalar('Training/Mean_Return', mean_return, iteration)
                self.logger.add_scalar('Training/Std_Return', std_return, iteration)
                self.logger.add_scalar('Training/Mean_Steps', mean_steps, iteration)
                self.logger.add_scalar('Training/Std_Steps', std_steps, iteration)
                self.logger.add_scalar('Training/Time_Taken', time_taken, iteration)
                self.logger.add_scalar('Training/Episodes', len(ep_returns), iteration)
                
                # Log gate coefficient if available
                if gate_coeff is not None:
                    self.logger.add_scalar('Model/Gate_Coefficient', gate_coeff, iteration)
                
                # Log losses
                self.logger.add_scalar('Losses/Total_Loss', mean_losses['total_loss'], iteration)
                self.logger.add_scalar('Losses/Policy_Loss', mean_losses['policy_loss'], iteration)
                self.logger.add_scalar('Losses/Value_Loss', mean_losses['value_loss'], iteration)
                self.logger.add_scalar('Losses/Entropy_Loss', mean_losses['entropy_loss'], iteration)
                self.logger.add_scalar('Losses/Gate_Loss', mean_losses['gate_loss'], iteration)

            # Log current learning rates
            current_lrs = [group['lr'] for group in self.optimizer.param_groups]
            lr_info = f"LRs: visual={current_lrs[0]:.2e}, vector={current_lrs[1]:.2e}, general={current_lrs[2]:.2e}"
            
            gate_info = f" | Gate Coeff: {gate_coeff:.4f}" if gate_coeff is not None else ""
            print(f"Iteration {iteration} completed. Episodes: {len(ep_returns)} | Time taken: {time_taken:.2f}s | "
                  f"Mean Return: {mean_return:.4f} | Std Return: {std_return:.4f} | "
                  f"Mean steps: {mean_steps:.4f} | Std steps: {std_steps:.4f}{gate_info} | "
                  f"Mean losses: total: {mean_losses['total_loss']:.6f}, policy: {mean_losses['policy_loss']:.6f}, value: {mean_losses['value_loss']:.6f}, entropy: {mean_losses['entropy_loss']:.6f}, gate: {mean_losses['gate_loss']:.6f} | {lr_info}")
            
            # Step the learning rate scheduler
            self.scheduler.step()
        
        # Close TensorBoard logger
        if self.logger is not None:
            self.logger.close()