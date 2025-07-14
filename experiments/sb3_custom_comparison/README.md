THis is an experiment comparing results of 2 PPO implementations: my custom PPO at `src/algorithms/PPO_algorithm.py` with stable-baselines-3 implementation.

## Setup

### Sourcecode
At the moment of experiment, source code of custom ppo looks as follows:

```py
import time
import torch as th
import torch.optim as optim
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import os
import sys

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
        self.optimizer = optim.Adam(self.model.parameters(), lr=settings['lr'])

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
        self.max_grad_norm = settings.get('max_grad_norm', 0.5)
        
        # Store other settings as instance variables for logging
        self.learning_rate = settings.get('lr', 3e-4)
        self.gamma = settings.get('gamma', 0.99)
        self.lambda_val = settings.get('lambda', 0.95)
        self.ppo_epochs = settings.get('ppo_epochs', 4)
        self.batch_size = settings.get('batch_size', 64)
        self.update_timesteps = settings.get('update_timesteps', 1024)
        self.val_loss_coef = settings.get('val_loss_coef', 0.5)
        self.ent_loss_coef = settings.get('ent_loss_coef', 0.01)
        
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

    def set_seed(self, seed):
        """Set seeds for all random components to ensure reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
        
        # Ensure deterministic model initialization
        th.nn.init.seed = seed

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

        # Apply coefficients
        value_loss *= self.settings['val_loss_coef']
        entropy_bonus *= self.settings['ent_loss_coef']

        # Return combined loss for single optimizer
        total_loss = policy_loss + value_loss - entropy_bonus
        return total_loss, policy_loss, value_loss, entropy_bonus
    
    # Returns average loss of the batch
    def update(self, obs, acts, old_logps, returns, advantages):
        losses = {"total_loss": [], "policy_loss": [], "value_loss": [], "entropy_loss": []}
      
        for epoch in range(self.settings['ppo_epochs']):
            # Get batch size based on observation type
            batch_len = len(obs) if not isinstance(obs, dict) else len(list(obs.values())[0])
            
            # Use deterministic random state for batch sampling
            rng = np.random.RandomState(self.seed + epoch)
            idxs = rng.permutation(batch_len)
            
            epoch_losses = {"total_loss": [], "policy_loss": [], "value_loss": [], "entropy_loss": []}
            
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
                total_loss, policy_loss, value_loss, entropy_bonus = self.calculate_loss(
                    mb_obs, mb_acts, mb_old_logps, mb_returns, mb_advantages
                )
                
                # Update both networks with single optimizer
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Add gradient clipping
                th.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optimizer.step()
                
                epoch_losses["total_loss"].append(total_loss.item())
                epoch_losses["policy_loss"].append(policy_loss.item())
                epoch_losses["value_loss"].append(value_loss.item())
                epoch_losses["entropy_loss"].append(entropy_bonus.item())
                
            # Accumulate losses
            for key in losses.keys():
                losses[key].extend(epoch_losses[key])

        return losses
    
    def train(self, env, iterations):
        for iteration in range(iterations):
            time_start = time.time()
          
            # Reset seeds for this iteration to ensure reproducibility
            self.reset_seed_for_iteration(iteration)
            
            # Seed the environment for reproducibility
            obs, _ = env.reset(seed=self.settings.get('seed', 0) + iteration)
            buffer = RolloutBuffer(self.device)

            # Reset reward normalizer for new iteration
            self.reward_normalizer.reset()
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
                advantages
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

            # Log metrics to TensorBoard
            if self.logger is not None:
                self.logger.add_scalar('Training/Mean_Return', mean_return, iteration)
                self.logger.add_scalar('Training/Std_Return', std_return, iteration)
                self.logger.add_scalar('Training/Mean_Steps', mean_steps, iteration)
                self.logger.add_scalar('Training/Std_Steps', std_steps, iteration)
                self.logger.add_scalar('Training/Time_Taken', time_taken, iteration)
                self.logger.add_scalar('Training/Episodes', len(ep_returns), iteration)
                
                # Log losses
                self.logger.add_scalar('Losses/Total_Loss', mean_losses['total_loss'], iteration)
                self.logger.add_scalar('Losses/Policy_Loss', mean_losses['policy_loss'], iteration)
                self.logger.add_scalar('Losses/Value_Loss', mean_losses['value_loss'], iteration)
                self.logger.add_scalar('Losses/Entropy_Loss', mean_losses['entropy_loss'], iteration)

            print(f"Iteration {iteration} completed. Episodes: {len(ep_returns)} | Time taken: {time_taken:.2f}s | "
                  f"Mean Return: {mean_return:.4f} | Std Return: {std_return:.4f} | "
                  f"Mean steps: {mean_steps:.4f} | Std steps: {std_steps:.4f} | "
                  f"Mean losses: total: {mean_losses['total_loss']:.6f}, policy: {mean_losses['policy_loss']:.6f}, value: {mean_losses['value_loss']:.6f}, entropy: {mean_losses['entropy_loss']:.6f}")
        
        # Close TensorBoard logger
        if self.logger is not None:
            self.logger.close()
```

The comparison script can be found at `experiments/sb3_custom_comparison/ppo_comparison.py`:

for reproducability, its source code is following:

```py
#!/usr/bin/env python3
"""
PPO Implementation Comparison with Multiple Seeds

Runs custom and SB3 PPO implementations on multiple seeds and provides
statistical comparison of results.
"""

import time
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import warnings
import random
import sys
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings('ignore')

# Get the root directory (two levels up from this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(script_dir))

# Add src directory to path
sys.path.insert(0, ROOT_DIR)

from src.algorithms.PPO_algorithm import PPOAgent
from src.models.actor_critic import ActorCritic

# Setup tensorboard logging
log_dir = os.path.join(script_dir, "tensorboard_logs")
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

def evaluate_policy(agent, env, num_episodes=10, seed=0):
    """Evaluate policy and return mean/std of returns"""
    returns = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_return = 0
        done = False
        
        while not done:
            obs_tensor = th.tensor(obs, dtype=th.float32, device=agent.device).unsqueeze(0)
            with th.no_grad():
                action, _, _, _ = agent.model.get_action(obs_tensor, deterministic=True)
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            episode_return += reward
            done = terminated or truncated
        
        returns.append(episode_return)
    
    return np.mean(returns), np.std(returns)

def test_custom_ppo(env_name, seed, iterations=10):
    """Test custom PPO implementation"""
    set_seed(seed)
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model_net = ActorCritic(obs_dim, act_dim)
    
    settings = {
        'device': device,
        'lr': 3e-4,
        'gamma': 0.99,
        'lambda': 0.95,
        'clip_eps': 0.2,
        'max_grad_norm': 0.5,
        'ppo_epochs': 4,
        'batch_size': 64,
        'update_timesteps': 1024,
        'val_loss_coef': 0.5,
        'ent_loss_coef': 0.01,
        'seed': seed,
        'use_tensorboard': False,
    }
    agent = PPOAgent(model_net, settings, seed=seed)
    
    start_time = time.time()
    agent.train(env, iterations=iterations)
    training_time = time.time() - start_time
    
    mean_return, std_return = evaluate_policy(agent, env, num_episodes=10, seed=seed)
    
    env.close()
    return mean_return, std_return, training_time

def test_sb3_ppo(env_name, seed, total_timesteps=10240):
    """Test SB3 PPO implementation"""
    set_seed(seed)
    
    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])
    
    # Add reward normalization only
    env = VecNormalize(
        env,
        norm_obs=False,
        norm_reward=True, # Only normalize rewards
        clip_obs=np.inf, # No clipping
        clip_reward=np.inf, # No clipping
    )
    

    # Policy kwargs to match custom network architecture exactly
    policy_kwargs = {
        "net_arch": {
            "pi": [128, 128],
            "vf": [128, 128]
        },
        "activation_fn": nn.Tanh,
        "ortho_init": True,
    }
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        clip_range=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        policy_kwargs=policy_kwargs
    )
    
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    training_time = time.time() - start_time
    
    # Evaluate
    env.norm_reward = False  # Disable reward normalization for evaluation
    eval_returns = []
    for episode in range(10):
        obs = env.reset()
        episode_return = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_return += reward[0]
            done = done[0]
        
        eval_returns.append(episode_return)
    
    mean_return = np.mean(eval_returns)
    std_return = np.std(eval_returns)
    
    env.close()
    return mean_return, std_return, training_time

def run_comparison(env_name, seeds, custom_iterations=10, sb3_timesteps=10240):
    """Run comparison for one environment across multiple seeds"""
    print(f"\n{env_name}:")
    
    custom_results = []
    sb3_results = []
    
    for i, seed in enumerate(seeds):
        print(f"  Seed {seed} ({i+1}/{len(seeds)})...", end=" ")
        
        try:
            custom_mean, custom_std, custom_time = test_custom_ppo(env_name, seed, custom_iterations)
            sb3_mean, sb3_std, sb3_time = test_sb3_ppo(env_name, seed, sb3_timesteps)

            writer.add_scalars(f'{env_name}/mean_returns', {
                'Custom_PPO': custom_mean,
                'SB3_PPO': sb3_mean
            }, i)
            writer.add_scalars(f'{env_name}/mean_stds', {
                'Custom_PPO': custom_std,
                'SB3_PPO': sb3_std
            }, i)
            writer.add_scalars(f'{env_name}/mean_times', {
                'Custom_PPO': custom_time,
                'SB3_PPO': sb3_time
            }, i)
            
            custom_results.append({
                'mean': custom_mean,
                'std': custom_std,
                'time': custom_time
            })
            
            sb3_results.append({
                'mean': sb3_mean,
                'std': sb3_std,
                'time': sb3_time
            })
            
        except Exception as e:
            print(f"Error: {str(e)}")
            continue
    
    if not custom_results or not sb3_results:
        print(f"No successful runs for {env_name}")
        return None
    
    # Calculate statistics
    custom_means = [r['mean'] for r in custom_results]
    custom_stds = [r['std'] for r in custom_results]
    custom_times = [r['time'] for r in custom_results]
    
    sb3_means = [r['mean'] for r in sb3_results]
    sb3_stds = [r['std'] for r in sb3_results]
    sb3_times = [r['time'] for r in sb3_results]
    
    # Overall statistics
    custom_mean_of_means = np.mean(custom_means)
    custom_std_of_means = np.std(custom_means)
    custom_mean_of_stds = np.mean(custom_stds)
    custom_mean_time = np.mean(custom_times)
    
    sb3_mean_of_means = np.mean(sb3_means)
    sb3_std_of_means = np.std(sb3_means)
    sb3_mean_of_stds = np.mean(sb3_stds)
    sb3_mean_time = np.mean(sb3_times)
    
    print(f"  Custom PPO: {custom_mean_of_means:.1f} ± {custom_std_of_means:.1f} (eval_std: {custom_mean_of_stds:.1f}, time: {custom_mean_time:.1f}s)")
    print(f"  SB3 PPO:    {sb3_mean_of_means:.1f} ± {sb3_std_of_means:.1f} (eval_std: {sb3_mean_of_stds:.1f}, time: {sb3_mean_time:.1f}s)")
    
    writer.add_scalars(f'{env_name}/mean_returns/all_runs', {
        'Custom_PPO': custom_mean_of_means,
        'SB3_PPO': sb3_mean_of_means
    }, 0)
    writer.add_scalars(f'{env_name}/mean_stds/all_runs', {
        'Custom_PPO': custom_std_of_means,
        'SB3_PPO': sb3_std_of_means
    }, 0)
    writer.add_scalars(f'{env_name}/mean_times/all_runs', {
        'Custom_PPO': custom_mean_time,
        'SB3_PPO': sb3_mean_time
    }, 0)
    writer.close()

    return {
        'env_name': env_name,
        'custom_mean_of_means': custom_mean_of_means,
        'custom_std_of_means': custom_std_of_means,
        'custom_mean_of_stds': custom_mean_of_stds,
        'custom_mean_time': custom_mean_time,
        'sb3_mean_of_means': sb3_mean_of_means,
        'sb3_std_of_means': sb3_std_of_means,
        'sb3_mean_of_stds': sb3_mean_of_stds,
        'sb3_mean_time': sb3_mean_time,
        'performance_winner': 'custom' if custom_mean_of_means > sb3_mean_of_means else 'sb3' if sb3_mean_of_means > custom_mean_of_means else 'tie',
        'speed_ratio': custom_mean_time / sb3_mean_time,
        'n_runs': len(custom_results)
    }

def main():
    """Main comparison function"""
    print("PPO Implementation Comparison with Multiple Seeds")
    print("=" * 60)
    
    # Configuration
    seeds = range(20)
    custom_iterations = [10, 10]
    sb3_timesteps = [10240, 10240] # 10 * 1024 to match custom PPO
    
    environments = ["CartPole-v1", "Acrobot-v1"]

    print(f"Environments to test: {environments}")
    print(f"Seeds: {seeds}")

    print(f"Config: {custom_iterations} iterations vs {sb3_timesteps} timesteps")
    print(f"Tensorboard logs saved to: {log_dir}")
    
    # Run comparisons
    all_results = []
    
    for i, env_name in enumerate(environments):
        result = run_comparison(env_name, seeds, custom_iterations[i], sb3_timesteps[i])
        if result:
            all_results.append(result)
    
    if not all_results:
        print("\nNo successful comparisons completed")
        return
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    # Detailed results table
    print(f"\n{'='*100}")
    print("DETAILED RESULTS")
    print(f"{'='*100}")
    print(f"{'Environment':<15} {'Custom (mean +- std)':<20} {'SB3 (mean +- std)':<20} {'Speed':<10} {'Runs':<5}")
    print(f"{'-'*100}")
    
    for result in all_results:
        
        custom_str = f"{result['custom_mean_of_means']:.1f}±{result['custom_std_of_means']:.1f}"
        sb3_str = f"{result['sb3_mean_of_means']:.1f}±{result['sb3_std_of_means']:.1f}"
        speed_str = f"{result['speed_ratio']:.1f}x"
        
        print(f"{result['env_name']:<15} {custom_str:<20} {sb3_str:<20} {speed_str:<10} {result['n_runs']:<5}")

if __name__ == "__main__":
    main() 
```

### Configuration

Both custom ppo algorithm and sb3's implementation shared the same hyperparameters:

- device: device
- lr: 3e-4
- gamma: 0.99
- lambda: 0.95
- clip_eps: 0.2
- max_grad_norm: 0.5
- ppo_epochs: 4
- batch_size: 64
- update_timesteps: 1024
- val_loss_coef: 0.5
- ent_loss_coef: 0.01
- seed: seed
- use_tensorboard: False

with separate networks of following configuration:

```py
self.actor = nn.Sequential(
    nn.Linear(obs_dim, 128),
    nn.Tanh(),
    nn.Linear(128, 128),
    nn.Tanh(),
    nn.Linear(128, act_dim)
)

self.critic = nn.Sequential(
    nn.Linear(obs_dim, 128),
    nn.Tanh(),
    nn.Linear(128, 128),
    nn.Tanh(),
    nn.Linear(128, 1)
)
```

### Results
Results of the comparison can be found at `experiments/sb3_custom_comparison/tensorboard_logs` and can be run with following command: `tensorboard --logdir=<path>` where path is absolute path to tensorboard_logs

## Conclusions
add here