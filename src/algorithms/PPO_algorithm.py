import torch as th
import torch.optim as optim
import numpy as np
import random

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

# Reward Normalizer - dig deeper how it works
class RewardNormalizer:
    def __init__(self, gamma=0.99, epsilon=1e-8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.ret = 0.0
        self.ret_rms = RunningMeanStd()
    
    def normalize(self, rewards):
        """Normalize rewards using running statistics of discounted returns"""
        normalized_rewards = []
        
        for reward in rewards:
            self.ret = self.ret * self.gamma + reward
            self.ret_rms.update(self.ret)
            normalized_reward = reward / np.sqrt(self.ret_rms.var + self.epsilon)
            normalized_rewards.append(normalized_reward)
        
        return np.array(normalized_rewards)
    
    def reset(self):
        """Reset the return tracker"""
        self.ret = 0.0

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        # Handle scalar values
        if np.isscalar(x):
            batch_mean = x
            batch_var = 0.0
            batch_count = 1
        else:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = len(x) if len(x.shape) > 0 else 1
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

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
        self.value_clip_eps = settings.get('value_clip_eps', 0.2)  # Separate clipping for value function
        self.max_grad_norm = settings.get('max_grad_norm', 0.5)
        
        # Store the seed for later use
        self.seed = seed

    def set_seed(self, seed):
        """Set seeds for all random components to ensure reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)  # For multi-GPU setups
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

    # Returns: loss, policy_loss, value_loss, entropy_bonus
    def calculate_loss(self, mb_obs, mb_acts, mb_old_logps, mb_returns, mb_advantages):
        logps, entropy, values = self.model.evaluate_actions(mb_obs, mb_acts)
        ratios = th.exp(logps - mb_old_logps)

        # Policy loss with clipping
        surr1 = ratios * mb_advantages
        surr2 = th.clamp(ratios, 1 - self.settings['clip_eps'], 1 + self.settings['clip_eps']) * mb_advantages
        policy_loss = -th.min(surr1, surr2).mean()


        # print("-" * 10)

        # print("logps mean: ", logps.mean(), "old logps mean: ", mb_old_logps.mean())
        # print("ratios mean: ", ratios.mean())

        # print("advantages mean: ", mb_advantages.mean())
        # print("values mean: ", values.mean())
        # print("returns mean: ", mb_returns.mean())

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
        
        # Normalize advantages
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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

            # Normalize advantages for better training stability
            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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

            print(f"Iteration {iteration} completed. Episodes: {len(ep_returns)} | "
                  f"Mean Return: {mean_return:.4f} | Std Return: {std_return:.4f} | "
                  f"Mean steps: {mean_steps:.4f} | Std steps: {std_steps:.4f} | "
                  f"Mean losses: total: {mean_losses['total_loss']:.6f}, policy: {mean_losses['policy_loss']:.6f}, value: {mean_losses['value_loss']:.6f}, entropy: {mean_losses['entropy_loss']:.6f}")