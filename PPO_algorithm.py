import torch as th
import torch.optim as optim
import numpy as np

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

    def compute_gae(self, gamma=0.99, lam=0.95, last_val=0.0):
        vals = self.buffer['vals'] + [last_val]
        vals = th.tensor(vals, dtype=th.float32, device=self.device)

        advantages = []
        gae = 0.0
        for t in reversed(range(len(self.buffer['rews']))):
            delta = self.buffer['rews'][t] + gamma * vals[t + 1] * (1 - self.buffer['dones'][t]) - vals[t]
            gae = delta + gamma * lam * (1 - self.buffer['dones'][t]) * gae
            advantages.insert(0, gae)

        advantages = th.tensor(advantages, dtype=th.float32, device=self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        returns = advantages + vals[:-1]

        # Handle dictionary observations
        if isinstance(self.buffer['obs'][0], dict):
            # Stack dictionary observations
            obs = {}
            for key in self.buffer['obs'][0].keys():
                obs[key] = th.stack([th.tensor(obs_item[key], dtype=th.float32, device=self.device) 
                                   for obs_item in self.buffer['obs']])
        else:
            # Handle simple tensor observations
            obs = th.stack([th.tensor(o, dtype=th.float32, device=self.device) for o in self.buffer['obs']])
            
        acts = th.tensor(self.buffer['acts'], dtype=th.int64, device=self.device)
        logps = th.tensor(self.buffer['logps'], dtype=th.float32, device=self.device)

        # Clear buffer
        for key in self.buffer:
            self.buffer[key].clear()

        return obs, acts, logps, returns, advantages

# PPO Agent
class PPOAgent:
    def __init__(self, model_net, settings):
        self.settings = settings
        self.device = settings['device']
        self.model = model_net.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=settings['lr'])

    # Returns: loss, policy_loss, value_loss, entropy_bonus
    def calculate_loss(self, mb_obs, mb_acts, mb_old_logps, mb_returns, mb_advantages):
        logps, entropy, values = self.model.evaluate_actions(mb_obs, mb_acts)
        ratios = th.exp(logps - mb_old_logps)

        surr1 = ratios * mb_advantages
        surr2 = th.clamp(ratios, 1 - self.settings['clip_eps'], 1 + self.settings['clip_eps']) * mb_advantages
        policy_loss = -th.min(surr1, surr2).mean()

        # Debugging predicted values vs. returns
        print("values: ", values)
        print("mb_returns: ", mb_returns)
        print("difference: ", values - mb_returns)

        value_loss = ((values - mb_returns)**2).mean()
        entropy_bonus = entropy.mean()

        value_loss *= self.settings['val_loss_coef']
        entropy_bonus *= self.settings['ent_loss_coef']
        loss = policy_loss + value_loss - entropy_bonus
        
        print("Returns range:", mb_returns.min().item(), "to", mb_returns.max().item())
        print("Values range:", values.min().item(), "to", values.max().item())
        print("Value loss before coefficient:", ((values - mb_returns)**2).mean().item())
        print("Value loss after coefficient:", value_loss.item())

        return loss, policy_loss, value_loss, entropy_bonus
    
    # Returns average loss of the batch
    def update(self, obs, acts, old_logps, returns, advantages):
        losses = {"total_loss": [], "policy_loss": [], "value_loss": [], "entropy_loss": []}
        
        print(f"Advantages - mean: {advantages.mean():.4f}, std: {advantages.std():.4f}")

        for _ in range(self.settings['ppo_epochs']):
            # Get batch size based on observation type
            batch_len = len(obs) if not isinstance(obs, dict) else len(list(obs.values())[0])
            idxs = np.random.permutation(batch_len)
            
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

                loss, policy_loss, value_loss, entropy_bonus = self.calculate_loss(mb_obs, mb_acts, mb_old_logps, mb_returns, mb_advantages)
                losses["total_loss"].append(loss.item())
                losses["policy_loss"].append(policy_loss.item())
                losses["value_loss"].append(value_loss.item())
                losses["entropy_loss"].append(entropy_bonus.item())

                self.optimizer.zero_grad()
                loss.backward()
                
                # Add gradient clipping
                th.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optimizer.step()

        return losses
    
    def train(self, env, iterations):
        for iteration in range(iterations):
            obs, _ = env.reset()
            buffer = RolloutBuffer(self.device)
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
                    obs_tensor = {key: th.tensor(obs_tensor[key].squeeze(0), dtype=th.float32, device=self.device) for key in obs.keys()}
                else:
                    obs_tensor = th.tensor(obs_tensor.squeeze(0), dtype=th.float32, device=self.device) # Unsure of squeeze

                # Store observations in buffer
                buffer.add(obs_tensor, action.item(), logp.item(), reward, value.item(), done)
                ep_return += reward
                obs = next_obs

                if done:
                    ep_returns.append(ep_return)
                    ep_return = 0
                    ep_steps.append(ep_t)
                    ep_t = 0
                    obs, _ = env.reset()
                    if t >= self.settings['update_timesteps']:
                        break
                t += 1
                ep_t += 1

            # Training step
            obs, acts, logps, returns, advantages = buffer.compute_gae(self.settings['gamma'], self.settings['lam'])
            losses = self.update(obs, acts, logps, returns, advantages)
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