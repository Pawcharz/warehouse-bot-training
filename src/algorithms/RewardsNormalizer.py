import numpy as np

# Implements rewards normalization algorithm inspired by stable-baselines3 and gymnasium implementations
# https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/vec_env/vec_normalize.html SOURCE
# https://gymnasium.farama.org/v0.29.0/_modules/gymnasium/wrappers/normalize/ SOURCE

class RewardNormalizer:
  def __init__(self, gamma=0.99, epsilon=1e-8):
    self.gamma = gamma
    self.epsilon = epsilon # To not divide by 0
    self.cur_return = 0.0 # current return
    self.running_stats = RunningMeanStd()
  
  def normalize(self, rewards):
    """Normalize rewards using running statistics of discounted returns"""
    normalized_rewards = []
    
    for reward in rewards:
      self.cur_return = self.cur_return * self.gamma + reward
      self.running_stats.update(self.cur_return)
      
      normalized_reward = reward / np.sqrt(self.running_stats.var + self.epsilon)
      normalized_rewards.append(normalized_reward)
    
    return np.array(normalized_rewards)
  
  def reset(self):
    """Reset the return tracker"""
    self.cur_return = 0.0


# FIX - could potentially be simplified - if only used in rewards normalization, the x is always a scalar, no need to calculate variance etc
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
  
    delta_mean = batch_mean - self.mean
    tot_count = self.count + batch_count

    new_mean = self.mean + delta_mean * batch_count / tot_count

    # Combined Variance (https://www.emathzone.com/tutorials/basic-statistics/combined-variance.html) SOURCE
    # {S_c}^2 formula
    new_var = self.count*self.var + batch_count*batch_var + self.count * (self.mean-new_mean)**2 + batch_count * (batch_mean-new_mean)**2
    new_var /= tot_count
    
    self.mean = new_mean
    self.var = new_var
    self.count = tot_count
