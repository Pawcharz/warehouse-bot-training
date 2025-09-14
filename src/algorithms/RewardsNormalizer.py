import numpy as np

# Implements rewards normalization algorithm inspired by stable-baselines3 implementation
# https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/vec_env/vec_normalize.html

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
