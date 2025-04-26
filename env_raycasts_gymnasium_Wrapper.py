import numpy as np
from gymnasium import spaces
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import gymnasium as gym

class UnityRaycastsGymWrapper(gym.Env):
    def __init__(self, unity_env, seed=None):
        super().__init__()
        self.unity_env = unity_env
        self.unity_env.reset()
        self.behavior_name = list(self.unity_env.behavior_specs.keys())[0]
        self.spec = self.unity_env.behavior_specs[self.behavior_name]   
        
        # Define observation space (assuming visual input)
        obs_shape = self.spec.observation_specs[0].shape
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8) # ???
        
        # Define action space
        if self.spec.action_spec.is_discrete():
            self.action_space = spaces.Discrete(self.spec.action_spec.discrete_branches[0])

        
    def reset(self, seed=None, options=None):
        self.unity_env.reset()
        decision_steps, _ = self.unity_env.get_steps(self.behavior_name)
        obs = decision_steps.obs[0]  # Assuming single-agent scenario
        return obs, {}

    def step(self, action):
        action_tuple = ActionTuple()
        
        if self.spec.action_spec.is_discrete():
            action_tuple.add_discrete(np.array(action).reshape(1, -1))
        
        self.unity_env.set_action_for_agent(self.behavior_name, 0, action_tuple)
        self.unity_env.step()
        
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)

        if 0 in terminal_steps:
            obs = terminal_steps.obs[0]
            reward = terminal_steps.reward[0]
            
            # terminated - Natural episode ending.
            terminated = not terminal_steps.interrupted[0]
            
            # truncated - "Whether the truncation condition outside the scope of the MDP is satisfied. Typically, this is a timelimit"
            # interrupted - "The episode ended due to max steps or external termination, not because the episode ended naturally (failed/succeeded)."
            truncated = terminal_steps.interrupted[0]
            
            # terminated and truncated are mutually exclusive
        else:
            obs = decision_steps.obs[0]
            reward = decision_steps.reward[0]
            terminated = False
            truncated = False
        
        return obs, reward, terminated, truncated, {}

    def render(self, mode='human'):
        pass  # Unity renders its own environment
    
    def close(self):
        self.unity_env.close()