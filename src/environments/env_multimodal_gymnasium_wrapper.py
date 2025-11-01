import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

class UnityMultimodalGymWrapper(gym.Env):
  def __init__(self, unity_env: UnityEnvironment, seed=None, add_previous_action=False):
    super().__init__()
    self.unity_env = unity_env
    self.unity_env.reset()
    self.behavior_name = list(self.unity_env.behavior_specs.keys())[0]
    self.spec = self.unity_env.behavior_specs[self.behavior_name]
    
    # Define observation space (assuming visual input)
    vis_obs_shape = self.spec.observation_specs[0].shape
    vec_obs_shape = self.spec.observation_specs[1].shape
    
    self.observation_space = spaces.Dict({
      "visual": spaces.Box(low=0, high=255, shape=vis_obs_shape, dtype=np.uint8),
      "vector": spaces.Box(low=0, high=255, shape=vec_obs_shape, dtype=np.uint8),
      "previous_action": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
    })
    
    # Define action space
    if self.spec.action_spec.is_discrete():
      self.action_space = spaces.Discrete(self.spec.action_spec.discrete_branches[0])

    self.add_previous_action = add_previous_action
    
  def reset(self):
    self.unity_env.reset()
    decision_steps, _ = self.unity_env.get_steps(self.behavior_name)
    
    observation, info = self.prepare_obs(decision_steps.obs, 0)
    return observation, info
  
  def split_vector_observation(self, vector_obs):
    """
    Split vector observation into actual observations and info
    Context: MlAgents framework does not allow to pass info separately - so it had to be passed as observations and split in the gymnasium wrapper
    """

    actual_observations = np.array([vector_obs[0][:2]])

    info_observations = vector_obs[0][2:]
    info = {
      "map_position": info_observations[0:2],
      "forward_direction": info_observations[2:4],
    }

    return actual_observations, info
  
  # action is an integer - id of the action
  def prepare_obs(self, env_obs, action: int):
    vector_obs = env_obs[1]
    vector_obs, info = self.split_vector_observation(vector_obs)
    if self.add_previous_action:
      return {"visual": env_obs[0], "vector": vector_obs, "previous_action": np.array([action], dtype=np.uint8)}, info
    else:
      return {"visual": env_obs[0], "vector": vector_obs}, info

  def step(self, action):
    action_tuple = ActionTuple()

    action_proper = None
    if self.add_previous_action:
      action_proper = action
    
    if self.spec.action_spec.is_discrete():
      action_tuple.add_discrete(np.array(action).reshape(1, -1))
    
    self.unity_env.set_action_for_agent(self.behavior_name, 0, action_tuple)
    self.unity_env.step()
    
    decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)

    if 0 in terminal_steps:
      obs, info = self.prepare_obs(terminal_steps.obs, action_proper)
      reward = terminal_steps.reward[0]
      
      # terminated - Natural episode ending.
      terminated = not terminal_steps.interrupted[0]
      
      # truncated - "Whether the truncation condition outside the scope of the MDP is satisfied. Typically, this is a timelimit"
      # interrupted - "The episode ended due to max steps or external termination, not because the episode ended naturally (failed/succeeded)."
      truncated = terminal_steps.interrupted[0]
      
      # terminated and truncated are mutually exclusive
    else:
      obs, info = self.prepare_obs(decision_steps.obs, action_proper)
      
      reward = decision_steps.reward[0]
      terminated = False
      truncated = False
    
    return obs, reward, terminated, truncated, info

  def render(self, mode='human'):
    pass  # Unity renders its own environment
  
  def close(self):
    self.unity_env.close()