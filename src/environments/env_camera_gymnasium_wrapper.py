import gymnasium as gym
import numpy as np
from gymnasium import spaces
from mlagents_envs.environment import UnityEnvironment

class UnityCameraGymWrapper(gym.Env):
    def __init__(self, unity_env_path):
        super().__init__()

        # Launch Unity environment
        self.env = UnityEnvironment(file_name=unity_env_path, no_graphics=False)
        self.env.reset()
        
        # Assume single behavior; get its name
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        spec = self.env.behavior_specs[self.behavior_name]

        # Observation: visual observations only (usually obs[0])
        vis_obs_shape = spec.observation_shapes[0]  # e.g., (84, 84, 3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=vis_obs_shape, dtype=np.uint8
        )

        # Action space (assuming discrete or continuous, adjust if needed)
        if spec.action_spec.is_discrete():
            self.action_space = spaces.Discrete(spec.action_spec.discrete_branches[0])
        else:
            self.action_space = spaces.Box(
                low=spec.action_spec.continuous_min,
                high=spec.action_spec.continuous_max,
                shape=(spec.action_spec.continuous_size,),
                dtype=np.float32
            )

    def reset(self):
        self.env.reset()
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        obs = decision_steps.obs[0]  # visual observation only
        return obs[0]  # return first agent's visual obs

    def step(self, action):
        # Set action for the first agent
        self.env.set_actions(self.behavior_name, np.array([action]))
        self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        # Check if agent is done
        if len(terminal_steps) > 0:
            obs = terminal_steps.obs[0][0]
            reward = terminal_steps.reward[0]
            done = True
        else:
            obs = decision_steps.obs[0][0]
            reward = decision_steps.reward[0]
            done = False

        info = {}
        return obs, reward, done, info

    def close(self):
        self.env.close()
