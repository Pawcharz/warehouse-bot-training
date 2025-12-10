import os
import sys
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from config import ROOT_DIR

def make_env(env_path=None, time_scale=1, no_graphics=True, verbose=True, env_type="vector", seed=0):
    """
    Create and configure the Unity environment

    Args:
        env_path: Path to the Unity environment .exe file
        time_scale: time scale of simulation
        no_graphics: if graphics should be rendered - Unity handles seeding differently if graphics are rendered - results will be constand withing these groups (with/without graphics)
        verbose: if True, log to console
        env_type: type of environment to create (vector or multimodal for simple vector or camera+vector observations)
        seed: random seed for the Unity environment
    
    Returns: UnityVectorGymWrapper or UnityMultimodalGymWrapper
    """
    if env_path is None:
        raise ValueError("env_path must be specified. Please provide the path to the Unity environment executable.")
    
    if verbose:
        print(f"Environment: {env_path}")
        print(f"Seed: {seed}")
    
    channel = EngineConfigurationChannel()
    env_params_channel = EnvironmentParametersChannel()
    
    unity_env = UnityEnvironment(
        file_name=env_path,
        side_channels=[channel, env_params_channel],
        no_graphics=no_graphics,
        seed=seed
    )
    
    channel.set_configuration_parameters(time_scale=time_scale, quality_level=0, target_frame_rate=60)
    env_params_channel.set_float_parameter("seed", float(seed))
    
    if env_type == "multimodal":
        from src.environments.env_multimodal_gymnasium_wrapper import UnityMultimodalGymWrapper
        gymnasium_env = UnityMultimodalGymWrapper(unity_env, add_previous_action=True)
    else:
        raise ValueError(f"Unknown env_type: {env_type}. Currently only 'multimodal' is supported.")
    
    if verbose:
        print(f"Observation space: {gymnasium_env.observation_space}")
        print(f"Action space: {gymnasium_env.action_space}")
        print(f"Time scale: {time_scale}")
        print(f"Environment type: {env_type}")
    
    return gymnasium_env

 