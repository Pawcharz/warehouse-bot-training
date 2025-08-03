#!/usr/bin/env python3
"""
Environment Utilities for Warehouse Bot

This module provides a shared environment creation utility that can be used
across training, inference, and evaluation scripts.

Usage:
    from src.environments.env_utils import make_env
    
    # For training (fast, no graphics)
    env = make_env(time_scale=6, no_graphics=True, env_type="raycasts")
    
    # For inference (real-time, with graphics)
    env = make_env(time_scale=1.0, no_graphics=False, env_type="raycasts")
    
    # For camera + raycasts environment
    env = make_env(time_scale=1.0, no_graphics=False, env_type="camera_raycasts")
"""

import os
import sys
from pathlib import Path

# Environment imports
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.environment import UnityEnvironment

# Add root directory to path to find config module
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from config import ROOT_DIR

def make_env(env_path=None, time_scale=6, no_graphics=True, verbose=True, env_type="vector"):
    """
    Create and configure the Unity environment
    
    Args:
        env_path (str, optional): Path to the Unity environment executable.
                           If None, uses the default training environment.
        time_scale (float): Time scale for simulation (1.0 = real-time, higher = faster).
                           Default is 6 for training speed.
        no_graphics (bool): Whether to disable graphics rendering.
                           Default is True for training (faster).
        verbose (bool): Whether to print debug information.
        env_type (str): Type of environment wrapper to use.
                       Options: "vector", "multimodal"
    
    Returns:
        UnityVectorGymWrapper or UnityMultimodalGymWrapper: Configured gymnasium environment
    """
    if env_path is None:
        raise ValueError("env_path must be specified. Please provide the path to the Unity environment executable.")
    
    if verbose:
        print(f"Looking for environment at: {env_path}")
        print(f"File exists: {os.path.exists(env_path)}")
    
    channel = EngineConfigurationChannel()
    
    unity_env = UnityEnvironment(
        file_name=env_path,
        side_channels=[channel],
        no_graphics=no_graphics
    )
    
    # Set time scale for simulation
    channel.set_configuration_parameters(time_scale=time_scale)
    
    # Choose appropriate wrapper based on env_type
    if env_type == "vector":
        from src.environments.env_vector_gymnasium_wrapper import UnityVectorGymWrapper
        gymnasium_env = UnityVectorGymWrapper(unity_env)
    elif env_type == "multimodal":
        from src.environments.env_multimodal_gymnasium_wrapper import UnityMultimodalGymWrapper
        gymnasium_env = UnityMultimodalGymWrapper(unity_env)
    else:
        raise ValueError(f"Unknown env_type: {env_type}. Must be one of: vector, multimodal")
    
    if verbose:
        print(f"Observation space: {gymnasium_env.observation_space}")
        print(f"Action space: {gymnasium_env.action_space}")
        print(f"Time scale: {time_scale}")
        print(f"Environment type: {env_type}")
    
    return gymnasium_env

 