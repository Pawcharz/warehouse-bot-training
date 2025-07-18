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
from src.environments.env_raycasts_gymnasium_wrapper import UnityRaycastsGymWrapper

# Add root directory to path to find config module
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from config import ROOT_DIR

def make_env(env_path=None, time_scale=6, no_graphics=True, verbose=True, env_type="raycasts"):
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
                       Options: "raycasts", "camera_raycasts", "camera"
    
    Returns:
        UnityRaycastsGymWrapper or UnityCameraRaycastsGymWrapper: Configured gymnasium environment
    """
    if env_path is None:
        # Default environment path for training
        env_path = os.path.join(ROOT_DIR, "environment_builds/stage1/S1_Find_Deliver_16rays_rew0_100_200/Warehouse_Bot.exe")
    
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
    if env_type == "raycasts":
        from src.environments.env_raycasts_gymnasium_wrapper import UnityRaycastsGymWrapper
        gymnasium_env = UnityRaycastsGymWrapper(unity_env)
    elif env_type == "camera_raycasts":
        from src.environments.env_camera_raycasts_gymnasium_wrapper import UnityCameraRaycastsGymWrapper
        gymnasium_env = UnityCameraRaycastsGymWrapper(unity_env)
    elif env_type == "camera":
        from src.environments.env_camera_gymnasium_wrapper import UnityCameraGymWrapper
        gymnasium_env = UnityCameraGymWrapper(unity_env)
    else:
        raise ValueError(f"Unknown env_type: {env_type}. Must be one of: raycasts, camera_raycasts, camera")
    
    if verbose:
        print(f"Observation space: {gymnasium_env.observation_space}")
        print(f"Action space: {gymnasium_env.action_space}")
        print(f"Time scale: {time_scale}")
        print(f"Environment type: {env_type}")
    
    return gymnasium_env

 