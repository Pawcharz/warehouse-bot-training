#!/usr/bin/env python3
"""
Evaluation utilities for trained policies
"""

import torch as th
import numpy as np
from typing import Tuple


def prepare_observation(obs, device: th.device, obs_type: str = "vector"):
    """
    Prepare observation for model input based on observation type
    """
    if obs_type == "auto":
        if isinstance(obs, dict):
            obs_type = "multimodal"
        else:
            obs_type = "vector"
    
    if obs_type == "multimodal":
        return obs
    elif obs_type == "vector":
        if isinstance(obs, th.Tensor):
            obs_tensor = obs.to(device)
        else:
            obs_tensor = th.tensor(obs, dtype=th.float32, device=device)
            
        # FIX - is it needed?
        return obs_tensor.unsqueeze(0) if obs_tensor.dim() == 1 else obs_tensor
    else:
        raise ValueError(f"Unknown observation type: {obs_type}. Use 'vector', 'multimodal', or 'auto'.")


def evaluate_policy(model, env, device: th.device, num_episodes: int = 10, seed: int = 0, 
                   obs_type: str = "auto", verbose: bool = True) -> Tuple[float, float, float, float]:
    returns = []
    steps = []
    
    if verbose:
        print(f"Evaluating policy for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_return = 0
        episode_steps = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            model_obs = prepare_observation(obs, device, obs_type)
            
            with th.no_grad():
                action, _, _, _ = model.get_action(model_obs, deterministic=True)
            
            # Take action in the environment
            obs, reward, done, truncated, _ = env.step(action.item())
            episode_return += reward
            episode_steps += 1
        
        returns.append(episode_return)
        steps.append(episode_steps)
        
        if verbose:
            print(f"Episode {episode + 1}: Return = {episode_return:.2f}, Steps = {episode_steps}")
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    mean_steps = np.mean(steps)
    std_steps = np.std(steps)
    
    if verbose:
        print(f"\nEvaluation Results:")
        print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")
        print(f"Mean steps: {mean_steps:.2f} ± {std_steps:.2f}")
        print(f"Best episode: {max(returns):.2f}")
        print(f"Worst episode: {min(returns):.2f}")
    
    return mean_return, std_return, mean_steps, std_steps
