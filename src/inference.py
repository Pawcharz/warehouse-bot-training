#!/usr/bin/env python3
"""
Inference Script for Warehouse Bot

This script loads a trained model and runs inference with rendering for thousands of steps.
The simulation runs at time_scale=1 for real-time visualization.
"""

import warnings
warnings.filterwarnings("ignore")

import time
import torch as th
from torch import multiprocessing
import numpy as np
import random
import os
import sys
import argparse
from pathlib import Path

# Add root directory to path to find config module
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Environment imports
from src.environments.env_utils import make_env

# Model imports
from src.models.actor_critic import ActorCritic
from src.models.model_utils import load_model_checkpoint, list_available_models

def load_model(model_path, obs_dim, act_dim, device):
    """Load a trained model from checkpoint"""
    # Create model architecture
    model = ActorCritic(obs_dim, act_dim)
    
    # Load checkpoint using the utility function
    model, checkpoint = load_model_checkpoint(model_path, model, device)
    
    return model

def run_inference(model, env, num_steps=5000, seed=42, render=True, device=None):
    """Run inference with the loaded model"""
    print(f"\nStarting inference for {num_steps} steps...")
    print(f"Seed: {seed}")
    print(f"Rendering: {render}")
    
    # Set seed for reproducibility
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Reset environment
    obs, _ = env.reset(seed=seed)
    
    # Statistics
    total_reward = 0
    episode_count = 0
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0
    
    start_time = time.time()
    
    for step in range(num_steps):
        # Convert observation to tensor
        obs_tensor = th.tensor(obs, dtype=th.float32, device=device).unsqueeze(0)
        
        # Get action from model (deterministic for inference)
        with th.no_grad():
            action, _, _, _ = model.get_action(obs_tensor, deterministic=True)
        
        # Take action in environment
        obs, reward, done, truncated, info = env.step(action.item())
        
        # Update statistics
        total_reward += reward
        current_episode_reward += reward
        current_episode_length += 1
        
        # Handle episode completion
        if done or truncated:
            episode_count += 1
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            
            print(f"\nEpisode {episode_count} completed:")
            print(f"  Length: {current_episode_length} steps")
            print(f"  Reward: {current_episode_reward:.2f}")
            print(f"  Average reward so far: {np.mean(episode_rewards):.2f}")
            
            # Reset for next episode
            obs, _ = env.reset(seed=seed + episode_count)
            current_episode_reward = 0
            current_episode_length = 0
    
    inference_time = time.time() - start_time
    
    # Print final statistics
    print(f"\n=== INFERENCE RESULTS ===")
    print(f"Total steps: {num_steps}")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Steps per second: {num_steps / inference_time:.2f}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per step: {total_reward / num_steps:.4f}")
    print(f"Episodes completed: {episode_count}")
    if episode_rewards:
        print(f"Average episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        print(f"Best episode reward: {max(episode_rewards):.2f}")
        print(f"Worst episode reward: {min(episode_rewards):.2f}")

def main():
    parser = argparse.ArgumentParser(description='Run inference with trained warehouse bot model')
    parser.add_argument('--model_path', type=str, 
                       default='saved_models/custom/stage1/S1_Find_Deliver_16rays_rew0_100_200_speed6x_ppo_seed_0.pth',
                       help='Path to the trained model checkpoint')
    parser.add_argument('--env_path', type=str, default=None,
                       help='Path to the Unity environment executable (default: training environment)')
    parser.add_argument('--num_steps', type=int, default=5000,
                       help='Number of steps to run inference for')
    parser.add_argument('--time_scale', type=float, default=1.0,
                       help='Time scale for simulation (1.0 = real-time)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--no_graphics', action='store_true',
                       help='Disable graphics rendering (faster but no visualization)')
    
    args = parser.parse_args()
    
    print("=== Warehouse Bot Inference ===")
    print(f"Model path: {args.model_path}")
    print(f"Environment path: {args.env_path or 'Default'}")
    print(f"Number of steps: {args.num_steps}")
    print(f"Time scale: {args.time_scale}")
    print(f"Seed: {args.seed}")
    print(f"Graphics: {'Disabled' if args.no_graphics else 'Enabled'}")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Use --list_models to see available models")
        return
    
    # Setup device
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        th.device(0)
        if th.cuda.is_available() and not is_fork
        else th.device("cpu")
    )
    print(f"Using device: {device}")
    
    # Create environment
    print("\nCreating environment...")
    env = make_env(env_path=args.env_path, time_scale=args.time_scale, no_graphics=args.no_graphics, verbose=True, env_type="raycasts")
    
    # Get environment dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {act_dim}")
    
    # Load model
    model = load_model(args.model_path, obs_dim, act_dim, device)
    
    try:
        # Run inference
        run_inference(model, env, num_steps=args.num_steps, seed=args.seed, render=not args.no_graphics, device=device)
    except KeyboardInterrupt:
        print("\nInference interrupted by user")
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close environment
        env.close()
        print("\nInference completed!")

if __name__ == "__main__":
    main()
