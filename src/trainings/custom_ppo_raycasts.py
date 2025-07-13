#!/usr/bin/env python3
"""
PPO Training Script for Warehouse Stage1 Complex Pos Neg 3 Environment

This script trains a PPO agent on the custom warehouse environment using raycast observations.
Based on the ppo_raw.ipynb notebook structure.
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

# Add root directory to path to find config module
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Environment imports
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.environment import UnityEnvironment
from src.environments.env_raycasts_gymnasium_wrapper import UnityRaycastsGymWrapper

# Algorithm imports
from src.algorithms.PPO_algorithm import PPOAgent
# from src.algorithms.PPO_algorithm_returns_clipping import PPOAgent
from src.models.actor_critic import ActorCritic, count_parameters

from config import ROOT_DIR

def make_env():
    """Create and configure the Unity environment"""
    env_path = os.path.join(ROOT_DIR, "environment_builds/stage1/S1_Find_16rays_rew0_100/Warehouse_Bot.exe")
    
    # Debug: print the path to verify it's correct
    print(f"Looking for environment at: {env_path}")
    print(f"File exists: {os.path.exists(env_path)}")
    
    channel = EngineConfigurationChannel()
    
    unity_env = UnityEnvironment(
        file_name=env_path,
        side_channels=[channel],
        no_graphics=True
    )
    
    # channel.set_configuration_parameters(time_scale=1)
    
    gymnasium_env = UnityRaycastsGymWrapper(unity_env)
    
    print(f"Observation space: {gymnasium_env.observation_space}")
    print(f"Action space: {gymnasium_env.action_space}")
    
    return gymnasium_env

def evaluate_policy(agent, env, num_episodes=10, seed=0):
    """Evaluate the trained policy"""
    returns = []
    steps = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_return = 0
        episode_steps = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Convert observation to tensor
            obs_tensor = th.tensor(obs, dtype=th.float32, device=agent.device).unsqueeze(0)
            
            # Get action from policy
            with th.no_grad():
                action, _, _, _ = agent.model.get_action(obs_tensor)
            
            # Take action in environment
            obs, reward, done, truncated, _ = env.step(action.item())
            episode_return += reward
            episode_steps += 1
        
        returns.append(episode_return)
        steps.append(episode_steps)
    
    return np.mean(returns), np.std(returns), np.mean(steps), np.std(steps)

def main():
    print("Starting PPO Training for Warehouse Stage1 Complex Pos Neg 3...")
    
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
    env = make_env()
    
    # Get environment dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {act_dim}")
    
    # Set seed for reproducibility
    seed = 0
    print(f"Using seed: {seed}")
    
    # Set seeds before creating model to ensure deterministic initialization
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    
    # PPO settings
    settings = {
        'gamma': 0.99,
        'lambda': 0.95,
        'clip_eps': 0.2,
        'ppo_epochs': 4,
        'batch_size': 128,
        'update_timesteps': 2048,
        'lr': 3e-4,
        'val_loss_coef': 0.5,
        'ent_loss_coef': 0.01,
        'device': device,
        'seed': seed,
        'use_tensorboard': True,
        'tensorboard_log_dir': 'logs/stage1/S1_Find_16rays_rew0_100',
        'experiment_name': f'ppo_seed_{seed}'
    }
    training_iterations = 200

    # Create model
    model_net = ActorCritic(obs_dim, act_dim)
    
    # Count and display parameters
    model_params = count_parameters(model_net)
    print(f"\nModel parameters: {model_params}")
    print(f"Total parameters: {model_params['total']}")
    
    print(f"\nPPO Settings:")
    for key, value in settings.items():
        print(f"  {key}: {value}")
    
    # Create PPO agent
    print("\nCreating PPO agent...")
    agent = PPOAgent(model_net, settings, seed=seed)
    
    # Training
    print("\nStarting training...")
    start_time = time.time()
    
    # Training iterations
    agent.train(env, iterations=training_iterations)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluation
    print("\nEvaluating trained policy...")
    mean_return, std_return, mean_steps, std_steps = evaluate_policy(
        agent, env, num_episodes=10, seed=seed
    )
    
    print(f"\n=== TRAINING RESULTS ===")
    print(f"Training iterations: {training_iterations}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Mean evaluation return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"Mean evaluation steps: {mean_steps:.2f} ± {std_steps:.2f}")
    
    # Save model (optional)
    try:
        save_dir = os.path.join("saved_models", "custom", "stage1")
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, f"S1_Find_16rays_rew0_100_ppo_seed_{seed}.pth")
        th.save({
            'model_state_dict': agent.model.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'settings': settings,
            'seed': seed,
            'training_iterations': training_iterations,
            'final_mean_return': mean_return,
            'final_std_return': std_return
        }, model_path)
        print(f"Model saved to: {model_path}")
    except Exception as e:
        print(f"Could not save model: {e}")
    
    # Close environment
    env.close()
    print("\nTraining script completed!")

if __name__ == "__main__":
    main()
