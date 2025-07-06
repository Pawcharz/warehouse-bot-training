#!/usr/bin/env python3
"""
Simple PPO Test Script for CartPole-v1

A basic script to test your custom PPO implementation on CartPole-v1
and log evaluation results after training.
"""

import time
import numpy as np
import torch as th
import gymnasium as gym
from src.algorithms.PPO_algorithm import PPOAgent
from src.models.actor_critic import ActorCritic, count_parameters
import random

def evaluate_policy(agent, env, num_episodes=10, seed=0):
    """Evaluate the trained policy"""
    returns = []
    
    for episode in range(num_episodes):
        # Seed the environment for reproducible evaluation
        obs, _ = env.reset(seed=seed + episode)
        episode_return = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Convert observation to tensor
            obs_tensor = th.tensor(obs, dtype=th.float32, device=agent.device).unsqueeze(0)
            
            # Get action from policy
            with th.no_grad():
                action, _, _, _ = agent.model.get_action(obs_tensor, deterministic=True)
            
            # Take action in environment
            obs, reward, done, truncated, _ = env.step(action.item())
            episode_return += reward
        
        returns.append(episode_return)
    
    return np.mean(returns), np.std(returns)

def main():
    print("Starting PPO CartPole-v1 Test...")
    
    # Setup
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    print(f"Environment: {env_name}")
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {act_dim}")
    
    # Set seed for reproducibility
    seed = 0
    
    # Set seeds before creating model to ensure deterministic initialization
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    
    # Create model after setting seeds
    model_net = ActorCritic(obs_dim, act_dim)
    
    # Count and display parameters
    model_params = count_parameters(model_net)
    print(f"Model parameters: {model_params}")
    print(f"Total parameters: {model_params['total']}")
    
    # PPO settings
    settings = {
        'device': device,
        'lr': 3e-4,
        'gamma': 0.99,
        'lambda': 0.95,
        'clip_eps': 0.2,
        'target_kl': 0.01,
        'max_grad_norm': 0.5,
        'ppo_epochs': 4,
        'batch_size': 64,
        'update_timesteps': 1024,
        'val_loss_coef': 0.5,
        'ent_loss_coef': 0.01,
        'seed': seed  # Add seed to settings
    }
    
    # Create PPO agent with seed
    agent = PPOAgent(model_net, settings, seed=seed)
    
    # Training
    print("\nStarting training...")
    start_time = time.time()
    
    agent.train(env, iterations=10)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluation
    print("\nEvaluating trained policy...")
    mean_return, std_return = evaluate_policy(agent, env, num_episodes=10, seed=seed)
    
    print(f"\n=== RESULTS ===")
    print(f"Training iterations: 10")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Mean evaluation return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"Max possible return: 500")
    print(f"Performance: {mean_return/500*100:.1f}% of max")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()
