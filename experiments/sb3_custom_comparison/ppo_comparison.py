#!/usr/bin/env python3
"""
PPO Implementation Comparison with Multiple Seeds

Runs custom and SB3 PPO implementations on multiple seeds and provides
statistical comparison of results.
"""

import time
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import warnings
import random
import sys
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings('ignore')

# Get the root directory (two levels up from this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(script_dir))

# Add src directory to path
sys.path.insert(0, ROOT_DIR)

from src.algorithms.PPO_algorithm import PPOAgent
from src.models.actor_critic import ActorCritic

# Setup tensorboard logging
log_dir = os.path.join(script_dir, "tensorboard_logs")
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

def evaluate_policy(agent, env, num_episodes=10, seed=0):
    """Evaluate policy and return mean/std of returns"""
    returns = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_return = 0
        done = False
        
        while not done:
            obs_tensor = th.tensor(obs, dtype=th.float32, device=agent.device).unsqueeze(0)
            with th.no_grad():
                action, _, _, _ = agent.model.get_action(obs_tensor, deterministic=True)
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            episode_return += reward
            done = terminated or truncated
        
        returns.append(episode_return)
    
    return np.mean(returns), np.std(returns)

def test_custom_ppo(env_name, seed, iterations=10):
    """Test custom PPO implementation"""
    set_seed(seed)
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model_net = ActorCritic(obs_dim, act_dim)
    
    settings = {
        'device': device,
        'lr': 3e-4,
        'gamma': 0.99,
        'lambda': 0.95,
        'clip_eps': 0.2,
        'max_grad_norm': 0.5,
        'ppo_epochs': 4,
        'batch_size': 64,
        'update_timesteps': 1024,
        'val_loss_coef': 0.5,
        'ent_loss_coef': 0.01,
        'seed': seed,
        'use_tensorboard': False,
    }
    agent = PPOAgent(model_net, settings, seed=seed)
    
    start_time = time.time()
    agent.train(env, iterations=iterations)
    training_time = time.time() - start_time
    
    mean_return, std_return = evaluate_policy(agent, env, num_episodes=10, seed=seed)
    
    env.close()
    return mean_return, std_return, training_time

def test_sb3_ppo(env_name, seed, total_timesteps=10240):
    """Test SB3 PPO implementation"""
    set_seed(seed)
    
    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])
    
    # Add reward normalization only
    env = VecNormalize(
        env,
        norm_obs=False,
        norm_reward=True, # Only normalize rewards
        clip_obs=np.inf, # No clipping
        clip_reward=np.inf, # No clipping
    )
    

    # Policy kwargs to match custom network architecture exactly
    policy_kwargs = {
        "net_arch": {
            "pi": [128, 128],
            "vf": [128, 128]
        },
        "activation_fn": nn.Tanh,
        "ortho_init": True,
    }
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        clip_range=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        policy_kwargs=policy_kwargs
    )
    
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    training_time = time.time() - start_time
    
    # Evaluate
    env.norm_reward = False  # Disable reward normalization for evaluation
    eval_returns = []
    for episode in range(10):
        obs = env.reset()
        episode_return = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_return += reward[0]
            done = done[0]
        
        eval_returns.append(episode_return)
    
    mean_return = np.mean(eval_returns)
    std_return = np.std(eval_returns)
    
    env.close()
    return mean_return, std_return, training_time

def run_comparison(env_name, seeds, custom_iterations=10, sb3_timesteps=10240):
    """Run comparison for one environment across multiple seeds"""
    print(f"\n{env_name}:")
    
    custom_results = []
    sb3_results = []
    
    for i, seed in enumerate(seeds):
        print(f"  Seed {seed} ({i+1}/{len(seeds)})...", end=" ")
        
        try:
            custom_mean, custom_std, custom_time = test_custom_ppo(env_name, seed, custom_iterations)
            sb3_mean, sb3_std, sb3_time = test_sb3_ppo(env_name, seed, sb3_timesteps)

            writer.add_scalars(f'{env_name}/mean_returns', {
                'Custom_PPO': custom_mean,
                'SB3_PPO': sb3_mean
            }, i)
            writer.add_scalars(f'{env_name}/mean_stds', {
                'Custom_PPO': custom_std,
                'SB3_PPO': sb3_std
            }, i)
            writer.add_scalars(f'{env_name}/mean_times', {
                'Custom_PPO': custom_time,
                'SB3_PPO': sb3_time
            }, i)
            
            custom_results.append({
                'mean': custom_mean,
                'std': custom_std,
                'time': custom_time
            })
            
            sb3_results.append({
                'mean': sb3_mean,
                'std': sb3_std,
                'time': sb3_time
            })
            
        except Exception as e:
            print(f"Error: {str(e)}")
            continue
    
    if not custom_results or not sb3_results:
        print(f"No successful runs for {env_name}")
        return None
    
    # Calculate statistics
    custom_means = [r['mean'] for r in custom_results]
    custom_stds = [r['std'] for r in custom_results]
    custom_times = [r['time'] for r in custom_results]
    
    sb3_means = [r['mean'] for r in sb3_results]
    sb3_stds = [r['std'] for r in sb3_results]
    sb3_times = [r['time'] for r in sb3_results]
    
    # Overall statistics
    custom_mean_of_means = np.mean(custom_means)
    custom_std_of_means = np.std(custom_means)
    custom_mean_of_stds = np.mean(custom_stds)
    custom_mean_time = np.mean(custom_times)
    
    sb3_mean_of_means = np.mean(sb3_means)
    sb3_std_of_means = np.std(sb3_means)
    sb3_mean_of_stds = np.mean(sb3_stds)
    sb3_mean_time = np.mean(sb3_times)
    
    print(f"  Custom PPO: {custom_mean_of_means:.1f} +- {custom_std_of_means:.1f} (eval_std: {custom_mean_of_stds:.1f}, time: {custom_mean_time:.1f}s)")
    print(f"  SB3 PPO:    {sb3_mean_of_means:.1f} +- {sb3_std_of_means:.1f} (eval_std: {sb3_mean_of_stds:.1f}, time: {sb3_mean_time:.1f}s)")
    
    writer.add_scalars(f'{env_name}/mean_returns/all_runs', {
        'Custom_PPO': custom_mean_of_means,
        'SB3_PPO': sb3_mean_of_means
    }, 0)
    writer.add_scalars(f'{env_name}/mean_stds/all_runs', {
        'Custom_PPO': custom_std_of_means,
        'SB3_PPO': sb3_std_of_means
    }, 0)
    writer.add_scalars(f'{env_name}/mean_times/all_runs', {
        'Custom_PPO': custom_mean_time,
        'SB3_PPO': sb3_mean_time
    }, 0)
    writer.close()

    return {
        'env_name': env_name,
        'custom_mean_of_means': custom_mean_of_means,
        'custom_std_of_means': custom_std_of_means,
        'custom_mean_of_stds': custom_mean_of_stds,
        'custom_mean_time': custom_mean_time,
        'sb3_mean_of_means': sb3_mean_of_means,
        'sb3_std_of_means': sb3_std_of_means,
        'sb3_mean_of_stds': sb3_mean_of_stds,
        'sb3_mean_time': sb3_mean_time,
        'performance_winner': 'custom' if custom_mean_of_means > sb3_mean_of_means else 'sb3' if sb3_mean_of_means > custom_mean_of_means else 'tie',
        'speed_ratio': custom_mean_time / sb3_mean_time,
        'n_runs': len(custom_results)
    }

def main():
    """Main comparison function"""
    print("PPO Implementation Comparison with Multiple Seeds")
    print("=" * 60)
    
    # Configuration
    seeds = range(20)
    custom_iterations = [10, 10]
    sb3_timesteps = [10240, 10240] # 10 * 1024 to match custom PPO
    
    environments = ["CartPole-v1", "Acrobot-v1"]

    print(f"Environments to test: {environments}")
    print(f"Seeds: {seeds}")

    print(f"Config: {custom_iterations} iterations vs {sb3_timesteps} timesteps")
    print(f"Tensorboard logs saved to: {log_dir}")
    
    # Run comparisons
    all_results = []
    
    for i, env_name in enumerate(environments):
        result = run_comparison(env_name, seeds, custom_iterations[i], sb3_timesteps[i])
        if result:
            all_results.append(result)
    
    if not all_results:
        print("\nNo successful comparisons completed")
        return
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    # Detailed results table
    print(f"\n{'='*100}")
    print("DETAILED RESULTS")
    print(f"{'='*100}")
    print(f"{'Environment':<15} {'Custom (mean +- std)':<20} {'SB3 (mean +- std)':<20} {'Speed':<10} {'Runs':<5}")
    print(f"{'-'*100}")
    
    for result in all_results:
        
        custom_str = f"{result['custom_mean_of_means']:.1f}+-{result['custom_std_of_means']:.1f}"
        sb3_str = f"{result['sb3_mean_of_means']:.1f}+-{result['sb3_std_of_means']:.1f}"
        speed_str = f"{result['speed_ratio']:.1f}x"
        
        print(f"{result['env_name']:<15} {custom_str:<20} {sb3_str:<20} {speed_str:<10} {result['n_runs']:<5}")

if __name__ == "__main__":
    main() 