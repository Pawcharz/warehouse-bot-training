#!/usr/bin/env python3
"""
Quick PPO Test Script

A simplified script to quickly test your PPO implementation against SB3
on multiple environments.
"""

import time
import numpy as np
import torch as th
import gymnasium as gym
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

# Import your custom implementations
from src.algorithms.PPO_algorithm import PPOAgent
from src.models.actor_critic import ActorCritic, ActorCriticMultimodal, count_parameters

def test_custom_ppo(env_name="CartPole-v1", iterations=10):
    """Test your custom PPO implementation"""
    print(f"Testing Custom PPO on {env_name}...")
    
    # Setup
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # Create model with separate policy and value networks
    model_net = ActorCritic(obs_dim, act_dim)

    # Count parameters
    model_params = count_parameters(model_net)
    print("Model parameters:", model_params)
    print("Total parameters:", model_params['total'])

    # Create PPO agent with separate networks
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
        'ent_loss_coef': 0.01
    }
    agent = PPOAgent(model_net, settings)
    
    # Train
    start_time = time.time()
    agent.train(env, iterations=iterations)
    training_time = time.time() - start_time
    
    # Evaluate
    eval_returns = []
    for _ in range(10):
        obs, _ = env.reset()
        episode_return = 0
        done = False
        
        while not done:
            obs_tensor = th.tensor(obs, dtype=th.float32, device=device).unsqueeze(0)
            with th.no_grad():
                action, _, _, _ = agent.model.get_action(obs_tensor, deterministic=True)
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            episode_return += reward
            done = terminated or truncated
        
        eval_returns.append(episode_return)
    
    mean_return = np.mean(eval_returns)
    std_return = np.std(eval_returns)
    
    print(f"  Custom PPO: {mean_return:.1f} ± {std_return:.1f} ({training_time:.1f}s)")
    
    env.close()
    return mean_return, std_return, training_time

def test_sb3_ppo(env_name="CartPole-v1", total_timesteps=10000):
    """Test SB3 PPO implementation"""
    print(f"Testing SB3 PPO on {env_name}...")
    
    # Create environment
    env = gym.make(env_name)
    
    # Create model with similar hyperparameters
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
        max_grad_norm=0.5
    )
    
    # Train
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    training_time = time.time() - start_time
    
    # Evaluate
    eval_returns = []
    for _ in range(10):
        obs, _ = env.reset()
        episode_return = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
        
        eval_returns.append(episode_return)
    
    mean_return = np.mean(eval_returns)
    std_return = np.std(eval_returns)
    
    print(f"  SB3 PPO:    {mean_return:.1f} ± {std_return:.1f} ({training_time:.1f}s)")
    
    env.close()
    return mean_return, std_return, training_time

def run_environment_comparison(env_name, custom_iterations, sb3_timesteps):
    """Run comparison for a single environment"""
    print(f"\n{env_name}:")
    
    # Test Custom PPO
    custom_mean, custom_std, custom_time = test_custom_ppo(env_name, iterations=custom_iterations)
    
    # Test SB3 PPO
    sb3_mean, sb3_std, sb3_time = test_sb3_ppo(env_name, total_timesteps=sb3_timesteps)
    
    # Quick comparison
    if custom_mean > sb3_mean:
        improvement = ((custom_mean - sb3_mean) / sb3_mean) * 100
        print(f"  ✅ Custom PPO {improvement:.0f}% better")
    elif sb3_mean > custom_mean:
        improvement = ((sb3_mean - custom_mean) / custom_mean) * 100
        print(f"  ❌ SB3 PPO {improvement:.0f}% better")
    else:
        print(f"  🤝 Similar performance")
    
    speed_ratio = custom_time / sb3_time
    print(f"  Speed: {speed_ratio:.1f}x {'slower' if speed_ratio > 1 else 'faster'}")
    
    return {
        'env_name': env_name,
        'custom_mean': custom_mean,
        'custom_std': custom_std,
        'custom_time': custom_time,
        'sb3_mean': sb3_mean,
        'sb3_std': sb3_std,
        'sb3_time': sb3_time,
        'performance_winner': 'custom' if custom_mean > sb3_mean else 'sb3' if sb3_mean > custom_mean else 'tie',
        'speed_ratio': speed_ratio
    }

def main():
    """Run the comparison across multiple environments"""
    print("PPO Implementation Comparison")
    print("=" * 50)
    
    # Configuration
    custom_iterations = 10
    sb3_timesteps = 10000
    
    environments = [
        "CartPole-v1",
        "Acrobot-v1"
    ]
    
    print(f"Config: {custom_iterations} iterations vs {sb3_timesteps} timesteps")
    
    # Store results
    all_results = []
    
    # Test each environment
    for env_name in environments:
        try:
            result = run_environment_comparison(env_name, custom_iterations, sb3_timesteps)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Error testing {env_name}: {str(e)}")
            continue
    
    # Summary
    if not all_results:
        print("❌ No successful tests completed")
        return
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    custom_wins = sum(1 for r in all_results if r['performance_winner'] == 'custom')
    sb3_wins = sum(1 for r in all_results if r['performance_winner'] == 'sb3')
    ties = sum(1 for r in all_results if r['performance_winner'] == 'tie')
    
    avg_custom_return = np.mean([r['custom_mean'] for r in all_results])
    avg_sb3_return = np.mean([r['sb3_mean'] for r in all_results])
    avg_speed_ratio = np.mean([r['speed_ratio'] for r in all_results])
    
    print(f"Environments tested: {len(all_results)}")
    print()
    print(f"Average Returns:")
    print(f"  Custom PPO: {avg_custom_return:.2f}")
    print(f"  SB3 PPO: {avg_sb3_return:.2f}")
    print()
    print(f"Average speed ratio: {avg_speed_ratio:.2f}x")
    print(f"  (Custom PPO is {'slower' if avg_speed_ratio > 1 else 'faster'} on average)")
    
    # Detailed results table
    print(f"\n{'='*80}")
    print("DETAILED RESULTS")
    print(f"{'='*80}")
    print(f"{'Environment':<15} {'Custom Return':<15} {'SB3 Return':<15} {'Winner':<10} {'Speed Ratio':<12}")
    print(f"{'-'*80}")
    
    for result in all_results:
        winner_symbol = {
            'custom': '✅ Custom',
            'sb3': '❌ SB3',
            'tie': '🤝 Tie'
        }[result['performance_winner']]
        
        print(f"{result['env_name']:<15} {result['custom_mean']:<15.2f} {result['sb3_mean']:<15.2f} {winner_symbol:<10} {result['speed_ratio']:<12.2f}x")

def example_vector_only():
    """Example for vector-only observations"""
    print("=== Vector-only Example ===")
    
    # Create model with separate policy and value networks for vector observations
    obs_dim = 128  # Example observation dimension
    act_dim = 4    # Example action space size
    
    model_net = ActorCritic(obs_dim, act_dim)
    
    # Count parameters
    model_params = count_parameters(model_net)
    print("Model parameters:", model_params)
    print("Total parameters:", model_params['total'])
    
    # Create PPO agent
    settings = {
        'device': 'cuda' if th.cuda.is_available() else 'cpu',
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
        'ent_loss_coef': 0.01
    }
    agent = PPOAgent(model_net, settings)
    print("Vector-only PPO Agent created with separate policy and value networks!")

if __name__ == "__main__":
    main()