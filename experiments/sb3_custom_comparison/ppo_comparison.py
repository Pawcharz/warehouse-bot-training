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
import wandb
warnings.filterwarnings('ignore')

# Get the root directory (two levels up from this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(script_dir))

# Add src directory to path
sys.path.insert(0, ROOT_DIR)

from src.algorithms.PPO_algorithm import PPOAgent, create_optimizer_and_lr_scheduler
from src.models.actor_critic import ActorCritic

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
    
    # Put model in eval mode
    agent.model.eval()
    
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_return = 0
        done = False
        
        while not done:
            # Convert observation to tensor (handle both dict and vector observations)
            if isinstance(obs, dict):
                obs_tensor = {key: th.tensor(obs[key], dtype=th.float32, device=agent.device) 
                             for key in obs.keys()}
            else:
                obs_tensor = th.tensor(obs, dtype=th.float32, device=agent.device)
            
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
    
    # Create model and move to device
    model_net = ActorCritic(obs_dim, act_dim, device=device)
    
    # PPO settings (aligned with PPOAgent requirements)
    learning_rate = 3e-4
    settings = {
        'device': device,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_eps': 0.2,
        'value_clip_eps': 0.2,
        'max_grad_norm': 0.5,
        'epochs': 4,
        'batch_size': 64,
        'buffer_size': 1024,
        'loss_val_coef': 0.5,
        'loss_entr_coef': 0.01,
        'seed': seed,
        'reward_norm_epsilon': 1e-8,
        'intrinsic_reward_scale': 0.0,
        'icm_loss_weight': None,
    }
    
    # Create optimizer and scheduler
    param_groups = [{'params': model_net.parameters(), 'lr': learning_rate}]
    optimizer, scheduler = create_optimizer_and_lr_scheduler(
        param_groups, 
        weight_decay=1e-5
    )
    
    # Create PPO agent
    agent = PPOAgent(model_net, settings, optimizer, scheduler, start_iteration=0)
    
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
        clip_obs=np.inf,
        clip_reward=np.inf,
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
    
    # Evaluate on fresh environment without normalization
    eval_env = gym.make(env_name)
    eval_returns = []
    
    for episode in range(10):
        obs, _ = eval_env.reset(seed=seed + 1000 * episode)
        episode_return = 0
        done = False
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # VecEnv expects batched observations, but eval_env is not vectorized
            # We need to add batch dimension for the model
            obs_batch = np.expand_dims(obs, axis=0)
            action, _ = model.predict(obs_batch, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action[0])
            episode_return += reward
        
        eval_returns.append(episode_return)
    
    mean_return = np.mean(eval_returns)
    std_return = np.std(eval_returns)
    
    eval_env.close()
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
            
            # Store results for table logging later
            custom_results.append({
                'mean': custom_mean,
                'std': custom_std,
                'time': custom_time,
                'seed': seed
            })
            
            sb3_results.append({
                'mean': sb3_mean,
                'std': sb3_std,
                'time': sb3_time,
                'seed': seed
            })
            
            print(f"Custom: {custom_mean:.1f} +- {custom_std:.1f}, SB3: {sb3_mean:.1f} +- {sb3_std:.1f}")
            
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
    
    # Create detailed WandB table for per-seed results
    table_data = []
    for custom_r, sb3_r in zip(custom_results, sb3_results):
        table_data.append([
            custom_r['seed'],
            custom_r['mean'],
            custom_r['std'],
            custom_r['time'],
            sb3_r['mean'],
            sb3_r['std'],
            sb3_r['time'],
            custom_r['mean'] - sb3_r['mean'],
            'Custom' if custom_r['mean'] > sb3_r['mean'] else ('SB3' if custom_r['mean'] < sb3_r['mean'] else 'Tie'),
            custom_r['time'] / sb3_r['time']
        ])
    
    table = wandb.Table(
        columns=["Seed", "Custom Mean", "Custom Std", "Custom Time (s)", 
                 "SB3 Mean", "SB3 Std", "SB3 Time (s)", 
                 "Return Diff", "Winner", "Time Ratio"],
        data=table_data
    )
    wandb.log({f"{env_name}_detailed_results": table})

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
        'n_runs': len(custom_results),
        'custom_results': custom_results,
        'sb3_results': sb3_results
    }

def main():
    """Main comparison function"""
    print("PPO Implementation Comparison with Multiple Seeds")
    print("=" * 60)
    
    # Configuration
    seeds = range(10)
    custom_iterations = [15, 15]
    sb3_timesteps = [15 * 1024, 15 * 1024] # 15 * 1024 to match custom PPO
    
    environments = ["CartPole-v1", "Acrobot-v1"]

    print(f"Environments to test: {environments}")
    print(f"Seeds: {list(seeds)}")
    print(f"Config: {custom_iterations} iterations vs {sb3_timesteps} timesteps")
    
    # Initialize WandB
    run_name = f"ppo_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="ppo-sb3-custom-comparison",
        name=run_name,
        config={
            "environments": environments,
            "seeds": list(seeds),
            "custom_iterations": custom_iterations,
            "sb3_timesteps": sb3_timesteps,
            "num_seeds": len(list(seeds)),
        },
        tags=["comparison", "ppo", "stable-baselines3", "custom"]
    )
    
    print(f"WandB run initialized: {run_name}")
    
    # Run comparisons
    all_results = []
    
    for i, env_name in enumerate(environments):
        result = run_comparison(env_name, seeds, custom_iterations[i], sb3_timesteps[i])
        if result:
            all_results.append(result)
    
    if not all_results:
        print("\nNo successful comparisons completed")
        wandb.finish()
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
    
    # Create overall summary table for WandB
    summary_table_data = []
    
    for result in all_results:
        custom_str = f"{result['custom_mean_of_means']:.1f} +- {result['custom_std_of_means']:.1f}"
        sb3_str = f"{result['sb3_mean_of_means']:.1f} +- {result['sb3_std_of_means']:.1f}"
        speed_str = f"{result['speed_ratio']:.1f}x"
        
        print(f"{result['env_name']:<15} {custom_str:<20} {sb3_str:<20} {speed_str:<10} {result['n_runs']:<5}")
        
        # Add to WandB summary table
        summary_table_data.append([
            result['env_name'],
            result['custom_mean_of_means'],
            result['custom_std_of_means'],
            result['custom_mean_time'],
            result['sb3_mean_of_means'],
            result['sb3_std_of_means'],
            result['sb3_mean_time'],
            result['custom_mean_of_means'] - result['sb3_mean_of_means'],
            result['performance_winner'],
            result['speed_ratio'],
            result['n_runs']
        ])
    
    # Log overall summary table to WandB
    summary_table = wandb.Table(
        columns=["Environment", "Custom Mean", "Custom Std", "Custom Time (s)",
                 "SB3 Mean", "SB3 Std", "SB3 Time (s)",
                 "Mean Diff", "Winner", "Speed Ratio", "Num Seeds"],
        data=summary_table_data
    )
    wandb.log({"overall_summary": summary_table})
    
    # Log summary metrics
    custom_wins = sum(1 for r in all_results if r['performance_winner'] == 'custom')
    sb3_wins = sum(1 for r in all_results if r['performance_winner'] == 'sb3')
    ties = sum(1 for r in all_results if r['performance_winner'] == 'tie')
    
    wandb.summary['total_environments'] = len(all_results)
    wandb.summary['custom_wins'] = custom_wins
    wandb.summary['sb3_wins'] = sb3_wins
    wandb.summary['ties'] = ties
    wandb.summary['avg_speed_ratio'] = np.mean([r['speed_ratio'] for r in all_results])
    
    print(f"\n{'='*60}")
    print(f"Overall: Custom wins: {custom_wins}, SB3 wins: {sb3_wins}, Ties: {ties}")
    print(f"Average speed ratio: {np.mean([r['speed_ratio'] for r in all_results]):.2f}x")
    print(f"{'='*60}")
    
    # Close WandB run
    print(f"\nResults logged to WandB: {wandb.run.url if wandb.run else 'N/A'}")
    wandb.finish()

if __name__ == "__main__":
    main() 