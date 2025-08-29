#!/usr/bin/env python3
"""
PPO Training Script for Warehouse Stage2 Environments

This script trains a PPO agent on the custom warehouse environment using camera observations.
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
from src.environments.env_utils import make_env

# Algorithm imports
from src.algorithms.PPO_algorithm import PPOAgent
# from src.algorithms.PPO_algorithm_returns_clipping import PPOAgent
from src.models.actor_critic_multimodal import ActorCriticMultimodal
from src.models.model_utils import count_parameters, save_model_checkpoint, create_model_filename, get_default_save_dir
from src.utils.evaluation import evaluate_policy

def main():
    print("Starting PPO Training for Warehouse Stage2...")
    
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
    # env = make_env(time_scale=1, no_graphics=False, verbose=True, env_type="multimodal", env_path='environment_builds/stage2/S2_Find_Items_64x36camera120deg_rew0_100/Warehouse_Bot.exe')
    env = make_env(time_scale=2, no_graphics=True, verbose=True, env_type="multimodal", env_path='environment_builds/stage2/S2_Find_Items_64x36camera120deg_rew0_20_100/Warehouse_Bot.exe')

    try:
        print(env.observation_space)
        # Get environment dimensions
        obs_dim_visual = env.observation_space['visual'].shape
        obs_dim_vector = env.observation_space['vector'].shape[0]
        act_dim = env.action_space.n
        
        print(f"Observation dimension: {obs_dim_vector}")
        print(f"Observation dimension: {obs_dim_visual}")
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
            'batch_size': 64,
            'update_timesteps': 1024,
            'lr': 3e-4, 
            'visual_lr': 1e-4,
            'vector_lr': 1e-4,
            'max_grad_norm': 0.5,
            'val_loss_coef': 0.5,
            'ent_loss_coef': 0.015,
            # 'scheduler_step_size': 50,
            # 'scheduler_gamma': 0.90,
            'weight_decay': 1e-5,
            'device': device,
            'seed': seed,
            'value_clip_eps': 0.2,
            'use_wandb': True,  # Changed from use_tensorboard
            'experiment_name': f'test_wandb_1',
            'experiment_notes': 'test wandb logging',
            'histogram_logging_interval': 5,
        }
        training_iterations = 12

        # Create model
        model_net = ActorCriticMultimodal(act_dim, visual_obs_size=obs_dim_visual, vector_obs_size=obs_dim_vector, device=device)
        
        # Count and display parameters
        model_params = count_parameters(model_net)
        print(f"\nModel parameters: {model_params}")
        print(f"Total parameters: {model_params['total']}")
        
        print(f"\nPPO Settings:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
        
        # Create PPO agent
        print("\nCreating PPO agent...")
        agent = PPOAgent(model_net, settings)
        
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
            agent, env, num_episodes=5, seed=seed, obs_type="multimodal"
        )
        
        print(f"\n=== TRAINING RESULTS ===")
        print(f"Training iterations: {training_iterations}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Mean evaluation return: {mean_return:.2f} +- {std_return:.2f}")
        print(f"Mean evaluation steps: {mean_steps:.2f} +- {std_steps:.2f}")
        
        # Save model (optional)
        try:
            save_dir = get_default_save_dir("custom", "test")
            filename = create_model_filename("tensorboard_test", seed)
            
            model_path = save_model_checkpoint(
                model=agent.model,
                optimizer=agent.optimizer,
                save_dir=save_dir,
                filename=filename,
                settings=settings,
                seed=seed,
                training_iterations=training_iterations,
                final_mean_return=mean_return,
                final_std_return=std_return
            )
        except Exception as e:
            print(f"Could not save model: {e}")
    
        print("\nTraining script completed!")

    except KeyboardInterrupt:
        print("\nReceived Ctrl+C! Closing environment safely...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Closing environment...")
    finally:
        # Always close environment, whether training completed or was interrupted
        try:
            env.close()
            print("Environment closed successfully.")
        except Exception as e:
            print(f"Error closing environment: {e}")
    
    

if __name__ == "__main__":
    main()
