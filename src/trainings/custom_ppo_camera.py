#!/usr/bin/env python3
"""
PPO Training Script for Warehouse Stage2 Environments

This script trains a PPO agent on the custom warehouse environment using camera observations.
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
from src.algorithms.PPO_algorithm import PPOAgent, create_optimizer_and_scheduler
from src.models.actor_critic_multimodal_embedding import ActorCriticMultimodal
from src.models.model_utils import count_parameters, save_model_checkpoint, create_model_filename, get_default_save_dir
from src.utils.evaluation import evaluate_policy

def create_param_groups(model, visual_lr, task_lr, general_lr):
    
    visual_params = list(model.visual_encoder_cnn.parameters()) + list(model.visual_encoder_mlp.parameters())
    task_params = list(model.task_encoder.parameters())
    general_params = list(model.policy_net.parameters()) + list(model.value_net.parameters())
    
    param_groups = [
        {'params': visual_params, 'lr': visual_lr, 'name': 'visual_encoder'},
        {'params': task_params, 'lr': task_lr, 'name': 'task_encoder'},
        {'params': general_params, 'lr': general_lr, 'name': 'policy_value'}
    ]
    
    return param_groups

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
    env = make_env(time_scale=1, no_graphics=False, verbose=True, env_type="multimodal", env_path='environment_builds/stage2/S2_Find_2Items_64x36camera120deg_rew0_100/Warehouse_Bot.exe')
    # env = make_env(time_scale=3, no_graphics=True, verbose=True, env_type="multimodal", env_path='environment_builds/stage2/S2_Find_2Items_64x36camera120deg_rew0_20_100/Warehouse_Bot.exe')

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
            'batch_size': 128,
            'update_timesteps': 2048,
            'max_grad_norm': 0.5,
            'val_loss_coef': 0.5,
            'ent_loss_coef': 0.015,
            'weight_decay': 1e-5,
            'scheduler_step_size': 100,
            'scheduler_gamma': 0.95,
            'device': device,
            'seed': seed,
            'value_clip_eps': 0.2,
            'experiment_name': f'ppo_camera_120deg_0_20_100_find_2_items_task_embedding_attempt_1',
            'experiment_notes': 'ppo with 120deg camera with rewards: [0, 20, 100] with task of only finding 2 items',
        }
        training_iterations = 200

        # Create model
        model_net = ActorCriticMultimodal(act_dim, visual_obs_size=obs_dim_visual, num_items=2, device=device)
        
        # Create parameter groups and optimizer/scheduler
        param_groups = create_param_groups(model_net, visual_lr=1e-4, task_lr=1e-4, general_lr=3e-4)
        optimizer, scheduler = create_optimizer_and_scheduler(param_groups, settings)
        
        # Count and display parameters
        model_params = count_parameters(model_net)
        print(f"\nModel parameters: {model_params}")
        print(f"Total parameters: {model_params['total']}")
        
        print(f"\nPPO Settings:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
        
        # Create PPO agent
        print("\nCreating PPO agent...")
        agent = PPOAgent(model_net, optimizer, scheduler, settings)
        
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
            save_dir = get_default_save_dir("custom", "ppo_camera_120deg_0_20_100_find_2_items_task_embedding_attempt_1")
            filename = create_model_filename("ppo_camera_120deg_0_20_100_find_2_items_task_embedding_attempt_1", seed)
            
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
