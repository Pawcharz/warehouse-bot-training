#!/usr/bin/env python3
"""
PPO Training Script for Warehouse Stage2 Environments

This script trains a PPO agent on the custom warehouse environment using camera observations.
"""

import warnings
warnings.filterwarnings("ignore")

import time
import torch as th
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
from src.algorithms.PPO_algorithm import PPOAgent, create_optimizer_and_lr_scheduler

from src.models.actor_critic_multimodal_embedding import ActorCriticMultimodal
from src.utils.seed_utils import set_all_seeds
from src.models.model_utils import count_parameters, save_model_checkpoint, create_model_filename, get_default_save_dir
from src.utils.evaluation import evaluate_policy
from src.models.intrinsic_curiosity_module import ActorCriticWithICM, IntrinsicCuriosityModule

def create_param_groups(model, visual_lr, task_lr, general_lr, icm = None, icm_lr = None):
    
    visual_params = list(model.visual_encoder_cnn.parameters()) + list(model.visual_encoder_mlp.parameters())
    task_params = list(model.task_encoder.parameters())
    general_params = list(model.policy_net.parameters()) + list(model.value_net.parameters())
    
    
    param_groups = [
        {'params': visual_params, 'lr': visual_lr, 'name': 'visual_encoder'},
        {'params': task_params, 'lr': task_lr, 'name': 'task_encoder'},
        {'params': general_params, 'lr': general_lr, 'name': 'policy_value'}
    ]
    
    if icm is not None:
        icm_params = list(icm.parameters())
        param_groups.append({'params': icm_params, 'lr': icm_lr, 'name': 'icm'})
    
    return param_groups

def main():
    print("Starting PPO Training for Warehouse Stage2...")
    
    # Setup device
    device = th.device(0) if th.cuda.is_available() else th.device("cpu")
    
    print(f"Using device: {device}")
    
    # Set seed for reproducibility FIRST, before creating anything
    seed = 0
    print(f"Using seed: {seed}")
    
    # Set all seeds before creating model or environment
    set_all_seeds(seed)
    
    # Create environment
    print("\nCreating environment...")
    env = make_env(time_scale=1, no_graphics=False, verbose=True, env_type="multimodal", env_path='environment_builds/stage3/S3_Find_2Items_64x36camera120deg_obstacles_1_stackedObs_x5/Warehouse_Bot.exe', seed=seed)

    try:
        print(env.observation_space)
        # Get environment dimensions
        obs_dim_visual = env.observation_space['visual'].shape
        obs_dim_vector = env.observation_space['vector'].shape[0]
        act_dim = env.action_space.n
        
        print(f"Vector observation dimension: {obs_dim_vector}")
        print(f"Visual observation dimension: {obs_dim_visual}")
        print(f"Action dimension: {act_dim}")
        
        # PPO settings
        settings = {
            'gamma': 0.99,
            'lambda': 0.95,
            'clip_eps': 0.2,
            'value_clip_eps': 0.2,
            'ppo_epochs': 4,
            'batch_size': 128,
            'buffer_size': 2048,
            'max_grad_norm': 0.5,
            'val_loss_coef': 0.5,
            'icm_loss_weight': 0.1,
            'ent_loss_coef': 0.01,
            'icm_eta': 0.01,
            'icm_beta': 0.6,
            'icm_normalizer_gamma': 0.995,
            'intrinsic_reward_scale': 0.05,
            'weight_decay': 1e-5,
            'scheduler_step_size': 100,
            'scheduler_gamma': 0.95,
            'device': device,
            'seed': seed,
            'experiment_name': f'icm_module_test_small_env_with_textures_stackedObs_x5',
            'experiment_notes': 'ppo with 120deg camera with rewards: [0, 20, 100] with task of only finding 2 items and ICM module on environment with more complex textures and obstacles',
        }
        training_iterations = 200

        # Create model
        model_net = ActorCriticMultimodal(act_dim, visual_obs_size=obs_dim_visual, num_items=2, device=device)
        icm_eta = settings['icm_eta']
        icm_beta = settings['icm_beta']
        icm = IntrinsicCuriosityModule(feature_dim=model_net.fusion_size, action_dim=act_dim, eta=icm_eta, beta=icm_beta, device=device)
        
        model = ActorCriticWithICM(model_net, icm)
        # Create parameter groups and optimizer/scheduler
        param_groups = create_param_groups(model_net, visual_lr=1e-4, task_lr=1e-4, general_lr=3e-4, icm=icm, icm_lr=1e-4)
        optimizer, scheduler = create_optimizer_and_lr_scheduler(
            param_groups, 
            weight_decay=settings['weight_decay'],
            scheduler_step_size=settings['scheduler_step_size'],
            scheduler_gamma=settings['scheduler_gamma']
        )
        
        # Print model structure
        print(f"\nModel Structure:")
        print(model)
        
        # Count and display parameters
        model_params = count_parameters(model.actor_critic)
        icm_params = count_parameters(icm)
        print(f"\nICM parameters: {icm_params}")
        print(f"Total ICM parameters: {icm_params['total']}")
        print(f"\nModel parameters: {model_params}")
        print(f"Total model parameters: {model_params['total']}")
        
        print(f"\nPPO Settings:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
        print(f"ICM: eta: {icm_eta}, beta: {icm_beta}")
        
        # Create PPO agent
        print("\nCreating PPO agent...")
        agent = PPOAgent(model, settings, optimizer, scheduler, 0)
        
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
            agent.model, env, device, num_episodes=100, seed=seed, obs_type="multimodal"
        )
        
        print(f"\n=== TRAINING RESULTS ===")
        print(f"Training iterations: {training_iterations}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Mean evaluation return: {mean_return:.2f} +- {std_return:.2f}")
        print(f"Mean evaluation steps: {mean_steps:.2f} +- {std_steps:.2f}")
        
        # Save model (optional)
        try:
            save_dir = get_default_save_dir("custom", "icm_module_performance_test_complex_env_02_10_2025_stackedObs_x5")
            filename = create_model_filename("icm_module_performance_test_complex_env_02_10_2025_stackedObs_x5", seed)
            
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
