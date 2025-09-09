#!/usr/bin/env python3
"""
PPO Delivery Training Script for Warehouse Stage2 Environments

This script loads a pre-trained PPO agent and continues training on the delivery environment.
The delivery environment includes both finding items and delivering them to specified locations.

Environment: S2_Find_2Items_Deliver_64x36camera120deg_rew0_20_100_100
Rewards: [0, 20, 100, 100] for [timeout, wrong_item, correct_item, successful_delivery]
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
from src.models.actor_critic_multimodal_embedding import ActorCriticMultimodal
from src.models.model_utils import count_parameters, save_model_checkpoint, create_model_filename, get_default_save_dir, load_model_checkpoint
from src.utils.evaluation import evaluate_policy

def main():
    print("Starting PPO Delivery Training for Warehouse Stage2...")
    
    # Setup device
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        th.device(0)
        if th.cuda.is_available() and not is_fork
        else th.device("cpu")
    )
    print(f"Using device: {device}")
    
    # Create delivery environment
    print("\nCreating delivery environment...")
    env = make_env(
        time_scale=1, 
        no_graphics=False, 
        verbose=True, 
        env_type="multimodal", 
        env_path='environment_builds/stage2/S2_Find_2Items_Deliver_64x36camera120deg_rew0_20_100_100/Warehouse_Bot.exe'
    )

    try:
        print(env.observation_space)
        # Get environment dimensions
        obs_dim_visual = env.observation_space['visual'].shape
        obs_dim_vector = env.observation_space['vector'].shape[0]
        act_dim = env.action_space.n
        
        print(f"Vector observation dimension: {obs_dim_vector}")
        print(f"Visual observation dimension: {obs_dim_visual}")
        print(f"Action dimension: {act_dim}")
        
        # Set seed for reproducibility
        seed = 0
        print(f"Using seed: {seed}")
        
        # Set seeds before creating model
        th.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
        
        # PPO settings for delivery training
        settings = {
            'gamma': 0.99,
            'lambda': 0.95,
            'clip_eps': 0.2,
            'ppo_epochs': 4,
            'batch_size': 128,
            'update_timesteps': 2048,
            'lr': 3e-4, 
            'visual_lr': 1e-4,
            'vector_lr': 1e-4,
            'max_grad_norm': 0.5,
            'val_loss_coef': 0.5,
            'ent_loss_coef': 0.015,
            'weight_decay': 1e-5,
            'device': device,
            'seed': seed,
            'value_clip_eps': 0.2,
            'experiment_name': f'ppo_camera_120deg_0_20_100_find_2_items_deliver_task_embedding_attempt_1',
            'experiment_notes': 'PPO delivery training with 120deg camera, rewards: [0, 20, 100, 100], find and deliver item task with 2 items',
        }
        training_iterations = 300
        
        # Create model architecture (same as original)
        model_net = ActorCriticMultimodal(act_dim, visual_obs_size=obs_dim_visual, num_items=2, device=device)
        
        # Load pre-trained model
        pretrained_model_path = "saved_models/custom/ppo_camera_120deg_0_20_100_find_2_items_task_embedding_attempt_1_default_weights/ppo_camera_120deg_0_20_100_find_2_items_task_embedding_attempt_1_default_weights_seed_0.pth"
        print(f"\nLoading pre-trained model from: {pretrained_model_path}")
        
        if os.path.exists(pretrained_model_path):
            try:
                model_net, checkpoint = load_model_checkpoint(
                    model_path=pretrained_model_path,
                    model=model_net,
                    device=device,
                    load_optimizer=False  # Create new optimizer for delivery training
                )
                print(f"Successfully loaded pre-trained model")
                print(f"Original model trained for {checkpoint.get('training_iterations', 'unknown')} iterations")
                print(f"Original final mean return: {checkpoint.get('final_mean_return', 'unknown')}")
                
            except Exception as e:
                print(f"Warning: Could not load pre-trained model: {e}")
                raise e
        else:
            raise Exception("No pre-trained model found")
        
        # Count and display parameters
        model_params = count_parameters(model_net)
        print(f"\nModel parameters: {model_params}")
        print(f"Total parameters: {model_params['total']}")
        
        print(f"\nPPO Delivery Training Settings:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
        
        # Create PPO agent with the loaded model
        print("\nCreating PPO agent for delivery training...")
        agent = PPOAgent(model_net, settings)
        
        # Training on delivery environment
        print("\nStarting delivery training...")
        start_time = time.time()
        
        # Training iterations - get from checkpoint data
        pretrained_iterations = checkpoint.get('training_iterations', 0)
        agent.train(env, iterations=training_iterations, start_iteration=pretrained_iterations)
        
        training_time = time.time() - start_time
        print(f"\nDelivery training completed in {training_time:.2f} seconds")
        
        # Evaluation on delivery environment
        print("\nEvaluating delivery policy...")
        mean_return, std_return, mean_steps, std_steps = evaluate_policy(
            agent, env, num_episodes=100, seed=seed, obs_type="multimodal"
        )
        
        print(f"\n=== DELIVERY TRAINING RESULTS ===")
        print(f"Training iterations: {training_iterations}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Mean evaluation return: {mean_return:.2f} +- {std_return:.2f}")
        print(f"Mean evaluation steps: {mean_steps:.2f} +- {std_steps:.2f}")
        
        # Save delivery model with new name and path
        try:
            save_dir = get_default_save_dir("custom", "delivery")  # Use custom/delivery subdirectory
            filename = create_model_filename("ppo_delivery_camera_120deg_0_20_100_100_find_deliver_2_items_attempt_1", seed)
            
            model_path = save_model_checkpoint(
                model=agent.model,
                optimizer=agent.optimizer,
                save_dir=save_dir,
                filename=filename,
                settings=settings,
                seed=seed,
                training_iterations=training_iterations,
                final_mean_return=mean_return,
                final_std_return=std_return,
                additional_info={
                    'pretrained_from': pretrained_model_path,
                    'environment_type': 'delivery',
                    'task_description': 'find_and_deliver_2_items'
                }
            )
            print(f"Delivery model saved to: {model_path}")
        except Exception as e:
            print(f"Could not save delivery model: {e}")
    
        print("\nDelivery training script completed!")

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