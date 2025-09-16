#!/usr/bin/env python3
"""
PPO Evaluation Script for Warehouse Stage3 Environments

This script loads a PPO agent and evaluates it on the S3 (more complex) environment.
"""

import warnings
warnings.filterwarnings("ignore")

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
from src.models.actor_critic_multimodal_embedding import ActorCriticMultimodal
from src.models.model_utils import count_parameters, load_model_checkpoint
from src.utils.evaluation import evaluate_policy

def main():    
    # Setup device
    device = th.device(0) if th.cuda.is_available() else th.device("cpu")
    print(f"Using device: {device}")
    
    # Create environment
    print("\nCreating environment...")
    env = make_env(
        time_scale=1, 
        no_graphics=False, 
        verbose=True,
        env_type="multimodal", 
        env_path='environment_builds/stage3/S3_Find_2Items_Deliver_64x36camera120deg_room_big_1/Warehouse_Bot.exe'
    )

    try:
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
        
        # Set seeds before loading model
        th.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
        
        # Create model architecture (same as original)
        model_net = ActorCriticMultimodal(act_dim, visual_obs_size=obs_dim_visual, num_items=2, device=device)
        
        # Load model
        pretrained_model_path = "saved_models/custom/ppo_camera_120deg_0_20_100_find_2_items_task_embedding_attempt_1/ppo_camera_120deg_0_20_100_find_2_items_task_embedding_attempt_1_seed_0.pth"
        print(f"\nLoading model from: {pretrained_model_path}")
        
        if os.path.exists(pretrained_model_path):
            try:
                model_net, checkpoint = load_model_checkpoint(
                    model_path=pretrained_model_path,
                    model=model_net,
                    device=device,
                    load_optimizer=True
                )
                print(f"Successfully loaded model")
                print(f"Original model trained for {checkpoint.get('training_iterations', 'unknown')} iterations")
                print(f"Original final mean return: {checkpoint.get('final_mean_return', 'unknown')}")
                
            except Exception as e:
                print(f"Warning: Could not load model: {e}")
                raise e
        else:
            raise Exception("No model found")
        
        # Count and display parameters
        model_params = count_parameters(model_net)
        print(f"\nModel parameters: {model_params}")
        print(f"Total parameters: {model_params['total']}")
        
        
        # Evaluation on delivery environment
        print("\nEvaluating policy...")
        mean_return, std_return, mean_steps, std_steps = evaluate_policy(
            model_net, env, device, num_episodes=100, seed=seed, obs_type="multimodal"
        )
        
        print(f"\n=== EVALUATION RESULTS ===")
        print(f"Mean evaluation return: {mean_return:.2f} +- {std_return:.2f}")
        print(f"Mean evaluation steps: {mean_steps:.2f} +- {std_steps:.2f}")

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