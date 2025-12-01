#!/usr/bin/env python3
"""
PPO Training Script for Warehouse Stage2 Environments

This script trains a PPO agent on the custom warehouse environment using camera observations.
"""

import warnings

import wandb
warnings.filterwarnings("ignore")

import time
import torch as th
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
from src.utils.early_stopping import EarlyStoppingCondition

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
    device = th.device(0) if th.cuda.is_available() else th.device("cpu")
    
    print(f"Using device: {device}")
    
    seed = 0
    print(f"Using seed: {seed}")
    
    # Set all seeds before creating model or environment
    set_all_seeds(seed)
    
    # Create environment
    print("\nCreating environment...")
    env = make_env(time_scale=1, no_graphics=False, verbose=True, env_type="multimodal", env_path='environment_builds/stage2/S2_Find_2Items_64x36camera120deg_rew0_20_100/Warehouse_Bot.exe', seed=seed)

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
            'gae_lambda': 0.95,
            'clip_eps': 0.2,
            'value_clip_eps': 0.2,
            'epochs': 4,
            'batch_size': 128,
            'buffer_size': 2048,
            'max_grad_norm': 0.5,
            'loss_val_coef': 0.5,
            'loss_entr_coef': 0.015,
            'weight_decay': 1e-5,
            'scheduler_step_size': 100,
            'scheduler_gamma': 0.95,
            'device': device,
            'seed': seed,
            'heatmap_logging_freq': 25,
            'eval_freq': 25,  # Evaluate every 25 iterations
            'eval_episodes': 100,  # Run 10 episodes for evaluation
            'eval_env_type': 'find',  # Use find outcome categorization
            'experiment_name': f'ppo_camera_120deg_0_20_100_find_2_items_train_0_seed_0',
            'experiment_notes': 'ppo with 120deg camera with rewards: [0, 20, 100] with task of only finding 2 items.',
        }
        training_iterations = 300

        # Create model
        model_net = ActorCriticMultimodal(act_dim, visual_obs_size=obs_dim_visual, num_items=2, device=device)
        
        # Create parameter groups and optimizer/scheduler
        param_groups = create_param_groups(model_net, visual_lr=1e-4, task_lr=1e-4, general_lr=3e-4)
        optimizer, scheduler = create_optimizer_and_lr_scheduler(param_groups, 1e-5, 100, 0.95)
        
        # Print model structure
        print(f"\nModel Structure:")
        print(model_net)
        
        # Count and display parameters
        model_params = count_parameters(model_net)
        print(f"\nModel parameters: {model_params}")
        print(f"Total parameters: {model_params['total']}")
        
        print(f"\nPPO Settings:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
        
        # Create PPO agent
        print("\nCreating PPO agent...")
        agent = PPOAgent(model_net, settings, optimizer, scheduler, 0)
        
        # Training
        print("\nStarting training...")
        start_time = time.time()
        
        # Optional: Enable early stopping (uncomment to use)
        # For PPO, early stopping is applied on the evaluation mean return
        early_stop_fn = EarlyStoppingCondition(window_size=1, metric_threshold=95.0)
        
        # Training iterations
        agent.train(env, iterations=training_iterations, early_stopping_fn=early_stop_fn)
        
        training_time = time.time() - start_time
        actual_iterations = agent.iteration
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Completed {actual_iterations} iterations (target was {training_iterations})")
        
        # Evaluation
        print("\nEvaluating trained policy...")
        mean_return, std_return, mean_steps, std_steps, ep_returns, ep_steps, eval_outcomes = evaluate_policy(
            agent.model, env, device, num_episodes=100, seed=seed, obs_type="multimodal"
        )
        
        # Logging evaluation results
        print(f"\n=== TRAINING RESULTS ===")
        print(f"Training time: {training_time:.2f}s | Mean return: {mean_return:.2f} +- {std_return:.2f}")
        
        wandb.log({
            "eval/mean_return": mean_return,
            "eval/std_return": std_return,
            "eval/mean_steps": mean_steps,
            "eval/std_steps": std_steps,
            "training/time_sec": training_time,
            "training/actual_iterations": actual_iterations
        })

        eval_table = wandb.Table(columns=["episode", "return", "steps"])
        for i, (ret, steps) in enumerate(zip(ep_returns, ep_steps)):
            eval_table.add_data(i, ret, steps)
        wandb.log({"evaluation_results": eval_table})
        
        # Save model (optional)
        try:
            experiment_name = "ppo_camera_120deg_0_20_100_find_2_items_train_0"
            save_dir = get_default_save_dir("custom", experiment_name)
            filename = create_model_filename(experiment_name, seed)
            
            model_path = save_model_checkpoint(
                model=agent.model,
                optimizer=agent.optimizer,
                save_dir=save_dir,
                filename=filename,
                settings=settings,
                seed=seed,
                training_iterations=actual_iterations,
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
