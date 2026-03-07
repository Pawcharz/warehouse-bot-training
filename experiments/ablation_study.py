#!/usr/bin/env python3
"""
Ablation Study: Test different model architectures.
Uses existing PPO training logic with proper WandB logging.
"""

import warnings
warnings.filterwarnings("ignore")

import time
import torch as th
import os
import sys
import wandb

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from config import ROOT_DIR
from src.environments.env_utils import make_env
from src.algorithms.PPO_algorithm import PPOAgent, create_optimizer_and_lr_scheduler
from src.models.actor_critic_multimodal_configurable import ActorCriticMultimodalConfigurable
from src.utils.seed_utils import set_all_seeds
from src.models.model_utils import count_parameters, save_model_checkpoint, create_model_filename, get_default_save_dir
from src.utils.evaluation import evaluate_policy

# Experiment configurations
CONFIGS = {
    "baseline": {"visual_encoder_blocks": 4, "task_embedding_dim": 32, "fusion_type": "concat"},
    "visual_shallow": {"visual_encoder_blocks": 3, "task_embedding_dim": 32, "fusion_type": "concat"},
    "visual_deep": {"visual_encoder_blocks": 5, "task_embedding_dim": 32, "fusion_type": "concat"},
    "embedding_small": {"visual_encoder_blocks": 4, "task_embedding_dim": 16, "fusion_type": "concat"},
    "embedding_large": {"visual_encoder_blocks": 4, "task_embedding_dim": 64, "fusion_type": "concat"},
}

def train_config(config_name, config, env_path, iterations=300, seed=0):
    """Train a single configuration using PPOAgent.train()."""
    
    print(f"\n{'='*80}\nEXPERIMENT: {config_name}\n{'='*80}")
    
    device = th.device(0) if th.cuda.is_available() else th.device("cpu")
    set_all_seeds(seed)
    
    # Create environment
    env = make_env(time_scale=1, no_graphics=False, verbose=False, 
                   env_type="multimodal", env_path=env_path, seed=seed)
    
    try:
        obs_dim_visual = env.observation_space['visual'].shape
        act_dim = env.action_space.n
        
        # Create model
        model = ActorCriticMultimodalConfigurable(
            act_dim, visual_obs_size=obs_dim_visual, num_items=2,
            visual_encoder_blocks=config['visual_encoder_blocks'],
            task_embedding_dim=config['task_embedding_dim'],
            fusion_type=config['fusion_type'], device=device
        )
        
        params = count_parameters(model)
        print(f"Parameters: {params['total']:,}")
        
        # PPO settings with WandB for ablation study
        settings = {
            # PPO hyperparameters
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
            
            # Evaluation
            'eval_freq': 25,
            'eval_episodes': 100,
            'eval_env_type': 'find',  # Use find outcome categorization
            'eval_initial': True,  # Evaluate at iteration 0 (before training) for complete plot
            
            # WandB - separate project for ablation
            'wandb_project': 'warehouse-bot-ablation',
            'wandb_entity': os.getenv('WANDB_ENTITY'),
            'wandb_api_key': os.getenv('WANDB_API_KEY'),
            'experiment_name': config_name,
            'wandb_tags': ['ablation'],
        }
        
        # Setup optimizer with parameter groups
        visual_params = list(model.visual_encoder_cnn.parameters()) + list(model.visual_encoder_mlp.parameters())
        task_params = list(model.task_encoder.parameters())
        general_params = list(model.policy_net.parameters()) + list(model.value_net.parameters())
        
        param_groups = [
            {'params': visual_params, 'lr': 1e-4, 'name': 'visual'},
            {'params': task_params, 'lr': 1e-4, 'name': 'task'},
            {'params': general_params, 'lr': 3e-4, 'name': 'policy_value'}
        ]
        
        optimizer, scheduler = create_optimizer_and_lr_scheduler(param_groups, 1e-5, 100, 0.95)
        
        # Create PPO agent (WandB logger is initialized inside)
        agent = PPOAgent(model, settings, optimizer, scheduler, 0)
        
        # Add architecture config to WandB
        if agent.logger and agent.logger.wandb_run:
            agent.logger.wandb_run.config.update({
                **config,
                'total_params': params['total']
            })
        
        # Train using existing PPO logic
        print(f"\nStarting training for {iterations} iterations...")
        start_time = time.time()
        agent.train(env, iterations=iterations)
        training_time = time.time() - start_time
        
        # Final evaluation
        print("\nFinal evaluation...")
        eval_mean, eval_std, _, _, _, _, _ = evaluate_policy(
            model, env, device, num_episodes=100, seed=seed, obs_type="multimodal", verbose=False
        )
        
        # Log summary metrics to WandB (use summary instead of log for final metrics)
        if agent.logger and agent.logger.wandb_run:
            agent.logger.wandb_run.summary.update({
                'summary/final_eval_return': eval_mean,
                'summary/final_eval_std': eval_std,
                'summary/total_params': params['total'],
                'summary/training_time_minutes': training_time / 60
            })
        
        print(f"\nExperiment {config_name} completed!")
        print(f"Final eval: {eval_mean:.2f} ± {eval_std:.2f}")
        print(f"Training time: {training_time/60:.1f} minutes")
        
        # Save model
        save_dir = get_default_save_dir("custom", f"ablation_{config_name}")
        filename = create_model_filename(f"ablation_{config_name}", seed)
        save_model_checkpoint(model, optimizer, save_dir, filename, settings, seed, 
                            iterations, eval_mean, eval_std)
        
        # Close WandB run
        if agent.logger:
            agent.logger.close()
        
    finally:
        env.close()

def main():
    """Run all ablation experiments."""
    
    print("\n" + "="*80)
    print("ABLATION STUDY: Model Architecture Comparison")
    print(f"Configs: {list(CONFIGS.keys())}")
    print("="*80)
    
    env_path = 'environment_builds/stage2/S2_Find_2Items_64x36camera120deg_rew0_20_100/Warehouse_Bot.exe'
    
    # Train all configs
    start = time.time()
    for name, config in CONFIGS.items():
        train_config(name, config, env_path, iterations=300, seed=0)
    
    total_time = (time.time() - start) / 60
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED in {total_time:.1f} minutes")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
