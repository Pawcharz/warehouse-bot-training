import os
import wandb
import numpy as np
from collections import defaultdict

# .env file loading
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print(".env loading failed")

class WandBLogger:
    """Simple WandB logger for PPO training."""
    
    def __init__(self, settings, seed=0):
        self.settings = settings
        self.seed = seed
        self.wandb_run = None
        
        # Initialize WandB
        api_key = settings.get('wandb_api_key', os.getenv('WANDB_API_KEY'))
        if not api_key:
            raise Exception("WANDB_API_KEY not found. Disabling WandB logging.")
        
        project_name = settings.get('wandb_project',  os.getenv('WANDB_PROJECT'))
        wandb_entity = settings.get('wandb_entity',  os.getenv('WANDB_ENTITY'))
        experiment_name = settings.get('experiment_name', f'ppo_seed_{seed}')

        try:
            self.wandb_run = wandb.init(
                project=project_name,
                name=experiment_name,
                entity=wandb_entity,
                tags=[f"seed_{seed}", "ppo"],
                reinit=True
            )
            print(f"WandB initialized: {self.wandb_run.name}")
        except Exception as e:
            raise Exception(f"WandB failed to initialize: {e}")
    
    def log_hyperparameters(self, hyperparams):
        """Log initial hyperparameters once at start of training."""
        
        wandb.config.update(hyperparams)
        print("Hyperparameters logged")
    
    def capture_parameters(self, model):
        """Capture current model parameters for change tracking."""
        params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                params[name] = param.data.clone().detach()
        return params
    
    def log_parameter_changes(self, model, iteration, old_params):
        """Log parameter changes for each component."""
        
        component_changes_abs_mean = defaultdict(list)
        
        # Collect mean average of changes for each component
        for name, param in model.named_parameters():
            if param.requires_grad and name in old_params:
                change = param.data - old_params[name]

                # Format: component.layer.weight - FIX - verify
                component = name.split('.')[0] if '.' in name else name
                abs_mean = change.flatten().abs().mean().item()
                component_changes_abs_mean[component].append(abs_mean)
        
        # Log aggregated component statistics
        log_dict = {}
        for component, changes_abs_mean in component_changes_abs_mean.items():
            if changes_abs_mean:
                log_dict[f'param_changes/abs_mean/{component}'] = np.mean(changes_abs_mean)
        
        if log_dict:
            wandb.log(log_dict, step=iteration)
    
    def log_gradients(self, model, iteration):
        """Log gradient statistics by component."""
        
        component_gradients_abs_mean = defaultdict(list)
        
        # Collect gradient abs mean for each component
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                component = name.split('.')[0] if '.' in name else name
                grad_abs_mean = param.grad.flatten().abs().mean().item()
                component_gradients_abs_mean[component].append(grad_abs_mean)
        
        # Log aggregated component statistics
        log_dict = {}
        for component, gradients_abs_mean in component_gradients_abs_mean.items():
            if gradients_abs_mean:
                log_dict[f'gradients/abs_mean/{component}'] = np.mean(gradients_abs_mean)
        
        if log_dict:
            wandb.log(log_dict, step=iteration)
    
    def log_weight_distributions(self, model, iteration):
        """Log weight statistics by component."""
        
        component_weights_abs_mean = defaultdict(list)
        
        # Collect weight abs mean for each component
        for name, param in model.named_parameters():
            if param.requires_grad:
                component = name.split('.')[0] if '.' in name else name
                weight_abs_mean = param.data.flatten().abs().mean().item()
                component_weights_abs_mean[component].append(weight_abs_mean)
        
        # Log aggregated component statistics
        log_dict = {}
        for component, weights_abs_mean in component_weights_abs_mean.items():
            if weights_abs_mean:
                log_dict[f'weights/abs_mean/{component}'] = np.mean(weights_abs_mean)
        
        if log_dict:
            wandb.log(log_dict, step=iteration)
    
    def log_training_metrics(self, iteration, metrics):
        """Log key training performance metrics like mean and std of returns etc."""
        
        # Log only the most important metrics to minimize columns
        log_dict = defaultdict(list)

        for key, value in metrics.items():
            if value is not None:
                log_dict[f'training/{key}'] = value
        
        wandb.log(log_dict, step=iteration)
    
    def log_losses(self, iteration, losses):
        """Log training loss components."""
        
        # Log main losses only
        log_dict = defaultdict(list)
        for loss_component, value in losses.items():
            if value is not None:
                log_dict[f'losses/{loss_component}'] = value
        
        wandb.log(log_dict, step=iteration)
    
    def log_learning_rates(self, iteration, optimizer):
        """Log current learning rates for each group of parameters."""
        
        # Log current learning rates as they change
        log_dict = {}
        for i, param_group in enumerate(optimizer.param_groups):
            group_name = param_group.get('name', f'group_{i}')
            log_dict[f'lr/{group_name}'] = param_group['lr']
        
        wandb.log(log_dict, step=iteration)
    
    def log_console_training_summary(self, iteration, ep_returns, time_taken, mean_return, std_return, mean_steps, std_steps, mean_losses, current_lrs):
        """Log training summary to console."""
        print(f"\n=== Iteration {iteration} ===")
        print(f"Episodes: {len(ep_returns)}; Return: {mean_return:.2f} +- {std_return:.2f}; Steps: {mean_steps:.1f} +- {std_steps:.1f}; Time: {time_taken:.2f}s")
        print(f"Losses: {', '.join([f'{name}: {loss:.4f}' for name, loss in mean_losses.items()])}")
        print(f"Learning Rates: {[f'{lr:.2e}' for lr in current_lrs]}") 
    
    def close(self):
        """Close the WandB run."""
        if self.wandb_run is not None:
            wandb.finish()
            print("WandB run finished") 