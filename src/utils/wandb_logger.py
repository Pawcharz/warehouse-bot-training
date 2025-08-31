import os
import wandb
import numpy as np
from collections import defaultdict

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    # Still try to load from .env manually if dotenv not installed
    print(".env loading failed")


class WandBLogger:
    """Simple WandB logger for PPO training with minimal tabular data creation."""
    
    def __init__(self, settings, seed=0):
        self.settings = settings
        self.seed = seed
        self.wandb_run = None
        
        # Initialize WandB
        api_key = settings.get('wandb_api_key', os.getenv('WANDB_API_KEY'))
        if not api_key:
            raise Exception("WANDB_API_KEY not found. Disabling WandB logging.")
        
        try:
            self.wandb_run = wandb.init(
                project=settings.get('wandb_project', 'warehouse-bot-training'),
                name=settings.get('experiment_name', f'ppo_seed_{seed}'),
                entity=settings.get('wandb_entity', None),
                tags=[f"seed_{seed}", "ppo"],
                reinit=True
            )
            print(f"WandB initialized: {self.wandb_run.name}")
        except Exception as e:
            raise Exception(f"Failed to initialize WandB: {e}")
    
    def log_hyperparameters(self, hyperparams):
        """Log initial hyperparameters once at start of training (static tabular data)."""
        if self.wandb_run is None:
            return
        
        wandb.config.update(hyperparams)
        print("Hyperparameters logged to WandB config")
    
    def capture_parameters(self, model):
        """Capture current model parameters for change tracking."""
        params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                params[name] = param.data.clone().detach()
        return params
    
    def log_parameter_changes(self, model, iteration, old_params):
        """Log aggregated parameter changes by component (time series plots)."""
        if self.wandb_run is None:
            return
        
        component_changes_abs_mean = defaultdict(list)
        component_changes_l2_norm = defaultdict(list)
        
        # Collect mean average of changes for each component
        for name, param in model.named_parameters():
            if param.requires_grad and name in old_params:
                change = param.data - old_params[name]
                component = name.split('.')[0] if '.' in name else name
                abs_mean = change.flatten().abs().mean().item()
                l2_norm = change.flatten().norm().item()
                component_changes_abs_mean[component].append(abs_mean)
                component_changes_l2_norm[component].append(l2_norm)
        
        # Log aggregated component statistics
        log_dict = {}
        for component, changes_abs_mean in component_changes_abs_mean.items():
            if changes_abs_mean:
                log_dict[f'param_changes/abs_mean/{component}'] = np.mean(changes_abs_mean)
        for component, changes_l2_norm in component_changes_l2_norm.items():
            if changes_l2_norm:
                log_dict[f'param_changes/l2_norm/{component}'] = np.mean(changes_l2_norm)
        
        if log_dict:
            wandb.log(log_dict, step=iteration)
    
    def log_gradients(self, model, iteration):
        """Log aggregated gradient statistics by component (time series plots)."""
        if self.wandb_run is None:
            return
        
        component_gradients_abs_mean = defaultdict(list)
        component_gradients_l2_norm = defaultdict(list)
        
        # Collect gradient L2 norms for each component
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                component = name.split('.')[0] if '.' in name else name
                grad_abs_mean = param.grad.flatten().abs().mean().item()
                grad_l2_norm = param.grad.flatten().norm().item()
                component_gradients_abs_mean[component].append(grad_abs_mean)
                component_gradients_l2_norm[component].append(grad_l2_norm)
        
        # Log aggregated component statistics  
        log_dict = {}
        for component, gradients_abs_mean in component_gradients_abs_mean.items():
            if gradients_abs_mean:
                log_dict[f'gradients/abs_mean/{component}'] = np.mean(gradients_abs_mean)
        for component, gradients_l2_norm in component_gradients_l2_norm.items():
            if gradients_l2_norm:
                log_dict[f'gradients/l2_norm/{component}'] = np.mean(gradients_l2_norm)
        
        if log_dict:
            wandb.log(log_dict, step=iteration)
    
    def log_weight_distributions(self, model, iteration):
        """Log aggregated weight statistics by component (time series plots)."""
        if self.wandb_run is None:
            return
        
        component_weights_abs_mean = defaultdict(list)
        
        # Collect weight L2 norms for each component
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
        """Log key training performance metrics (time series plots)."""
        if self.wandb_run is None:
            return
        
        # Log only the most important metrics to minimize columns
        log_dict = defaultdict(list)

        for key, value in metrics.items():
            if value is not None:
                log_dict[f'training/{key}'] = value
        
        wandb.log(log_dict, step=iteration)
    
    def log_losses(self, iteration, losses):
        """Log training losses (time series plots)."""
        if self.wandb_run is None:
            return
        
        # Log main losses only
        log_dict = defaultdict(list)
        for loss_component, value in losses.items():
            if value is not None:
                log_dict[f'losses/{loss_component}'] = value
        
        wandb.log(log_dict, step=iteration)
    
    def log_learning_rates(self, iteration, optimizer):
        """Log current learning rates (time series plots)."""
        if self.wandb_run is None:
            return
        
        # Log current learning rates as they change
        log_dict = {}
        for i, param_group in enumerate(optimizer.param_groups):
            group_name = param_group.get('name', f'group_{i}')
            log_dict[f'lr/{group_name}'] = param_group['lr']
        
        wandb.log(log_dict, step=iteration)
    
    def log_console_training_summary(self, iteration, ep_returns, time_taken, mean_return, std_return,
                                     mean_steps, std_steps, mean_losses, gate_coeff, current_lrs):
        """Log training summary to console."""
        print(f"\n=== Iteration {iteration} ===")
        print(f"Episodes: {len(ep_returns)}; Return: {mean_return:.2f} +- {std_return:.2f}; Steps: {mean_steps:.1f} +- {std_steps:.1f}; Time: {time_taken:.2f}s")
        print(f"Losses: {', '.join([f'{name}: {loss:.4f}' for name, loss in mean_losses.items()])}")
        if gate_coeff is not None:
            print(f"Gate Coeff: {gate_coeff:.4f}")
        print(f"Learning Rates: {[f'{lr:.2e}' for lr in current_lrs]}") 
    
    def close(self):
        """Close the WandB run."""
        if self.wandb_run is not None:
            wandb.finish()
            print("WandB run finished") 