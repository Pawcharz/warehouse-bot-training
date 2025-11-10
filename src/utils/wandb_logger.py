import os
import wandb
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')    # non-interactive backend - removes thinter issues
import matplotlib.pyplot as plt


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
        
        project_name = settings.get('wandb_project',    os.getenv('WANDB_PROJECT'))
        wandb_entity = settings.get('wandb_entity',    os.getenv('WANDB_ENTITY'))
        experiment_name = settings.get('experiment_name', None)

        if experiment_name is not None and project_name is not None and wandb_entity is not None:
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
        
        if self.wandb_run is not None:
            self.wandb_run.config.update(hyperparams)
            print("Hyperparameters logged")
    
    def _extract_component_name(self, param_name):
        """
        Extracts component name from named parameters. Assumes that layers are separated by a '.' in the name
        and components which names should be logged are separated by a '/'.
        """
        parts = param_name.split('.')
        return parts[0]

    def capture_parameters(self, named_model_params):
        """Capture current model parameters for change tracking."""

        if self.wandb_run is not None:
            params = {}
            for name, param in named_model_params:
                if param.requires_grad:
                    params[name] = param.data.clone().detach()
            return params
    
    def log_parameter_changes(self, named_model_params, iteration, old_params):
        """Log parameter changes for each component."""

        if self.wandb_run is not None:
            component_changes_abs_mean = defaultdict(list)
            
            # Collect mean average of changes for each component
            for name, param in named_model_params:
                if param.requires_grad and name in old_params:
                    change = param.data - old_params[name]

                    component = self._extract_component_name(name)
                    abs_mean = change.flatten().abs().mean().item()
                    component_changes_abs_mean[component].append(abs_mean)
        
            # Log aggregated component statistics
            log_dict = {f'param_changes/{comp}': np.mean(abs_change) for comp, abs_change in component_changes_abs_mean.items()}
            self.wandb_run.log(log_dict, step=iteration)
    
    def log_gradients(self, named_model_params, iteration):
        """Log gradient statistics by component."""
        
        if self.wandb_run is not None:
            component_gradients_abs_mean = defaultdict(list)
            
            # Collect gradient abs mean for each component
            for name, param in named_model_params:
                if param.requires_grad and param.grad is not None:
                    component = self._extract_component_name(name)
                    grad_abs_mean = param.grad.flatten().abs().mean().item()
                    component_gradients_abs_mean[component].append(grad_abs_mean)
            
            # Log aggregated component statistics
            log_dict = {}
            for component, gradients_abs_mean in component_gradients_abs_mean.items():
                if gradients_abs_mean:
                    log_dict[f'gradients/abs_mean/{component}'] = np.mean(gradients_abs_mean)
            
            if log_dict:
                self.wandb_run.log(log_dict, step=iteration)

    def log_weight_distributions(self, named_model_params, iteration):
        """Log weight statistics by component."""
        
        if self.wandb_run is not None:
            component_weights_abs_mean = defaultdict(list)
            
            # Collect weight abs mean for each component
            for name, param in named_model_params:
                if param.requires_grad:
                    component = self._extract_component_name(name)
                    weight_abs_mean = param.flatten().abs().mean().item()
                    component_weights_abs_mean[component].append(weight_abs_mean)
            
            # Log aggregated component statistics
            log_dict = {}
            for component, weights_abs_mean in component_weights_abs_mean.items():
                if weights_abs_mean:
                    log_dict[f'weights/abs_mean/{component}'] = np.mean(weights_abs_mean)
            
            if log_dict:
                self.wandb_run.log(log_dict, step=iteration)
    
    def log_training_metrics(self, iteration, metrics):
        """Log key training performance metrics like mean and std of returns etc."""
        
        if self.wandb_run is not None:
            # Log only the most important metrics to minimize columns
            log_dict = defaultdict(list)

            for key, value in metrics.items():
                if value is not None:
                    log_dict[f'training/{key}'] = value
        
            self.wandb_run.log(log_dict, step=iteration)
    
    def log_evaluation_metrics(self, iteration, metrics):
        """Log evaluation metrics from deterministic policy runs."""
        
        if self.wandb_run is not None:
            log_dict = defaultdict(list)
            
            for key, value in metrics.items():
                if value is not None:
                    log_dict[f'eval/{key}'] = value
            
            self.wandb_run.log(log_dict, step=iteration)
    
    def log_event(self, iteration, event_name):
        """Log a training event (e.g., early stopping)."""
        
        if self.wandb_run is not None:
            self.wandb_run.log({f'events/{event_name}': 1}, step=iteration)

    def log_losses(self, iteration, mean_losses):
        """Log training loss components."""
        
        if self.wandb_run is not None:
            # Log main losses only
            log_dict = defaultdict(list)
            for loss_component, value in mean_losses.items():
                if value is not None:
                    log_dict[f'losses/{loss_component}'] = value
            
            self.wandb_run.log(log_dict, step=iteration)

    def log_learning_rates(self, iteration, optimizer):
        """Log current learning rates for each group of parameters."""
        
        if self.wandb_run is not None:
            # Log current learning rates as they change
            log_dict = {}
            for i, param_group in enumerate(optimizer.param_groups):
                group_name = param_group.get('name', f'group_{i}')
                log_dict[f'lr/{group_name}'] = param_group['lr']
            
            self.wandb_run.log(log_dict, step=iteration)

    def log_console_training_summary(self, iteration, ep_returns: np.ndarray, time_taken, steps: np.ndarray, losses: dict, current_lrs, intrinsic_returns: np.ndarray = None):
        """Log training summary to console."""
        
        mean_losses = {key: np.mean(losses[key]) for key in losses}
        
        print(f"\n=== Iteration {iteration} ===")
        print(f"Episodes: {len(ep_returns)}; Return: {ep_returns.mean():.2f} +- {ep_returns.std():.2f}; Steps: {steps.mean():.1f} +- {steps.std():.1f}; Time: {time_taken:.2f}s")
        if intrinsic_returns is not None:
            print(f"Intrinsic Returns: {intrinsic_returns.mean():.2f} +- {intrinsic_returns.std():.2f}")
        print(f"Losses: {', '.join([f'{name}: {loss:.4f}' for name, loss in mean_losses.items()])}")
        print(f"Learning Rates: {[f'{lr:.2e}' for lr in current_lrs]}")
    
    def log_heatmap_data(self, iteration, heatmap_data: np.ndarray, name: str, title: str, x_label: str, y_label: str, bounds: tuple = (-5, 5), buckets: int = 10):
        """Log heatmap data.
        
        Args:
            iteration: The iteration number.
            heatmap_data: The heatmap data to log. Shape: (timesteps, features), features: [x, y].
            grid_size: The grid size. Shape: (x, y).
        """
        
        if self.wandb_run is not None:
            
            coords_range = bounds[1] - bounds[0]

            # --- Compute bounds and bucket indices ---
            x_coords, y_coords = heatmap_data[:, 0], heatmap_data[:, 1]
            
            # Round and clip coordinates
            x_coords = np.clip(x_coords, bounds[0], bounds[1])
            y_coords = np.clip(y_coords, bounds[0], bounds[1])
                
            heatmap = np.zeros((buckets, buckets), dtype=np.float32)

            # Positions transformation
            
            x_idx = np.clip(np.floor((x_coords - bounds[0]) * buckets / coords_range), 0, buckets-1).astype(int)
            y_idx = np.clip(np.floor((y_coords - bounds[0]) * buckets / coords_range), 0, buckets-1).astype(int)

            heatmap = np.zeros((buckets, buckets), dtype=np.float32)
            for xi, yi in zip(x_idx, y_idx):
                heatmap[yi, xi] += 1 # ax.imshow assumes [row, col], therefore [row, col] = [y, x]

            fig, ax = plt.subplots()
            cax = ax.imshow(
                heatmap, 
                cmap='hot', 
                origin='lower', 
                interpolation='nearest',
                extent=[bounds[0], bounds[1], bounds[0], bounds[1]]
            )
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            fig.colorbar(cax, ax=ax, label='Visit count')

            self.wandb_run.log({f"heatmaps/{name}/iteration_{iteration}": wandb.Image(fig)}, step=iteration)
            plt.close(fig)

            print(f"Heatmap data logged for iteration {iteration}")

    def close(self):
        """Close the WandB run."""
        if self.wandb_run is not None:
            self.wandb_run.finish()
            print("WandB run finished") 