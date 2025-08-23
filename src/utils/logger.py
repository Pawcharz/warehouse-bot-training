import time
import torch as th
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict


class PPOLogger:
    """
    Centralized logger for PPO training with TensorBoard integration and console output.
    Handles weight distributions, gradients, parameter changes, action distributions, and training metrics.
    """
    
    def __init__(self, settings, seed=0):
        """
        Initialize the logger with TensorBoard support if enabled.
        
        Args:
            settings: Dictionary containing logger configuration
            seed: Random seed for reproducible logging
        """
        self.settings = settings
        self.seed = seed
        self.logger = None
        self.histogram_logging_interval = settings.get('histogram_logging_interval', 10)
        
        # Initialize TensorBoard logger if enabled
        if settings.get('use_tensorboard', False):
            log_dir = settings.get('tensorboard_log_dir', 'runs/ppo_training')
            experiment_name = settings.get('experiment_name', f'ppo_seed_{seed}')
            full_log_dir = os.path.join(log_dir, experiment_name)
            self.logger = SummaryWriter(full_log_dir)
    
    def log_hyperparameters(self, hyperparams):
        """
        Log hyperparameters to TensorBoard and console.
        
        Args:
            hyperparams: Dictionary of hyperparameters to log
        """
        if not hyperparams:
            return
            
        # Filter and convert values to TensorBoard-compatible types
        clean_hparams = {}
        for key, value in hyperparams.items():
            if value is None:
                continue
                
            # Convert to TensorBoard-compatible types
            if isinstance(value, (int, float, str, bool)):
                clean_hparams[key] = value
            elif isinstance(value, th.Tensor):
                # Convert tensor to scalar if possible
                if value.numel() == 1:
                    clean_hparams[key] = value.item()
                else:
                    # Skip complex tensors
                    continue
            elif hasattr(value, '__str__'):
                # Convert other objects to strings (like device, etc.)
                clean_hparams[key] = str(value)
            else:
                # Skip unsupported types
                continue
        
        if not clean_hparams:
            return
        
        # Log hyperparameters to TensorBoard
        if self.logger is not None:
            try:
                self.logger.add_hparams(clean_hparams, {})
            except Exception as e:
                print(f"Warning: Could not log hyperparameters to TensorBoard: {e}")
        
        # Also log to console for visibility
        print("Hyperparameters:")
        for key, value in clean_hparams.items():
            print(f"  {key}: {value}")
    
    def log_runtime_hyperparameters(self, iteration, **hparams):
        """
        Log hyperparameters that change during training (e.g., learning rates).
        
        Args:
            iteration: Current training iteration
            **hparams: Hyperparameters to log as keyword arguments
        """
        if self.logger is None:
            return
            
        # Log each hyperparameter as a scalar with the iteration
        for key, value in hparams.items():
            if value is not None:
                self.logger.add_scalar(f'Hyperparameters/{key}', value, iteration)
    
    def log_weight_distributions(self, model, iteration, log_histograms=False):
        """
        Log weight distributions for different parts of the model.
        
        Args:
            model: The neural network model
            iteration: Current training iteration
            log_histograms: Whether to log full histograms (expensive operation)
        """
        model_parts = {
            'visual_encoder_cnn': model.visual_encoder_cnn,
            'visual_encoder_mlp': model.visual_encoder_mlp,
            'vector_encoder': model.vector_encoder,
            'policy_net': model.policy_net,
            'value_net': model.value_net
        }
        
        print(f"  Weight distributions:")
        for part_name, part_module in model_parts.items():
            weights = []
            for param in part_module.parameters():
                if param.requires_grad:
                    weights.extend(param.data.cpu().numpy().flatten())
            
            if weights:
                weights = np.array(weights)
                weight_norm = np.linalg.norm(weights)
                weight_min = np.min(weights)
                weight_max = np.max(weights)
                weight_mean = np.mean(weights)
                
                if self.logger is not None:
                    if log_histograms:
                        self.logger.add_histogram(f'Weights/Distributions/{part_name}', weights, iteration)
                    
                    self.logger.add_scalar(f'Weights/Norm/{part_name}', weight_norm, iteration)
                    self.logger.add_scalar(f'Weights/Mean/{part_name}', weight_mean, iteration)
                
                print(f"    {part_name}: norm={weight_norm:.6f}, mean={weight_mean:.6f}, range=[{weight_min:.4f}, {weight_max:.4f}]")
    
    def log_gradient_distributions(self, model, iteration, log_histograms=False):
        """
        Log gradient distributions for different parts of the model.
        
        Args:
            model: The neural network model
            iteration: Current training iteration
            log_histograms: Whether to log full histograms (expensive operation)
        """
        model_parts = {
            'visual_encoder_cnn': model.visual_encoder_cnn,
            'visual_encoder_mlp': model.visual_encoder_mlp,
            'vector_encoder': model.vector_encoder,
            'policy_net': model.policy_net,
            'value_net': model.value_net
        }
        
        print(f"  Gradient distributions:")
        for part_name, part_module in model_parts.items():
            gradients = []
            for param in part_module.parameters():
                if param.requires_grad and param.grad is not None:
                    gradients.extend(param.grad.data.cpu().numpy().flatten())
            
            if gradients:
                gradients = np.array(gradients)
                grad_norm = np.linalg.norm(gradients)
                grad_min = np.min(gradients)
                grad_max = np.max(gradients)
                grad_mean = np.mean(gradients)
                
                if self.logger is not None:
                    if log_histograms:
                        self.logger.add_histogram(f'Gradients/Distributions/{part_name}', gradients, iteration)
                    
                    self.logger.add_scalar(f'Gradients/Norm/{part_name}', grad_norm, iteration)
                    self.logger.add_scalar(f'Gradients/Mean/{part_name}', grad_mean, iteration)
                
                print(f"    {part_name}: norm={grad_norm:.6f}, mean={grad_mean:.6f}, range=[{grad_min:.6f}, {grad_max:.6f}]")
            else:
                print(f"    {part_name}: NO GRADIENTS (not being updated!)")
    
    def capture_parameters(self, model):
        """
        Capture current parameter values for change tracking.
        
        Args:
            model: The neural network model
            
        Returns:
            Dictionary containing parameter snapshots for each model part
        """
        model_parts = {
            'visual_encoder_cnn': model.visual_encoder_cnn,
            'visual_encoder_mlp': model.visual_encoder_mlp,
            'vector_encoder': model.vector_encoder,
            'policy_net': model.policy_net,
            'value_net': model.value_net
        }
        
        old_params = {}
        for part_name, part_module in model_parts.items():
            old_params[part_name] = []
            for param in part_module.parameters():
                if param.requires_grad:
                    old_params[part_name].append(param.data.cpu().numpy().copy())
        
        return old_params
    
    def log_parameter_changes(self, model, iteration, old_params):
        """
        Log parameter changes after optimization step.
        
        Args:
            model: The neural network model
            iteration: Current training iteration
            old_params: Previously captured parameters from capture_parameters()
        """
        model_parts = {
            'visual_encoder_cnn': model.visual_encoder_cnn,
            'visual_encoder_mlp': model.visual_encoder_mlp,
            'vector_encoder': model.vector_encoder,
            'policy_net': model.policy_net,
            'value_net': model.value_net
        }
        
        print(f"  Parameter changes:")
        for part_name, part_module in model_parts.items():
            changes = []
            param_idx = 0
            for param in part_module.parameters():
                if param.requires_grad:
                    old_param = old_params[part_name][param_idx]
                    change = (param.data.cpu().numpy() - old_param).flatten()
                    changes.extend(change)
                    param_idx += 1
            
            if changes:
                changes = np.array(changes)
                change_mean = np.mean(np.abs(changes))
                change_min = np.min(changes)
                change_max = np.max(changes)
                change_std = np.std(changes)
                
                if self.logger is not None:
                    self.logger.add_scalar(f'ParameterChanges/MeanAbs/{part_name}', change_mean, iteration)
                    self.logger.add_scalar(f'ParameterChanges/Std/{part_name}', change_std, iteration)
                
                print(f"    {part_name}: avg={change_mean:.6f}, range=[{change_min:.6f}, {change_max:.6f}]")
            else:
                print(f"    {part_name}: NO CHANGES (not being updated!)")
    
    def log_action_distribution(self, actions, iteration, log_histograms=False):
        """
        Log action distribution statistics for the collected actions.
        
        Args:
            actions: Array of actions taken during rollout
            iteration: Current training iteration
            log_histograms: Whether to log full histograms (expensive operation)
        """
        action_counts = defaultdict(int)
        for action in actions:
            action_counts[action] += 1
            
        if self.logger is not None:
            action_array = np.array(actions)
            if log_histograms:
                self.logger.add_histogram(f'Actions/Distribution', action_array, iteration)
            
            # Log action frequencies
            for action, count in action_counts.items():
                self.logger.add_scalar(f'Actions/Frequencies/action_{action}', count/len(actions), iteration)
            
            # Log action entropy for diversity measure
            frequencies = np.array([count/len(actions) for count in action_counts.values()])
            action_entropy = -np.sum(frequencies * np.log(frequencies + 1e-8))
            self.logger.add_scalar(f'Actions/Entropy', action_entropy, iteration)
        
        # Print action distribution summary
        action_summary = ", ".join([f"A{action}:{count/len(actions)*100:.1f}%" for action, count in sorted(action_counts.items())])
        print(f"  Actions: {action_summary}")
    
    def log_training_metrics(self, iteration, metrics):
        """
        Log training metrics to TensorBoard.
        
        Args:
            iteration: Current training iteration
            metrics: Dictionary containing training metrics to log
        """
        if self.logger is None:
            return
        
        # Log basic training metrics
        if 'mean_return' in metrics:
            self.logger.add_scalar('Training/Returns/mean', metrics['mean_return'], iteration)
        if 'std_return' in metrics:
            self.logger.add_scalar('Training/Returns/std', metrics['std_return'], iteration)
        if 'mean_steps' in metrics:
            self.logger.add_scalar('Training/Episode_Length/mean', metrics['mean_steps'], iteration)
        if 'std_steps' in metrics:
            self.logger.add_scalar('Training/Episode_Length/std', metrics['std_steps'], iteration)
        if 'time_taken' in metrics:
            self.logger.add_scalar('Training/Time_Taken', metrics['time_taken'], iteration)
        if 'episodes_count' in metrics:
            self.logger.add_scalar('Training/Episodes_Count', metrics['episodes_count'], iteration)
        
        # Log gate coefficient if available
        if 'gate_coeff' in metrics and metrics['gate_coeff'] is not None:
            self.logger.add_scalar('Model/Gate_Coefficient', metrics['gate_coeff'], iteration)
    
    def log_losses(self, iteration, losses):
        """
        Log loss values to TensorBoard.
        
        Args:
            iteration: Current training iteration
            losses: Dictionary containing loss values
        """
        if self.logger is None:
            return
            
        if 'total_loss' in losses:
            self.logger.add_scalar('Losses/All/total', losses['total_loss'], iteration)
        if 'policy_loss' in losses:
            self.logger.add_scalar('Losses/All/policy', losses['policy_loss'], iteration)
        if 'value_loss' in losses:
            self.logger.add_scalar('Losses/All/value', losses['value_loss'], iteration)
        if 'entropy_loss' in losses:
            self.logger.add_scalar('Losses/All/entropy', losses['entropy_loss'], iteration)
        if 'gate_loss' in losses and losses['gate_loss'] > 0:
            self.logger.add_scalar('Losses/All/gate', losses['gate_loss'], iteration)
    
    def log_learning_rates(self, iteration, optimizer):
        """
        Log current learning rates to TensorBoard.
        
        Args:
            iteration: Current training iteration
            optimizer: The optimizer containing parameter groups with learning rates
        """
        if self.logger is None:
            return
            
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Assume 3 parameter groups: visual_encoder, vector_encoder, policy_value
        if len(current_lrs) >= 3:
            self.logger.add_scalar('Optimization/Learning_Rates/visual_encoder', current_lrs[0], iteration)
            self.logger.add_scalar('Optimization/Learning_Rates/vector_encoder', current_lrs[1], iteration)
            self.logger.add_scalar('Optimization/Learning_Rates/policy_value', current_lrs[2], iteration)
        else:
            # Fallback for single learning rate
            self.logger.add_scalar('Optimization/Learning_Rate', current_lrs[0], iteration)
    
    def log_console_training_summary(self, iteration, ep_returns, time_taken, mean_return, std_return, 
                                   mean_steps, std_steps, mean_losses, gate_coeff=None, learning_rates=None):
        """
        Log training summary to console.
        
        Args:
            iteration: Current training iteration
            ep_returns: List of episode returns
            time_taken: Time taken for this iteration
            mean_return: Mean episode return
            std_return: Standard deviation of episode returns
            mean_steps: Mean episode length
            std_steps: Standard deviation of episode lengths  
            mean_losses: Dictionary of mean losses
            gate_coeff: Gate coefficient (optional)
            learning_rates: List of current learning rates (optional)
        """
        # Build learning rate info string
        lr_info = ""
        if learning_rates is not None and len(learning_rates) >= 3:
            lr_info = f" | LRs: visual={learning_rates[0]:.2e}, vector={learning_rates[1]:.2e}, general={learning_rates[2]:.2e}"
        elif learning_rates is not None and len(learning_rates) == 1:
            lr_info = f" | LR: {learning_rates[0]:.2e}"
        
        # Build gate info string
        gate_info = f" | Gate Coeff: {gate_coeff:.4f}" if gate_coeff is not None else ""
        
        # Format losses string
        losses_str = f"total: {mean_losses['total_loss']:.6f}, policy: {mean_losses['policy_loss']:.6f}, value: {mean_losses['value_loss']:.6f}, entropy: {mean_losses['entropy_loss']:.6f}"
        if 'gate_loss' in mean_losses:
            losses_str += f", gate: {mean_losses['gate_loss']:.6f}"
        
        print(f"Iteration {iteration} completed. Episodes: {len(ep_returns)} | Time taken: {time_taken:.2f}s | "
              f"Mean Return: {mean_return:.4f} | Std Return: {std_return:.4f} | "
              f"Mean steps: {mean_steps:.4f} | Std steps: {std_steps:.4f}{gate_info} | "
              f"Mean losses: {losses_str}{lr_info}")
    
    def close(self):
        """Close the TensorBoard logger."""
        if self.logger is not None:
            self.logger.close() 