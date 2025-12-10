import os
import torch as th
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

def save_model_checkpoint(
    model: th.nn.Module,
    optimizer: th.optim.Optimizer,
    save_dir: str,
    filename: str,
    settings: Dict[str, Any],
    seed: int,
    training_iterations: int,
    final_mean_return: float,
    final_std_return: float,
    additional_info: Optional[Dict[str, Any]] = None
) -> str:
    try:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, filename)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'settings': settings,
            'seed': seed,
            'training_iterations': training_iterations,
            'final_mean_return': final_mean_return,
            'final_std_return': final_std_return
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        th.save(checkpoint, model_path)
        
        print(f"Model checkpoint saved to: {model_path}")
        return model_path
        
    except Exception as e:
        raise Exception(f"Failed to save model checkpoint: {e}")

def load_model_checkpoint(
    model_path: str,
    model: th.nn.Module,
    device: th.device,
    load_optimizer: bool = False,
    optimizer: Optional[th.optim.Optimizer] = None
) -> Tuple[th.nn.Module, Dict[str, Any]]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    
    try:
        checkpoint = th.load(model_path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Load optimizer state if requested
        if load_optimizer and optimizer is not None:
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded successfully")
            else:
                print("Warning: No optimizer state found in checkpoint")
        
        print(f"Model loaded from {model_path}")
        print(f"Iterations: {checkpoint.get('training_iterations', 'N/A')}, "
              f"Return: {checkpoint.get('final_mean_return', 'N/A')}, "
              f"Seed: {checkpoint.get('seed', 'N/A')}")
        
        return model, checkpoint
        
    except Exception as e:
        raise Exception(f"Failed to load model checkpoint: {e}")

def create_model_filename(
    experiment_name: str,
    seed: int,
) -> str:
    return f"{experiment_name}_seed_{seed}.pth"

def get_default_save_dir(experiment_type: str = "custom", stage: str = "stage1") -> str:
    return os.path.join("saved_models", experiment_type, stage)

def count_parameters(model):
    total_params = 0
    block_params = {}
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        block_params[name] = params
        total_params += params
        
    block_params['total'] = total_params
    return block_params 