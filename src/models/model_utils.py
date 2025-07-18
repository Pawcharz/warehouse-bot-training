#!/usr/bin/env python3
"""
Model utilities for saving and loading trained models.

This module provides reusable functions for model checkpoint management
across the warehouse bot training and inference pipeline.
"""

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
    """
    Save a model checkpoint with all necessary information for later loading.
    
    Args:
        model: The trained model to save
        optimizer: The optimizer state to save
        save_dir: Directory to save the checkpoint in
        filename: Name of the checkpoint file
        settings: Training settings dictionary
        seed: Random seed used for training
        training_iterations: Number of training iterations completed
        final_mean_return: Final evaluation mean return
        final_std_return: Final evaluation standard deviation of return
        additional_info: Optional additional information to save
        
    Returns:
        str: Path to the saved checkpoint file
        
    Raises:
        Exception: If saving fails
    """
    try:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Construct full file path
        model_path = os.path.join(save_dir, filename)
        
        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'settings': settings,
            'seed': seed,
            'training_iterations': training_iterations,
            'final_mean_return': final_mean_return,
            'final_std_return': final_std_return
        }
        
        # Add additional info if provided
        if additional_info:
            checkpoint.update(additional_info)
        
        # Save checkpoint
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
    """
    Load a model checkpoint from file.
    
    Args:
        model_path: Path to the checkpoint file
        model: Model instance to load weights into
        device: Device to load the model on
        load_optimizer: Whether to also load optimizer state
        optimizer: Optimizer instance to load state into (required if load_optimizer=True)
        
    Returns:
        Tuple[th.nn.Module, Dict[str, Any]]: Loaded model and checkpoint metadata
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        Exception: If loading fails
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    
    try:
        print(f"Loading model checkpoint from: {model_path}")
        
        # Load checkpoint
        checkpoint = th.load(model_path, map_location=device, weights_only=False)
        
        # Load model state
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
        
        # Print checkpoint information
        print("Model loaded successfully!")
        print(f"Training settings: {checkpoint.get('settings', 'Not available')}")
        print(f"Training iterations: {checkpoint.get('training_iterations', 'Not available')}")
        print(f"Final mean return: {checkpoint.get('final_mean_return', 'Not available')}")
        print(f"Final std return: {checkpoint.get('final_std_return', 'Not available')}")
        print(f"Seed: {checkpoint.get('seed', 'Not available')}")
        
        return model, checkpoint
        
    except Exception as e:
        raise Exception(f"Failed to load model checkpoint: {e}")


def get_model_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    Get information about a saved model checkpoint without loading the full model.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dict[str, Any]: Dictionary containing checkpoint metadata
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        Exception: If loading metadata fails
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")
    
    try:
        # Load only the metadata (not the model weights)
        checkpoint = th.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract metadata
        metadata = {
            'settings': checkpoint.get('settings', {}),
            'seed': checkpoint.get('seed', None),
            'training_iterations': checkpoint.get('training_iterations', None),
            'final_mean_return': checkpoint.get('final_mean_return', None),
            'final_std_return': checkpoint.get('final_std_return', None),
            'has_optimizer_state': 'optimizer_state_dict' in checkpoint,
            'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024)
        }
        
        return metadata
        
    except Exception as e:
        raise Exception(f"Failed to load checkpoint metadata: {e}")


def list_available_models(models_dir: str = "saved_models") -> Dict[str, Dict[str, Any]]:
    """
    List all available model checkpoints in the models directory.
    
    Args:
        models_dir: Directory to search for models
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping model paths to their metadata
    """
    available_models = {}
    
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return available_models
    
    # Walk through the models directory
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.pth'):
                model_path = os.path.join(root, file)
                try:
                    metadata = get_model_info(model_path)
                    available_models[model_path] = metadata
                except Exception as e:
                    print(f"Warning: Could not read metadata for {model_path}: {e}")
    
    return available_models


def create_model_filename(
    experiment_name: str,
    seed: int,
    extension: str = ".pth"
) -> str:
    """
    Create a standardized filename for model checkpoints.
    
    Args:
        experiment_name: Name of the experiment
        seed: Random seed used
        extension: File extension (default: .pth)
        
    Returns:
        str: Standardized filename
    """
    return f"{experiment_name}_seed_{seed}{extension}"


def get_default_save_dir(experiment_type: str = "custom", stage: str = "stage1") -> str:
    """
    Get the default save directory for model checkpoints.
    
    Args:
        experiment_type: Type of experiment (e.g., "custom", "baseline")
        stage: Training stage (e.g., "stage1", "stage2")
        
    Returns:
        str: Default save directory path
    """
    return os.path.join("saved_models", experiment_type, stage) 