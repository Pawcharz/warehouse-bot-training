#!/usr/bin/env python3
"""
Evaluation utilities for trained policies
"""

import torch as th
import numpy as np
from typing import Tuple, Dict, Callable, Optional


# ============================================================================
# Outcome Categorization Strategies
# ============================================================================

class OutcomeCategorizer:
    """Base class for categorizing episode outcomes based on returns and termination type."""
    
    def categorize(self, episode_return: float, terminated: bool, truncated: bool) -> str:
        """
        Categorize an episode outcome.
        
        Args:
            episode_return: Total episode return
            terminated: Whether episode ended naturally (success/failure)
            truncated: Whether episode was truncated (timeout)
        
        Returns:
            Outcome category: 'success', 'timeout', 'wall_hit', or 'wrong_item'
        """
        raise NotImplementedError
    
    def get_all_categories(self):
        """
        Return all possible outcome categories for this categorizer.
        
        Returns:
            List of all possible outcome category strings
        """
        raise NotImplementedError


class FindItemCategorizer(OutcomeCategorizer):
    """
    Categorizer for Find Item tasks.
    Reward structure: 0 (timeout/wall), 20 (any item), 100 (correct item)
    """
    
    def categorize(self, episode_return: float, terminated: bool, truncated: bool) -> str:
        if truncated:
            return 'timeout'
        elif terminated:
            if episode_return >= 100:
                return 'success'  # Found correct item
            elif episode_return >= 15:
                return 'wrong_item'  # Found wrong item (reward ~20)
            else:
                return 'wall_hit'  # Hit wall or timeout with 0 reward
        return 'timeout'  # Fallback
    
    def get_all_categories(self):
        """Return all possible outcome categories for Find Item tasks."""
        return ['success', 'wrong_item', 'wall_hit', 'timeout']


class FindDeliverCategorizer(OutcomeCategorizer):
    """
    Categorizer for Find & Deliver tasks.
    Task: Find the correct item (out of 2) and deliver it to the deposit.
    
    Reward structure:
    - 0 points: Wall hit OR timeout (step limit exceeded)
    - 20 points: Wrong item gathered (episode ends immediately)
    - 100 points: Correct item gathered
    - 200 points: Correct item gathered + delivered to deposit (100 + 100)
    
    Episode ending scenarios:
    1. Success (200, terminated): Agent gathered correct item and delivered it to deposit
    2. Correct item not delivered (100, truncated): Agent gathered correct item but didn't deliver in time
    3. Correct item then wall hit (100, terminated): Agent gathered correct item then hit wall
    4. Wrong item gathered (20, terminated): Agent gathered wrong item - episode ends immediately
    5. Wall hit (0, terminated): Agent hit wall before gathering any item
    6. Timeout (0, truncated): Agent exceeded step limit without gathering any item
    """
    
    def categorize(self, episode_return: float, terminated: bool, truncated: bool) -> str:
        # Timeout - exceeded step limit
        if truncated:
            if episode_return >= 100:
                return 'timeout_correct_item'  # Gathered correct item but didn't deliver in time
            else:
                return 'timeout_no_item'  # Exceeded step limit without gathering any item (0 points)
        
        # Terminated - episode ended by agent action or event
        elif terminated:
            if episode_return >= 200:
                return 'correct_item_delivered'  # Correct item delivered to deposit (100 + 100)
            elif episode_return >= 100:
                return 'correct_item_wall_hit'  # Gathered correct item then hit wall
            elif episode_return >= 20:
                return 'wrong_item_gathered'  # Gathered wrong item - episode ends immediately
            else:
                return 'wall_hit_no_item'  # Hit wall before gathering any item
        
        # Fallback (shouldn't reach here)
        return 'unknown'
    
    def get_all_categories(self):
        """Return all possible outcome categories for Find & Deliver tasks."""
        return [
            'correct_item_delivered',
            'timeout_correct_item',
            'correct_item_wall_hit',
            'wrong_item_gathered',
            'wall_hit_no_item',
            'timeout_no_item',
            'unknown'
        ]


# Registry of categorizers by environment type
OUTCOME_CATEGORIZERS: Dict[str, OutcomeCategorizer] = {
    'find': FindItemCategorizer(),
    'find_deliver': FindDeliverCategorizer(),
}


def get_outcome_categorizer(env_type: str = 'find') -> OutcomeCategorizer:
    """
    Get the appropriate outcome categorizer for an environment type.
    
    Args:
        env_type: Type of environment ('find' or 'find_deliver')
    
    Returns:
        OutcomeCategorizer instance
    """
    if env_type not in OUTCOME_CATEGORIZERS:
        available = ', '.join(OUTCOME_CATEGORIZERS.keys())
        raise ValueError(f"Unknown env_type '{env_type}'. Available: {available}")
    return OUTCOME_CATEGORIZERS[env_type]


# ============================================================================
# Evaluation Functions
# ============================================================================

def prepare_observation(obs, device: th.device, obs_type: str = "vector"):
    """
    Prepare observation for model input based on observation type
    """
    if obs_type == "auto":
        if isinstance(obs, dict):
            obs_type = "multimodal"
        else:
            obs_type = "vector"
    
    if obs_type == "multimodal":
        return obs
    elif obs_type == "vector":
        if isinstance(obs, th.Tensor):
            obs_tensor = obs.to(device)
        else:
            obs_tensor = th.tensor(obs, dtype=th.float32, device=device)
            
        # FIX - is it needed?
        return obs_tensor.unsqueeze(0) if obs_tensor.dim() == 1 else obs_tensor
    else:
        raise ValueError(f"Unknown observation type: {obs_type}. Use 'vector', 'multimodal', or 'auto'.")


def evaluate_policy(model, env, device: th.device, num_episodes: int = 10, seed: int = 0, 
                   obs_type: str = "auto", verbose: bool = True, 
                   outcome_categorizer: Optional[OutcomeCategorizer] = None,
                   env_type: str = 'find'):
    """
    Evaluate a policy on the environment.
    
    Args:
        model: The policy model to evaluate
        env: The environment
        device: Torch device
        num_episodes: Number of evaluation episodes
        seed: Random seed
        obs_type: Observation type ('auto', 'multimodal', 'vector')
        verbose: Whether to print progress
        outcome_categorizer: Custom outcome categorizer (if None, uses env_type)
        env_type: Environment type for outcome categorization ('find' or 'find_deliver')
                 Only used if outcome_categorizer is None
    
    Returns:
        Tuple containing:
        - mean_return: Mean episode return
        - std_return: Std of episode returns
        - mean_steps: Mean episode steps
        - std_steps: Std of episode steps
        - returns: List of episode returns
        - steps: List of episode steps
        - outcomes: Dict with outcome counts and details
    """
    # Get outcome categorizer
    if outcome_categorizer is None:
        outcome_categorizer = get_outcome_categorizer(env_type)
    
    returns = []
    steps = []
    outcomes = {
        'details': []           # Per-episode details: (return, steps, terminated, truncated)
    }
    
    # Initialize all possible outcome categories to 0
    for category in outcome_categorizer.get_all_categories():
        outcomes[category] = 0
    
    if verbose:
        print(f"Evaluating policy for {num_episodes} episodes...")
        print(f"Using outcome categorizer: {outcome_categorizer.__class__.__name__}")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0
        episode_steps = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            model_obs = prepare_observation(obs, device, obs_type)
            
            with th.no_grad():
                action, _, _, _ = model.get_action(model_obs, deterministic=True)
            
            # Take action in the environment
            obs, reward, done, truncated, _ = env.step(action.item())
            episode_return += reward
            episode_steps += 1
        
        returns.append(episode_return)
        steps.append(episode_steps)
        
        # Categorize episode outcome using the categorizer
        outcome_category = outcome_categorizer.categorize(episode_return, done, truncated)
        
        # Track outcome category
        outcomes[outcome_category] += 1
        
        outcomes['details'].append({
            'return': episode_return,
            'steps': episode_steps,
            'terminated': done,
            'truncated': truncated,
            'outcome': outcome_category
        })
        
        if verbose:
            print(f"Episode {episode + 1}: Return = {episode_return:.2f}, Steps = {episode_steps}, Outcome = {outcome_category}")
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    mean_steps = np.mean(steps)
    std_steps = np.std(steps)
    
    if verbose:
        print(f"\nEvaluation Results:")
        print(f"Mean return: {mean_return:.2f} +- {std_return:.2f}")
        print(f"Mean steps: {mean_steps:.2f} +- {std_steps:.2f}")
        print(f"Best episode: {max(returns):.2f}")
        print(f"Worst episode: {min(returns):.2f}")
        print(f"\nOutcome Distribution:")

        for outcome_name, count in sorted(outcomes.items()):
            if outcome_name != 'details':
                print(f"  {outcome_name}: {count} ({count/num_episodes*100:.1f}%)")
    
    return mean_return, std_return, mean_steps, std_steps, returns, steps, outcomes