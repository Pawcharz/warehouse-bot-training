"""
Early stopping for PPO training.

Usage:
    early_stop = EarlyStoppingCondition(window_size=10, metric_threshold=95.0)
    agent.train(env, iterations=300, early_stopping_fn=early_stop)
"""

import numpy as np
from typing import Dict


class EarlyStoppingCondition:
    """Stop training when average return over past K iterations >= threshold."""
    
    def __init__(self, window_size: int, metric_threshold: float):
        """
        Args:
            window_size: Number of recent iterations to average over
            metric_threshold: Stop when average metric >= this value
        """
        self.window_size = window_size
        self.metric_threshold = metric_threshold
        self.metric_history = []
    
    def __call__(self, metric_name: str, metric: float) -> bool:
        """Check if should stop training."""
        self.metric_history.append(metric)
        
        # Check condition
        if len(self.metric_history) == self.window_size:
            mean_metric = np.mean(self.metric_history)
            if mean_metric >= self.metric_threshold:
                print(f"\n✓ Early stopping: avg return over last {self.window_size} iters "
                      f"{metric_name} = {mean_metric:.2f} >= {self.metric_threshold:.2f}")
                return True
        
        if len(self.metric_history) > self.window_size:
            self.metric_history.pop(0)
        
        return False

