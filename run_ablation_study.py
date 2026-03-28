#!/usr/bin/env python3
"""
Main entry point for architecture ablation study.

Runs multiple model configurations (CNN depth, embedding dimensions)
and logs results to WandB.
"""

from experiments.ablation_study import main

if __name__ == "__main__":
    main()
