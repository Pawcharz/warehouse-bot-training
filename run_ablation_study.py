#!/usr/bin/env python3
"""
Main entry point for architecture ablation study.

Runs multiple model configurations (CNN depth, embedding dimensions)
and logs results to WandB.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from config import ROOT_DIR
os.chdir(ROOT_DIR)

from experiments.ablation_study import main

if __name__ == "__main__":
    main()

