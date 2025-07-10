#!/usr/bin/env python3
"""
Simple runner for warehouse PPO training.
"""

import os
import sys
from config import ROOT_DIR, SRC_DIR

# Change to src directory and run the training script
os.chdir(SRC_DIR)
sys.path.insert(0, '.')

# Import and run the training script
from trainings.custom_ppo_raycasts import main

if __name__ == "__main__":
    main() 