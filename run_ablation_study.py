#!/usr/bin/env python3
"""
Simple runner for ablation study experiments.
"""

import os
import sys

# Add root directory to path to find config module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from config import ROOT_DIR

# Change to root directory and run the ablation study
os.chdir(ROOT_DIR)

# Import and run the ablation study
from experiments.ablation_study import main

if __name__ == "__main__":
    main()

