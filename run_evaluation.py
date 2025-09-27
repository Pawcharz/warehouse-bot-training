#!/usr/bin/env python3
"""
Simple runner for warehouse bot inference.
"""

import os
import sys

# Add root directory to path to find config module
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from config import ROOT_DIR

# Change to root directory and run the inference script
os.chdir(ROOT_DIR)
sys.path.insert(0, '.')

# Import and run the inference script
from src.evaluation.evaluate_model import main

if __name__ == "__main__":
    main() 