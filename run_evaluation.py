#!/usr/bin/env python3
"""
Main entry point for evaluating a trained warehouse bot model.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from config import ROOT_DIR
os.chdir(ROOT_DIR)

from src.evaluation.evaluate_model import main

if __name__ == "__main__":
    main()