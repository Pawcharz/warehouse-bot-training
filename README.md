# Warehouse Bot Training

Deep reinforcement learning system for autonomous object manipulation in 3D simulated environments.

## Overview

This repository contains the training implementation for a goal-conditioned RL agent that finds specific items and delivers them to designated locations in Unity simulations. The agent uses multimodal inputs (visual + task encodings) and achieves **96% success rate** on the delivery task and **100% on item finding** using a custom PyTorch implementation of Proximal Policy Optimization (PPO).

**Bachelor's Thesis Project** by Paweł Blicharz (2025)  
*Supervisor: Prof. dr hab. inż. Jacek Rumiński*  
*Data Engineering, Gdańsk University of Technology*

## Key Features

- **Custom PPO Implementation** - PyTorch-based with stability improvements (value clipping, reward normalization, GAE)
- **Multimodal Architecture** - Combines 36×64 camera input with task-specific encodings using feature-level fusion
- **Curriculum Learning** - 2-stage training: item finding -> full delivery task
- **Sparse Reward Handling** - Successfully learns with delayed feedback through GAE and exploration bonuses
- **Validated Implementation** - Tested against Stable-Baselines3 on Gymnasium benchmarks (CartPole, Acrobot)

## Quick Start

### Training
```bash
# Stage 1: Find items only
python run_training.py  # Uses custom_ppo_camera.py

# Stage 2: Transfer to delivery task
# (Update run_training.py to use custom_ppo_delivery_from_pretrained.py)
```

### Evaluation
```bash
python run_evaluation.py
```

### Ablation Studies
```bash
python run_ablation_study.py
```

## Results

| Task | Success Rate | Training Steps |
|------|--------------|----------------|
| Find Item (Stage 1) | 100% | 471K |
| Find + Deliver (Stage 2) | 96% | 679K |
| **Total** | **96%** | **1.15M** |

## Architecture

```
Visual Input (36×64×3) -> CNN (4 blocks) -> MLP -> [64 features]
                                                      ↓ Concatenate
Task Vector (2 items)  -> Embedding -> MLP  -> [64 features]
                                                      ↓
                                              [128 fused features]
                                                   ↙    ↘
                                           Policy Network  Value Network
                                           (3 actions)     (return pred.)
```

## Repository Structure

```
warehouse-bot-training/
├── src/
│   ├── algorithms/         # PPO implementation, reward normalization
│   ├── models/            # Neural network architectures
│   ├── environments/      # Unity-Gymnasium wrappers
│   ├── trainings/         # Training scripts (2-stage curriculum)
│   ├── evaluation/        # Policy evaluation utilities
│   └── utils/            # Logging, seeding, early stopping
├── experiments/
│   ├── ablation_study.py           # Architecture ablations
│   └── sb3_custom_comparison/      # Validation vs Stable-Baselines3
├── environment_builds/             # Unity executables
└── saved_models/                   # Trained checkpoints
```

## Key Components

- **PPO Algorithm** (`src/algorithms/PPO_algorithm.py`) - Custom implementation with GAE, value clipping, adaptive reward normalization
- **Multimodal Model** (`src/models/actor_critic_multimodal_embedding.py`) - CNN visual encoder + task embedding encoder
- **Training Scripts** - Curriculum learning: `custom_ppo_camera.py` -> `custom_ppo_delivery_from_pretrained.py`

## Technologies

- **Unity** + **ML-Agents Toolkit** - 3D environment simulation
- **PyTorch** - Deep learning framework
- **Gymnasium** - RL environment interface
- **WandB** - Experiment tracking and visualization

## Related

Unity Environment: [github.com/Pawcharz/warehouse-bot](https://github.com/Pawcharz/warehouse-bot)
