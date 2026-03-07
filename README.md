# Warehouse Bot Training

Training system for the [Warehouse Bot](https://github.com/Pawcharz/warehouse-bot) — a reinforcement learning agent that learns to navigate a 3D warehouse environment, find items, and deliver them. Built from scratch using **PyTorch** with a custom **PPO** (Proximal Policy Optimization) implementation, designed to work with **Unity ML-Agents** environments.

## Overview

This repository contains the complete RL training pipeline for a warehouse robot that learns through:

1. **Stage 1 — Raycast observations**: The agent uses vector-based raycast sensors and a simple MLP actor-critic to learn basic navigation and item finding/delivery.
2. **Stage 2 — Camera observations**: The agent uses a 64×36 RGB camera (120° FOV) combined with task-specific item embeddings. A CNN+MLP multimodal architecture processes the visual input alongside learned task representations.
3. **Stage 2 Delivery — Transfer learning**: A pre-trained "find items" model is fine-tuned on a delivery task (find items *and* deliver to locations), leveraging previously learned visual features.
4. **Stage 3 — Complex environments**: The trained agents are evaluated (and optionally trained with ICM curiosity) in more complex warehouse layouts with obstacles and varied textures.

### Key Features

- **Custom PPO implementation** with GAE (Generalized Advantage Estimation), value clipping, and reward normalization
- **Multimodal actor-critic architecture**: CNN visual encoder + learned task embeddings with concatenation fusion
- **Intrinsic Curiosity Module (ICM)** for curiosity-driven exploration in sparse-reward environments
- **Transfer learning** support — load pre-trained models and continue training on new tasks
- **Ablation study framework** with configurable model architecture (CNN depth, embedding dimensions)
- **WandB integration** for experiment tracking (metrics, losses, gradients, weight distributions, heatmaps)
- **Early stopping** based on evaluation performance
- **Reproducibility** through comprehensive seed management

---

## Project Structure

### Core System (actively used)

```
warehouse-bot-training/
│
├── config.py                           # Root directory configuration
├── run_training.py                     # Main training entry point
├── run_evaluation.py                   # Main evaluation entry point
├── run_ablation_study.py               # Ablation study entry point
│
├── src/
│   ├── algorithms/
│   │   ├── PPO_algorithm.py            # Custom PPO agent (GAE, RolloutBuffer, training loop)
│   │   └── RewardsNormalizer.py        # Running mean/std reward normalization
│   │
│   ├── environments/
│   │   ├── env_utils.py                # Environment factory (creates Unity envs with wrappers)
│   │   ├── env_multimodal_gymnasium_wrapper.py  # Gymnasium wrapper for camera+vector obs
│   │   └── env_vector_gymnasium_wrapper.py      # Gymnasium wrapper for vector-only obs
│   │
│   ├── models/
│   │   ├── actor_critic.py                          # MLP Actor-Critic (Stage 1 / raycasts)
│   │   ├── actor_critic_multimodal_embedding.py     # CNN+Task Embedding Actor-Critic (Stage 2+)
│   │   ├── actor_critic_multimodal_configurable.py  # Configurable variant (ablation study)
│   │   ├── intrinsic_curiosity_module.py            # ICM (forward/inverse models + wrapper)
│   │   ├── model_utils.py                           # Save/load checkpoints, parameter counting
│   │   └── icm_utils.py                             # Named parameter extraction for ICM models
│   │
│   ├── trainings/
│   │   ├── custom_ppo_camera.py                     # Stage 2 training (camera, no ICM)
│   │   ├── custom_ppo_camera_icm.py                 # Stage 2/3 training (camera + ICM)
│   │   ├── custom_ppo_delivery_from_pretrained.py   # Delivery fine-tuning from pre-trained model
│   │   └── custom_ppo_raycasts.py                   # Stage 1 training (raycasts only)
│   │
│   ├── evaluation/
│   │   └── evaluate_model.py           # Standalone model evaluation script
│   │
│   └── utils/
│       ├── evaluation.py               # Shared policy evaluation function
│       ├── early_stopping.py           # Early stopping condition
│       ├── seed_utils.py               # Seed management for reproducibility
│       └── wandb_logger.py             # WandB logging (metrics, gradients, heatmaps)
│
├── experiments/
│   ├── ablation_study.py               # Architecture ablation (CNN depth, embedding dims)
│   └── sb3_custom_comparison/          # Custom PPO vs Stable-Baselines3 PPO comparison
│       ├── ppo_comparison.py           # Multi-seed comparison on CartPole/Acrobot
│       ├── ppo_test.py                 # Simple PPO sanity test
│       └── README.md                   # Experiment documentation & results
│
├── environment_builds/                 # Unity builds go here (git-ignored)
│   └── stage2/
│       └── README.txt
│
├── saved_models/
│   ├── custom/                         # Custom PPO checkpoints (.pth)
│   └── baselines/                      # SB3 baseline checkpoints (.zip) — legacy
│
└── logs/                               # TensorBoard training logs
    ├── stage1/
    └── stage2/
```

### Experimental / Legacy (not part of the main pipeline)

| File / Directory | Status | Notes |
|---|---|---|
| `src/models/actor_critic_multimodal.py` | **Superseded** | Earlier multimodal model using vector repeat+MLP instead of task embeddings. Replaced by `actor_critic_multimodal_embedding.py`. |
| `src/models/actor_critic_multimodal_embedding_actions.py` | **Experimental** | Variant that also embeds the previous action as input. Not used by any training script. Contains copy-paste artifacts. |
| `src/notebooks/ppo_training_sb3.ipynb` | **Legacy** | Early exploration using SB3's PPO with a custom feature extractor. References a now-removed wrapper (`env_camera_raycasts_gymnasium_wrapper`). |
| `src/notebooks/ppo_inference_sb3.ipynb` | **Legacy** | Paired with the above notebook for SB3 model inference. |
| `test.ipynb` | **Experimental** | Ad-hoc prototyping notebook. Uses old import paths. |
| `src/saved_models/` | **Legacy** | Single old Stage 1 checkpoint, not referenced by current scripts. |
| `saved_models/baselines/` | **Legacy** | SB3-trained model checkpoints from early experiments. |

---

## Architecture

### PPO Algorithm (`PPO_algorithm.py`)

The custom PPO implementation includes:

- **GAE** (Generalized Advantage Estimation) with configurable γ and λ
- **Clipped surrogate objective** (policy loss) as described in the original PPO paper
- **Value function clipping** to reduce critic training variability (from OpenAI baselines)
- **Reward normalization** using running mean/std of discounted returns
- **Optional advantage normalization**
- **Per-component learning rates** via parameter groups (visual encoder, task encoder, policy/value heads)
- **Learning rate scheduling** with StepLR
- **Gradient clipping** (max grad norm)
- **Optional ICM integration** — intrinsic curiosity rewards are normalized separately and added to extrinsic rewards with a configurable scale

### Model Architectures

#### `ActorCritic` (Stage 1 — Vector observations)
Simple MLP with separate actor and critic networks:
- 2-layer Tanh-activated MLP (128→128) for both actor and critic
- Input: raycast observation vector
- Output: discrete action distribution + state value

#### `ActorCriticMultimodal` with Task Embeddings (Stage 2+ — Camera observations)
Multimodal architecture with:
- **Visual encoder**: 4-block CNN (Conv2d → BatchNorm → ReLU → Pool) followed by a 3-layer MLP with dropout and LayerNorm, producing a 64-dim visual embedding
- **Task encoder**: Learned item embeddings (pick item + held item) processed through a 3-layer MLP with LayerNorm, producing a 64-dim task embedding
- **Fusion**: Concatenation of visual and task embeddings (128-dim)
- **Policy head**: 3-layer MLP (128→64→act_dim)
- **Value head**: 3-layer MLP (128→64→1)

#### `ActorCriticMultimodalConfigurable` (Ablation Study)
Same architecture as above but with configurable:
- Number of CNN blocks (3, 4, or 5)
- Task embedding dimension (16, 32, or 64)
- Fusion strategy (currently concatenation)

#### Intrinsic Curiosity Module (ICM)
Optional wrapper around any actor-critic model:
- **Feature network**: Reuses the actor-critic's shared encoding
- **Inverse model**: Predicts action from (current, next) state features — learns useful feature representations
- **Forward model**: Predicts next state features from (current features, action) — prediction error = intrinsic reward
- Configurable η (reward scaling) and β (inverse vs. forward loss weight)

### Environment Wrappers

Unity ML-Agents environments are wrapped to conform to the Gymnasium API:

- **`UnityVectorGymWrapper`**: For vector-only observations (raycasts)
- **`UnityMultimodalGymWrapper`**: For camera + vector observations, with automatic splitting of vector observations into actual observations and info (map position, forward direction for heatmap logging)

---

## Setup

### Prerequisites

- Python 3.8+
- PyTorch (with CUDA support recommended)
- Unity ML-Agents Python package (`mlagents_envs`)
- Gymnasium
- NumPy
- WandB (for experiment tracking)
- `python-dotenv` (optional, for `.env` file loading)

### Required Python packages

```
torch
gymnasium
numpy
mlagents_envs
wandb
python-dotenv
matplotlib
```

For the SB3 comparison experiment, additionally:
```
stable-baselines3
```

### Environment Setup

1. **Build Unity environments** from the companion [warehouse-bot](https://github.com/Pawcharz/warehouse-bot) repository and export them to `environment_builds/`:
   ```
   environment_builds/
   └── stage2/
       └── <build_name>/
           └── Warehouse_Bot.exe
   ```

2. **Configure WandB** (optional but recommended) by creating a `.env` file in the project root:
   ```
   WANDB_API_KEY=your_api_key
   WANDB_PROJECT=your_project_name
   WANDB_ENTITY=your_entity
   ```

---

## Usage

### Training

Edit `run_training.py` to select the desired training script, then:

```bash
python run_training.py
```

Available training scripts (configured via import in `run_training.py`):

| Script | Description |
|---|---|
| `custom_ppo_raycasts` | Stage 1 — raycast observations, basic find/deliver |
| `custom_ppo_camera` | Stage 2 — camera observations, find items task |
| `custom_ppo_camera_icm` | Stage 2/3 — camera + ICM curiosity exploration |
| `custom_ppo_delivery_from_pretrained` | Stage 2 — delivery fine-tuning from pre-trained model |

Each training script contains its own hyperparameter configuration (γ, λ, clip_eps, learning rates, buffer size, etc.) and specifies the Unity environment build path.

### Evaluation

```bash
python run_evaluation.py
```

Loads a saved model checkpoint and evaluates it on a specified environment. Configure the model path and environment build path inside `src/evaluation/evaluate_model.py`.

### Ablation Study

```bash
python run_ablation_study.py
```

Runs multiple model architecture configurations sequentially:
- **baseline**: 4 CNN blocks, 32-dim task embedding
- **visual_shallow**: 3 CNN blocks
- **visual_deep**: 5 CNN blocks
- **embedding_small**: 16-dim task embedding
- **embedding_large**: 64-dim task embedding

Results are logged to WandB under a separate `warehouse-bot-ablation` project.

### SB3 Comparison Experiment

```bash
python experiments/sb3_custom_comparison/ppo_comparison.py
```

Runs a multi-seed comparison of the custom PPO vs. Stable-Baselines3's PPO on standard Gymnasium environments (CartPole-v1, Acrobot-v1). Results are logged to WandB and TensorBoard.

---

## Key Hyperparameters

Default PPO settings used across training scripts:

| Parameter | Value | Description |
|---|---|---|
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_eps` | 0.2 | PPO clip range |
| `value_clip_eps` | 0.2 | Value function clip range |
| `epochs` | 4 | PPO update epochs per iteration |
| `batch_size` | 128 | Mini-batch size |
| `buffer_size` | 2048 | Rollout buffer size (timesteps per iteration) |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |
| `loss_val_coef` | 0.5 | Value loss coefficient |
| `loss_entr_coef` | 0.01–0.015 | Entropy bonus coefficient |
| `visual_lr` | 1e-4 | Learning rate for visual encoder |
| `task_lr` | 1e-4 | Learning rate for task encoder |
| `general_lr` | 3e-4 | Learning rate for policy/value heads |

ICM-specific (when enabled):

| Parameter | Value | Description |
|---|---|---|
| `icm_loss_weight` | 0.1 | Weight of ICM loss in total loss |
| `icm_eta` | 0.01 | Intrinsic reward scaling factor |
| `icm_beta` | 0.6 | Inverse vs. forward loss balance |
| `intrinsic_reward_scale` | 0.05 | Scale of intrinsic reward when combining with extrinsic |

---

## Saved Models

Model checkpoints are saved as `.pth` files containing:
- Model state dict
- Optimizer state dict
- Training settings (hyperparameters)
- Seed, training iterations, and final evaluation metrics

### Custom PPO checkpoints (`saved_models/custom/`)

| Model | Description |
|---|---|
| `ppo_camera_120deg_0_20_100_find_2_items_train_0_seed_0` | Stage 2 base model — find 2 items with camera |
| `ppo_camera_120deg_0_20_100_find_2_items_train_1` | Stage 2 second training run |
| `ppo_camera_120deg_0_20_100_find_2_items_task_embedding_attempt_*` | Task embedding architecture iterations |
| `ppo_camera_120deg_0_20_100_find_2_items_small_env*` | Small environment training runs |
| `ppo_camera_120deg_0_20_100_find_2_items_deliver_task_embedding_attempt_1` | Delivery task (fine-tuned from pre-trained find model) |
| `icm_module_performance_test_complex_env_02_10_2025` | ICM exploration test in complex environment |

---

## Experiment Tracking

Training metrics are logged to **Weights & Biases (WandB)**, including:

- **Training metrics**: mean/std return, mean/std steps, episode count, timesteps, time per iteration
- **Evaluation metrics**: periodic deterministic evaluation returns/steps
- **Losses**: total, policy, value, entropy (+ ICM inverse/forward losses when enabled)
- **Gradients**: per-component gradient magnitude
- **Weight distributions**: per-component weight statistics
- **Parameter changes**: per-component parameter update magnitudes
- **Learning rates**: per parameter group
- **Heatmaps**: agent visit frequency and direction distribution (logged periodically)

---

## Related Repositories

- **[warehouse-bot](https://github.com/Pawcharz/warehouse-bot)** — Unity 3D warehouse environment with ML-Agents integration
