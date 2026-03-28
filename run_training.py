#!/usr/bin/env python3
"""
Main entry point for warehouse PPO training.

Select the desired training script by uncommenting the appropriate import below.
"""

# from src.trainings.custom_ppo_camera import main                        # Stage 1: Room_Find (camera)
from src.trainings.custom_ppo_delivery_from_pretrained import main        # Stage 2: Room_Find_Deliver (curriculum)
# from src.trainings.custom_ppo_camera_icm import main                   # Camera + ICM curiosity

if __name__ == "__main__":
    main()
