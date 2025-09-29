"""
Example integration of ICM with the existing PPO training loop.
This shows how to modify the training to include curiosity-driven exploration.
"""

import torch as th

def compute_intrinsic_rewards(model_with_icm, obs_buffer, next_obs_buffer, actions_buffer):
    """
    Compute intrinsic rewards for the collected experience.
    These should be added to the extrinsic rewards before computing returns.
    """
    intrinsic_rewards = []
    
    # Process in batches to avoid memory issues
    batch_size = 64
    buffer_len = len(list(obs_buffer.values())[0]) if isinstance(obs_buffer, dict) else len(obs_buffer)
    
    for start in range(0, buffer_len, batch_size):
        end = min(start + batch_size, buffer_len)
        
        if isinstance(obs_buffer, dict):
            batch_obs = {key: obs_buffer[key][start:end] for key in obs_buffer.keys()}
            batch_next_obs = {key: next_obs_buffer[key][start:end] for key in next_obs_buffer.keys()}
        else:
            batch_obs = obs_buffer[start:end]
            batch_next_obs = next_obs_buffer[start:end]
        
        batch_actions = actions_buffer[start:end]
        
        # Compute intrinsic rewards
        batch_intrinsic_rewards = model_with_icm.compute_curiosity_reward(
            batch_obs, batch_next_obs, batch_actions
        )
        
        intrinsic_rewards.append(batch_intrinsic_rewards)
    
    return th.cat(intrinsic_rewards, dim=0)
