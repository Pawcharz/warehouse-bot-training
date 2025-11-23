import random
import numpy as np
import torch as th


def set_all_seeds(seed: int):
  """Seed all the random libraries used acros codebase for reproducability."""
  
  random.seed(seed)
  np.random.seed(seed) # numpy random number generator
  th.manual_seed(seed) # torch cpu random generator
  th.cuda.manual_seed(seed) # torch random generator for current gpu
  th.cuda.manual_seed_all(seed) # torch random generator for all gpus (not strictly needed cause training is designed for 1 gpu)
  th.backends.cudnn.deterministic = True # defines determinism of cuDNN
  th.backends.cudnn.benchmark = False # disables automatic choice of cuDNN algorithms


def set_training_iteration_seed(base_seed: int, iteration: int):
  """Sets seed based on training or evaluation iteration"""
  
  iteration_seed = base_seed + 1000 * iteration
  set_all_seeds(iteration_seed)
  
  return iteration_seed
