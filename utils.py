import random
import numpy as np
import torch

global_batch = 0
global_step = 0

def seed_everything(seed):
    """Seed all sources of randomness for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
