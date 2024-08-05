import numpy as np
import torch
import random


def seed_everything(seed: int = None):
    print(f"Global seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("set torch.backends.cudnn.benchmark=False")
    torch.backends.cudnn.benchmark = False
    print("set torch.backends.cudnn.deterministic=True")
    torch.backends.cudnn.deterministic = True
