import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_default_device(device: str = "cuda") -> torch.device:
    if torch.cuda.is_available():
        resolved = torch.device(device)
        torch.cuda.set_device(resolved)
        return resolved
    return torch.device("cpu")
