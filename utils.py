import torch
import random
import numpy as np
import os
from torchvision.utils import save_image

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")
    return device

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_samples(imgs, path, nrow=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(imgs, path, nrow=nrow, normalize=True)
