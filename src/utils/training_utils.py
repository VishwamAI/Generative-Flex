"""."""
from dataclasses import dataclass
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict
from typing import Dict
import logging
import numpy as np
import os
import torch
"""."""
Args:
params: Optional training parameters
Get optimizer for model.
Args:
model: PyTorch model
Returns:
Configured optimizer
Module for handling specific functionality.
Get learning rate scheduler.
Args:
pass
pass
optimizer: PyTorch optimizer
Returns:
Learning rate scheduler
