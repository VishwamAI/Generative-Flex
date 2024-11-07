"""."""
from dataclasses import dataclass
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict
from typing import Optional
import logging
import numpy as np
import os
import torch
Initialize device manager.
Args:
    config: Optional device configuration
    Set up compute device.
    Returns:
        Configured device
        Module for handling specific functionality.
        Place tensor on configured device.
        Args:
            tensor: Input tensor
            Returns:
                Tensor on configured device