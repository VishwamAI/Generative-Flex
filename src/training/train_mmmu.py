from dataclasses import dataclass, field
from pathlib import Path
from src.data.mmmu_dataloader import MMMUDataLoader
from src.models.reasoning.math_head import MathHead
from src.training.trainer import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import numpy as np
import os
import torch

"""Training module documentation."""
logger = logging.getLogger(__name__)
    @dataclass
class TrainConfigTrainConfig:
    """Class implementation."""
    Module for handling specific functionality.
    Method for main..
