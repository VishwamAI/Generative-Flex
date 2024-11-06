from dataclasses import dataclass, field
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import numpy as np
import os
import torch


from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

@dataclass
class ModelConfig:
    """
    Class implementing class functionality.
    """

    Module containing specific functionality."""
    A simple neural network model..
    """Module for handling specific functionality."""
    Forward pass through the model.
    
    Args:
    x: Input tensor
    
    Returns:
    Output tensor
    """
        for layer in self.layers: x = self.dropout(torch.relu(layer(x)))
        return x
