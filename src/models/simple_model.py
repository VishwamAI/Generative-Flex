from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm
import os
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
@dataclass class:
    """
Class implementing class functionality.
"""

Module containing specific functionality."""
A simple neural network model..
"""Module containing specific functionality."""
Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor
"""
        for layer in self.layers: x = self.dropout(torch.relu(layer(x)))
        return x
