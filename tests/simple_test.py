from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import os
import unittest

from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import torch
import torch.nn as nn

from src.models import SimpleModel
















"""
    \1
"""

class TestSimpleModel:
    """
    \1
    """
    batch_size = 16
    input_tensor = torch.randn(batch_size, 32)
    output = self.model(input_tensor)
    self.assertEqual(output.shape[0], batch_size)
    """