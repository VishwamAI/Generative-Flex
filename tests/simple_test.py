from dataclasses import dataclass, field
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import numpy as np
import os
import torch
from src.models import SimpleModel
import torch
import torch.nn as nn
import unittest















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