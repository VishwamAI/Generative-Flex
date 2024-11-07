"""."""

from dataclasses import dataclass
from pathlib import Path
from src.models import SimpleModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict
import logging
import numpy as np
import os
import torch
import torch
import torch.nn as nn
import unittest
"""
batch_size = 16
input_tensor = torch.randn(batch_size, 32)
output = self.model(input_tensor)
self.assertEqual(output.shape[0], batch_size)