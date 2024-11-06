from dataclasses import dataclass, field
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import numpy as np
import os
import torch



"""
Module containing specific functionality.
"""

from src.models import SimpleModel
import torch
import unittest




class TestSimple:
    """
    Class implementing TestSimple functionality.
    """

    Module containing specific functionality.
    Test simple model functionality.
    
    Method for setUp..
    
    Method for test_model_output..
    Module for handling specific functionality.
    Method for test_model_batch..
    """
    batch_size = 16
    input_tensor = torch.randint(0, self.vocab_size, (batch_size, 32))
    output = self.model(input_tensor)
    self.assertEqual(output.shape[0], batch_size)
