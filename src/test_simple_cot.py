"""Test simple chain-of-thought model functionality."""
from dataclasses import dataclass, field
from pathlib import Path
from src.models import SimpleCoTModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import numpy as np
import os
import torch
import unittest


class TestSimpleCot(unittest.TestCase):
    def setUp(self):
        self.model = SimpleCoTModel()
        self.test_input = torch.randn(1, 512)

    def test_forward(self):
        output = self.model(self.test_input)
        self.assertIsNotNone(output)

    def test_batch_processing(self):
        batch_input = torch.randn(4, 512)
        output = self.model(batch_input)
        self.assertIsNotNone(output)

    def test_cot_generation(self):
        input_text = "What is 2 + 2?"
        output = self.model.generate_cot(input_text)
        self.assertIsNotNone(output)
