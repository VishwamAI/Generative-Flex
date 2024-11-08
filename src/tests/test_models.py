"""Test model functionality."""
from dataclasses import dataclass, field
from pathlib import Path
from src.models import BaseModel, EnhancedTransformer, MultiModalTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import numpy as np
import os
import torch
import unittest


class TestModels(unittest.TestCase):
    def setUp(self):
        self.base_model = BaseModel()
        self.enhanced_model = EnhancedTransformer()
        self.multimodal_model = MultiModalTransformer()
        self.test_input = torch.randn(1, 512)
        self.image_input = torch.randn(1, 3, 224, 224)

    def test_base_model_forward(self):
        output = self.base_model(self.test_input)
        self.assertIsNotNone(output)

    def test_enhanced_model_forward(self):
        output = self.enhanced_model(self.test_input)
        self.assertIsNotNone(output)

    def test_multimodal_model_forward(self):
        output = self.multimodal_model(self.test_input, self.image_input)
        self.assertIsNotNone(output)

    def test_batch_processing(self):
        batch_input = torch.randn(4, 512)
        batch_image = torch.randn(4, 3, 224, 224)
        base_output = self.base_model(batch_input)
        enhanced_output = self.enhanced_model(batch_input)
        multimodal_output = self.multimodal_model(batch_input, batch_image)
        self.assertIsNotNone(base_output)
        self.assertIsNotNone(enhanced_output)
        self.assertIsNotNone(multimodal_output)
