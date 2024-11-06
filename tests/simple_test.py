"""Test simple model functionality."""

import unittest
import torch
import torch.nn as nn
from src.models import SimpleModel

class TestSimpleModel(nn.Module):
    """Test simple model functionality."""

    def setUp(self):
        """Set up test environment."""
        self.model = SimpleModel()

    def test_forward_pass(self):
        """Test forward pass."""
        input_tensor = torch.randn(1, 32)
        output = self.model(input_tensor)
        self.assertEqual(output.shape[-1], 32)

    def test_batch_processing(self):
        """Test batch processing."""
        batch_size = 16
        input_tensor = torch.randn(batch_size, 32)
        output = self.model(input_tensor)
        self.assertEqual(output.shape[0], batch_size)
