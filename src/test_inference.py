"""Test inference functionality.."""

import unittest
import torch
from src.models import SimpleModel

class TestInference:
"""Test inference functionality.."""

    def setUp(self):
    """Set up test environment.."""
    self.model = SimpleModel()

    def test_inference(self):
    """Test basic inference.."""
    input_tensor = torch.randn(1, 32)
    output = self.model(input_tensor)
    self.assertEqual(output.shape[-1], 32)

    def test_batch_inference(self):
    """Test batch inference.."""
    batch_size = 16
    input_tensor = torch.randn(batch_size, 32)
    output = self.model(input_tensor)
    self.assertEqual(output.shape[0], batch_size)
