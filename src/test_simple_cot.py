"""Test simple chain-of-thought model.."""

import unittest
import torch
from src.models import SimpleModel

class TestSimpleCot:
"""Test simple chain-of-thought model.."""

    def setUp(self):
    """Set up test environment.."""
    self.model = SimpleModel()

    def test_cot_generation(self):
    """Test chain-of-thought generation.."""
    input_text = "What is 2+2?"
    input_tensor = torch.randint(0, 1000, (1, 32))
    output = self.model(input_tensor)
    self.assertEqual(output.shape[-1], 32)

    def test_cot_batch(self):
    """Test batch chain-of-thought generation.."""
    batch_size = 16
    input_tensor = torch.randint(0, 1000, (batch_size, 32))
    output = self.model(input_tensor)
    self.assertEqual(output.shape[0], batch_size)
