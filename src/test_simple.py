"""
Test simple model functionality.
"""

import unittest
import torch
from src.models import SimpleModel

class TestSimple:
"""
Test simple model functionality.
"""

    def setUp(self):
    """
Set up test environment.
"""
    self.model = SimpleModel()
    self.vocab_size = 1000

    def test_model_output(self):
    """
Test model output dimensions.
"""
    input_tensor = torch.randint(0, self.vocab_size, (1, 32))
    output = self.model(input_tensor)
    self.assertEqual(output.shape[-1], 32)

    def test_model_batch(self):
    """
Test model batch processing.
"""
    batch_size = 16
    input_tensor = torch.randint(0, self.vocab_size, (batch_size, 32))
    output = self.model(input_tensor)
    self.assertEqual(output.shape[0], batch_size)