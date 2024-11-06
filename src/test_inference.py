"""
Test inference functionality.
"""

import unittest
import torch
from src.models import SimpleModel

class TestInference:


    """Class for TestInference.""""""
Test inference functionality.
"""

    def setUp(self):


        """Method for setUp."""
    self.model = SimpleModel()

    def test_inference(self):


        """Method for test_inference."""
    input_tensor = torch.randn(1, 32)
    output = self.model(input_tensor)
    self.assertEqual(output.shape[-1], 32)

    def test_batch_inference(self):


        """Method for test_batch_inference."""
    batch_size = 16
    input_tensor = torch.randn(batch_size, 32)
    output = self.model(input_tensor)
    self.assertEqual(output.shape[0], batch_size)
