"""
Test minimal model functionality.
"""

import unittest
import torch
from src.models import SimpleModel

class TestMinimal:


    """Class for TestMinimal.""""""
Test minimal model functionality.
"""

    def setUp(self):


        """Method for setUp."""
    self.model = SimpleModel()
    self.vocab_size = 1000

    def test_forward_pass(self):


        """Method for test_forward_pass."""
    input_tensor = torch.randint(0, self.vocab_size, (1, 32))
    output = self.model(input_tensor)
    self.assertEqual(output.shape[0], 1)

    def test_batch_processing(self):


        """Method for test_batch_processing."""
    batch_size = 16
    input_tensor = torch.randint(0, self.vocab_size, (batch_size, 32))
    output = self.model(input_tensor)
    self.assertEqual(output.shape[0], batch_size)
