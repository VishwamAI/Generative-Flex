"""
Test simple model functionality.
"""

import torch

from src.models import SimpleModel
import unittest


class TestSimple:


    """
Class for TestSimple..
""""""
Test simple model functionality.
"""

    def setUp(self):


        """
Method for setUp..
"""
    self.model = SimpleModel()
    self.vocab_size = 1000

    def test_model_output(self):


        """
Method for test_model_output..
"""
    input_tensor = torch.randint(0, self.vocab_size, (1, 32))
    output = self.model(input_tensor)
    self.assertEqual(output.shape[-1], 32)

    def test_model_batch(self):


        """
Method for test_model_batch..
"""
    batch_size = 16
    input_tensor = torch.randint(0, self.vocab_size, (batch_size, 32))
    output = self.model(input_tensor)
    self.assertEqual(output.shape[0], batch_size)
