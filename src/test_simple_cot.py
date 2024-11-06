"""
Test simple chain-of-thought model.
"""

import torch

from src.models import SimpleModel
import unittest


class TestSimpleCot:


    """
Class for TestSimpleCot..
""""""
Test simple chain-of-thought model.
"""

    def setUp(self):


        """
Method for setUp..
"""
    self.model = SimpleModel()

    def test_cot_generation(self):


        """
Method for test_cot_generation..
"""
    input_text = "What is 2+2?"
    input_tensor = torch.randint(0, 1000, (1, 32))
    output = self.model(input_tensor)
    self.assertEqual(output.shape[-1], 32)

    def test_cot_batch(self):


        """
Method for test_cot_batch..
"""
    batch_size = 16
    input_tensor = torch.randint(0, 1000, (batch_size, 32))
    output = self.model(input_tensor)
    self.assertEqual(output.shape[0], batch_size)
