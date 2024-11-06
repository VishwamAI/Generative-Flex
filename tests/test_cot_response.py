"""Test chain-of-thought response generation."""

import unittest
import torch
import torch.nn as nn
from src.models import ChainOfThoughtModel

class TestCotResponse(unittest.TestCase):
    """Test chain-of-thought response generation."""

    def setUp(self):
        """Set up test environment."""
        self.model = ChainOfThoughtModel()

    def test_response_generation(self):
        """Test response generation."""
        input_text = "What is 2+2?"
        input_tensor = torch.randint(0, 1000, (1, 32))
        output = self.model(input_tensor)
        self.assertEqual(output.shape[-1], 32)

    def test_batch_response(self):
        """Test batch response generation."""
        batch_size = 16
        input_tensor = torch.randint(0, 1000, (batch_size, 32))
        output = self.model(input_tensor)
        self.assertEqual(output.shape[0], batch_size)
