"""Test environment setup functionality."""

import unittest
import torch
from transformers import AutoModelForCausalLM
from src.utils.environment_setup import EnvironmentSetup

class TestEnvironment(unittest.TestCase):
    """Test environment setup functionality."""

    def setUp(self):
        """Set up test environment."""
        self.env_setup = EnvironmentSetup()

    def test_environment_initialization(self):
        """Test environment initialization."""
        self.assertIsNotNone(self.env_setup)
        self.env_setup.setup()

    def test_cuda_availability(self):
        """Test CUDA availability check."""
        if torch.cuda.is_available():
            self.assertTrue(torch.cuda.is_initialized())
