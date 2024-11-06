"""Test environment setup functionality."""

import unittest
import torch
from src.utils.environment_setup import EnvironmentSetup

class TestEnvironment(unittest.TestCase):
    """Test environment setup functionality."""

    def setUp(self):
        """Set up test environment."""
        self.env = EnvironmentSetup()

    def test_environment(self):
        """Test environment setup."""
        self.assertIsNotNone(self.env)
        self.env.setup()

    def test_cuda_setup(self):
        """Test CUDA setup."""
        if torch.cuda.is_available():
            self.assertTrue(self.env.setup_cuda())
