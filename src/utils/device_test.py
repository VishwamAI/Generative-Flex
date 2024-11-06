"""Test device configuration functionality.."""

import unittest
import torch
from src.utils.device_config import DeviceConfig

class TestDeviceConfig(unittest.TestCase):
    """Test device configuration functionality.."""

    def setUp(self):
        """Set up test environment.."""
        self.config = DeviceConfig()

    def test_device_configuration(self):
        """Test device configuration.."""
        device = self.config.get_device()
        self.assertIsNotNone(device)

    def test_cuda_availability(self):
        """Test CUDA availability check.."""
        if torch.cuda.is_available():
            self.assertTrue(self.config.is_cuda_available())
