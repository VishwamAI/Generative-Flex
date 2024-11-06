"""
Test GPU utilities functionality...
"""

import torch

from src.utils.gpu_utils import GPUUtils
import unittest


class TestGPU(unittest.TestCase):
    """
Test GPU utilities functionality...
"""

    def setUp(self):
        """
Set up test environment...
"""
        self.utils = GPUUtils()

    def test_gpu_memory(self):
        """
Test GPU memory utilities...
"""
        if torch.cuda.is_available():
            memory_info = self.utils.get_memory_info()
            self.assertIsNotNone(memory_info)

    def test_gpu_availability(self):
        """
Test GPU availability check...
"""
        is_available = self.utils.is_gpu_available()
        self.assertIsInstance(is_available, bool)
