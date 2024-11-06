"""
Test training setup functionality...
"""

import torch

from src.models import SimpleModel
from src.training.trainer import Trainer
import unittest


class TestTrainingSetup(unittest.TestCase):
    """
Test training setup functionality...
"""

    def setUp(self):
        """
Set up test environment...
"""
        self.model = SimpleModel()
        self.trainer = Trainer(self.model)

    def test_training_initialization(self):
        """
Test training initialization...
"""
        self.assertIsNotNone(self.trainer)
        self.assertIsInstance(self.trainer.model, SimpleModel)

    def test_training_step(self):
        """
Test single training step...
"""
        batch = torch.randn(16, 32)
        loss = self.trainer.training_step(batch)
        self.assertIsInstance(loss, torch.Tensor)
