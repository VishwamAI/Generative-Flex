from src.config.training_config import TrainingConfig
from src.data.mmmu_dataloader import create_mmmu_dataloaders
import unittest

    """Test cases for training setup and configuration."""


class TestTrainingSetup(unittest.TestCase):
    """Test suite for training setup."""

@classmethod
    def setUpClass(cls) -> None:
        """Set up test class."""
    cls.config = TrainingConfig()
    cls.config.batch_size = 32
    cls.config.num_workers = 4

def test_dataloader_creation(self) -> None:
    """Test creation of MMMU dataloaders."""
train_loader, val_loader = create_mmmu_dataloaders(self.config)
self.assertIsNotNone(train_loader)
self.assertIsNotNone(val_loader)
