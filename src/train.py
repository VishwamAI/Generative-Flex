"""Main training script for Generative-Flex."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional
from src.models import SimpleModel
from src.training.trainer import Trainer
from src.utils.training_utils import TrainingUtils

@dataclass
class TrainingConfig:
    """Training configuration."""

    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    """Run main training loop."""
    config = TrainingConfig()
    model = SimpleModel().to(config.device)
    trainer = Trainer(model, config)
    trainer.train()

if __name__ == "__main__":
    main()
