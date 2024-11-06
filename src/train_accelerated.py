"""Training script using AcceleratedTrainer for efficient distributed training.."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional
from src.models import SimpleModel
from src.training.accelerated_trainer import AcceleratedTrainer
from src.utils.training_utils import TrainingUtils

@dataclass
class AcceleratedConfig:
    """Configuration for accelerated training.."""

    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    num_gpus: int = torch.cuda.device_count()
    mixed_precision: bool = True

def main():
    """Run accelerated training.."""
    config = AcceleratedConfig()
    model = SimpleModel()
    trainer = AcceleratedTrainer(model, config)
    trainer.train()

if __name__ == "__main__":
    main()
