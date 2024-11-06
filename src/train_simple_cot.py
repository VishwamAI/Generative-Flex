"""Training script for simple chain-of-thought model."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional
from src.models import SimpleChainOfThoughtModel
from src.training.trainer import Trainer

@dataclass
class SimpleChainOfThoughtConfig:
    """Configuration for simple chain-of-thought training."""

    batch_size: int = 16
    learning_rate: float = 5e-5
    num_epochs: int = 5
    max_length: int = 512
    hidden_size: int = 768

def main():
    """Run simple chain-of-thought training."""
    config = SimpleChainOfThoughtConfig()
    model = SimpleChainOfThoughtModel()
    trainer = Trainer(model, config)
    trainer.train()

if __name__ == "__main__":
    main()
