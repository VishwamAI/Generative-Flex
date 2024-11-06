"""Training script for chain-of-thought model with fixed prompts."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional
from src.models import ChainOfThoughtModel
from src.training.trainer import Trainer

@dataclass
class CotConfig:
    """Configuration for chain-of-thought training."""

    batch_size: int = 16
    learning_rate: float = 5e-5
    num_epochs: int = 5
    max_length: int = 1024
    prompt_template: str = "Let's solve this step by step:"

def main():
    """Run chain-of-thought training."""
    config = CotConfig()
    model = ChainOfThoughtModel()
    trainer = Trainer(model, config)
    trainer.train()

if __name__ == "__main__":
    main()
