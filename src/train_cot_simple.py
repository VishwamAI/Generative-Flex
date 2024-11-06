"""
Training script for simple chain-of-thought model.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional
from src.models import SimpleChainOfThoughtModel
from src.training.trainer import Trainer

@dataclass
class SimpleCotConfig:

    """Class for SimpleCotConfig.""""""
Configuration for simple chain-of-thought training.
"""

batch_size: int = 16
learning_rate: float = 5e-5
num_epochs: int = 5
max_length: int = 512

def main():


    """Method for main."""
config = SimpleCotConfig()
model = SimpleChainOfThoughtModel()
trainer = Trainer(model, config)
trainer.train()

if __name__ == "__main__":
main()
