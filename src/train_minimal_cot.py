"""
Training script for minimal chain-of-thought model.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from dataclasses import dataclass
from src.models import MinimalChainOfThoughtModel
from src.training.trainer import Trainer


@dataclass
class MinimalCotConfig:

    """
Class for MinimalCotConfig..
""""""
Configuration for minimal chain-of-thought training.
"""

hidden_size: int = 768
batch_size: int = 32
learning_rate: float = 1e-4
num_epochs: int = 5
max_length: int = 512

def main():


    """
Method for main..
"""
config = MinimalCotConfig()
model = MinimalChainOfThoughtModel(config.hidden_size)
trainer = Trainer(model, config)
trainer.train()

if __name__ == "__main__":
main()
