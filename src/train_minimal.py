"""Training script for minimal model.."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional
from src.models import MinimalModel
from src.training.trainer import Trainer

@dataclass
class MinimalConfig:
"""Configuration for minimal model training.."""

hidden_size: int = 768
batch_size: int = 32
learning_rate: float = 1e-4
num_epochs: int = 5

def main():
"""Run minimal model training.."""
config = MinimalConfig()
model = MinimalModel(config.hidden_size)
trainer = Trainer(model, config)
trainer.train()

if __name__ == "__main__":
main()
