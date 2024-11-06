"""
Training script for sequence-to-sequence chain-of-thought model.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from dataclasses import dataclass
from src.models import Seq2SeqChainOfThoughtModel
from src.training.trainer import Trainer


@dataclass
class Seq2SeqCotConfig:

    """
Class for Seq2SeqCotConfig..
""""""
Configuration for sequence-to-sequence chain-of-thought training.
"""

batch_size: int = 16
learning_rate: float = 5e-5
num_epochs: int = 5
max_length: int = 1024
encoder_layers: int = 6
decoder_layers: int = 6

def main():


    """
Method for main..
"""
config = Seq2SeqCotConfig()
model = Seq2SeqChainOfThoughtModel()
trainer = Trainer(model, config)
trainer.train()

if __name__ == "__main__":
main()
