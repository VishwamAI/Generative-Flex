"""
Training script for chatbot model.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional
from src.models import ChatbotModel
from src.training.trainer import Trainer

@dataclass
class ChatbotConfig:
"""
Configuration for chatbot training.
"""

batch_size: int = 16
learning_rate: float = 5e-5
num_epochs: int = 5
max_length: int = 512
file_path: str = "data/chatbot/training_data_cot.json"

def main():
"""
Run chatbot training.
"""
config = ChatbotConfig()
model = ChatbotModel()
trainer = Trainer(model, config)
trainer.train()

if __name__ == "__main__":
main()