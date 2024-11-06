"""
Training utility functions..
"""
import torch
from dataclasses import dataclass
from typing import Dict, Optional
@dataclass
class TrainingParams:
    """
Training parameters configuration..
"""

    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    gradient_clip_val: float = 1.0
    weight_decay: float = 0.01

class TrainingUtils:
    """
Utility functions for training..
"""

    def __init__(self, params: Optional[TrainingParams] = None):
        """
Initialize training utilities.

        Args:
            params: Optional training parameters
"""
        self.params = params or TrainingParams()

    def get_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
Get optimizer for model.

        Args:
            model: PyTorch model

        Returns:
            Configured optimizer
"""
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.params.learning_rate,
            weight_decay=self.params.weight_decay
        )

    def get_scheduler(self,
        optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        """
Get learning rate scheduler.

        Args:
            optimizer: PyTorch optimizer

        Returns:
            Learning rate scheduler
"""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.params.num_epochs
        )
