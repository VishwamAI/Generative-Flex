import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class SimpleModelConfig:
    """Configuration for SimpleModel."""

    hidden_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.1

class SimpleModel(nn.Module):
    """A simple neural network model."""

    def __init__(self, config: Optional[SimpleModelConfig] = None):
        super().__init__()
        self.config = config or SimpleModelConfig()

        self.layers = nn.ModuleList([
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
            for _ in range(self.config.num_layers)
        ])
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        for layer in self.layers:
            x = self.dropout(torch.relu(layer(x)))
        return x
