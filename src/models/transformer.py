import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
@dataclass
class TransformerConfig:

    """Configuration for Transformer model.
"""

hidden_size: int = 768
num_attention_heads: int = 12
num_hidden_layers: int = 12
intermediate_size: int = 3072
hidden_dropout_prob: float = 0.1
attention_probs_dropout_prob: float = 0.1

class Transformer:
"""
Transformer model implementation.
"""

    def __init__(self, config: Optional[TransformerConfig] = None):


        """Method for __init__."""super().__init__()
    self.config = config or TransformerConfig()

    self.encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(
    d_model=self.config.hidden_size,
    nhead=self.config.num_attention_heads,
    dim_feedforward=self.config.intermediate_size,
    dropout=self.config.hidden_dropout_prob,
    activation='gelu'
    ),
    num_layers=self.config.num_hidden_layers
    )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):


        """Method for forward."""
    return self.encoder(x, mask=mask)
