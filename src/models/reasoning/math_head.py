"""Mathematical reasoning head module."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class MathHeadConfig:
    """Configuration for mathematical reasoning head."""

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    dropout: float = 0.1
    num_experts: int = 4

class MathHead(nn.Module):
    """Mathematical reasoning head module."""

    def __init__(self, config: Optional[MathHeadConfig] = None):
        """Initialize math head.

        Args:
            config: Optional head configuration
        """
        super().__init__()
        self.config = config or MathHeadConfig()
        self.setup_layers()

    def setup_layers(self):
        """Set up neural network layers."""
        self.attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            dropout=self.config.dropout
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.intermediate_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.intermediate_size, self.config.hidden_size),
            nn.Dropout(self.config.dropout)
        )
        self.layer_norm1 = nn.LayerNorm(self.config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Process input through math head.

        Args:
            hidden_states: Input hidden states
            attention_mask: Optional attention mask

        Returns:
            Dictionary containing processed hidden states
        """
        # Self-attention
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.attention(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=attention_mask
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # Feed-forward
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return {"hidden_states": hidden_states}
