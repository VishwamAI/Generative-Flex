"""
Advanced Transformer Layer Implementation for Generative-Flex
Combines Flash Attention and Mixture of Experts for optimal performance
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import FlashAttention
from .experts import MixtureOfExperts


class TransformerLayer(nn.Module):
    """
    Advanced transformer layer combining Flash Attention and Mixture of Experts
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        num_experts: int = 8,
        expert_capacity_factor: float = 1.25,
        block_size: int = 1024,
    ):
        super().__init__()

        # Flash Attention for efficient self-attention
        self.self_attn = FlashAttention(
            d_model=d_model, n_heads=nhead, dropout=dropout, block_size=block_size
        )

        # Mixture of Experts for specialized computation
        self.moe = MixtureOfExperts(
            d_model=d_model,
            d_ff=dim_feedforward,
            num_experts=num_experts,
            capacity_factor=expert_capacity_factor,
            dropout=dropout,
        )

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass combining attention and expert computation
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional attention mask
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, mask)
        x = residual + self.dropout(x)

        # Mixture of Experts with residual connection
        residual = x
        x = self.norm2(x)
        x = self.moe(x, mask)
        x = residual + self.dropout(x)

        return x
