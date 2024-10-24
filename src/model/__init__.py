"""
Advanced Generative-Flex Model Implementation
Core model architecture with state-of-the-art optimizations
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from .transformer import TransformerLayer


class AdvancedGenerativeFlexModel(nn.Module):
    """
    Advanced transformer-based model with optimized architecture featuring:
    - Flash Attention for efficient O(N) memory complexity
    - Mixture of Experts for specialized computation paths
    - Optimized transformer layers with advanced normalization

    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of the model (default: 1024)
        nhead: Number of attention heads (default: 16)
        num_layers: Number of transformer layers (default: 24)
        dim_feedforward: Dimension of feedforward network (default: 4096)
        dropout: Dropout rate (default: 0.1)
        max_seq_length: Maximum sequence length (default: 2048)
        num_experts: Number of expert networks per layer (default: 8)
        expert_capacity_factor: Capacity factor for expert routing (default: 1.25)
        attention_block_size: Block size for flash attention (default: 1024)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 1024,
        nhead: int = 16,
        num_layers: int = 24,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
        max_seq_length: int = 2048,
        num_experts: int = 8,
        expert_capacity_factor: float = 1.25,
        attention_block_size: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model

        # Token and positional embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(max_seq_length, d_model)

        # Advanced transformer layers with Flash Attention and MoE
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    num_experts=num_experts,
                    expert_capacity_factor=expert_capacity_factor,
                    block_size=attention_block_size,
                )
                for _ in range(num_layers)
            ]
        )

        # Output layers
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Initialize parameters with scaled initialization
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters with scaled initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(
                    p, gain=1 / math.sqrt(2)  # Scale for better gradient flow
                )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the model

        Args:
            x: Input tensor of shape [batch_size, seq_len]
            mask: Optional attention mask
            return_attention_weights: Whether to return attention weights

        Returns:
            Output tensor of shape [batch_size, seq_len, vocab_size]
        """
        # Get sequence length and create position indices
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # Combine token and positional embeddings
        x = self.embedding(x) * math.sqrt(self.d_model)  # Scale embeddings
        x = x + self.pos_encoder(pos)

        # Process through transformer layers
        attention_weights = []
        for layer in self.transformer_layers:
            if return_attention_weights:
                x, attn = layer(x, mask, return_attention=True)
                attention_weights.append(attn)
            else:
                x = layer(x, mask)

        # Output processing
        x = self.norm(x)
        logits = self.fc_out(x)

        if return_attention_weights:
            return logits, attention_weights
        return logits
