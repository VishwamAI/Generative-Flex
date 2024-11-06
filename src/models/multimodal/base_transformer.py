"""Base transformer implementation."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class BaseTransformerConfig:
    """Configuration for base transformer."""

    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    activation: str = "gelu"
    dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 512

class BaseTransformer(nn.Module):
    """Base transformer model."""

    def __init__(self, config: Optional[BaseTransformerConfig] = None):
        """Initialize base transformer.

        Args:
            config: Optional model configuration
        """
        super().__init__()
        self.config = config or BaseTransformerConfig()
        self.setup_layers()

    def setup_layers(self):
        """Set up transformer layers."""
        self.embeddings = nn.ModuleDict({
            "word_embeddings": nn.Embedding(
                30522,  # Default vocab size
                self.config.hidden_size
            ),
            "position_embeddings": nn.Embedding(
                self.config.max_position_embeddings,
                self.config.hidden_size
            )
        })

        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.config.hidden_size,
                nhead=self.config.num_attention_heads,
                dim_feedforward=self.config.intermediate_size,
                dropout=self.config.dropout,
                activation=self.config.activation
            )
            for _ in range(self.config.num_hidden_layers)
        ])

        self.layernorm = nn.LayerNorm(self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Process input through transformer.


        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            position_ids: Optional position IDs

        Returns:
            Dictionary containing hidden states
        """
        # Embedding
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.size(1),
                device=input_ids.device
            ).unsqueeze(0)

        word_embeds = self.embeddings["word_embeddings"](input_ids)
        position_embeds = self.embeddings["position_embeddings"](position_ids)

        hidden_states = word_embeds + position_embeds
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Transformer layers
        for layer in self.encoder:
            hidden_states = layer(
                hidden_states,
                src_key_padding_mask=attention_mask
            )

        return {"hidden_states": hidden_states}
