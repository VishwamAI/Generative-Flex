"""Math reasoning module."""
from dataclasses import dataclass, field
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import numpy as np
import os
import torch
import torch.nn as nn

class MathReasoning(nn.Module):
    """Mathematical reasoning module."""

    def __init__(self, config):
        """Initialize the math reasoning module."""
        super().__init__()
        self.config = config
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()
        self.num_experts = config.num_experts
        self.expert_layers = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size)
            for _ in range(self.num_experts)
        ])
        self.router = nn.Linear(config.hidden_size, self.num_experts)
        self.output_layer = nn.Linear(config.hidden_size, config.num_math_tokens)

    def forward(self, x, attention_mask=None):
        """Forward pass of the math reasoning module.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask

        Returns:
            Tuple containing:
            - logits: Output tensor of shape (batch_size, seq_len, num_math_tokens)
            - router_probs: Router probabilities for each expert
        """
        # Layer norm and initial dense projection
        x = self.layer_norm(x)
        x = self.dense(x)
        x = self.activation(x)

        # Router logits and expert weights
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]
        router_probs = torch.softmax(router_logits, dim=-1)

        # Apply each expert and combine with routing weights
        expert_outputs = []
        for i, expert in enumerate(self.expert_layers):
            expert_out = expert(x)
            if attention_mask is not None:
                expert_out = expert_out * attention_mask.unsqueeze(-1)
            expert_outputs.append(expert_out * router_probs[..., i:i+1])

        # Combine expert outputs
        x = sum(expert_outputs)
        x = self.dropout(x)

        # Final output projection
        logits = self.output_layer(x)

        return logits, router_probs
