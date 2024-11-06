"""Mathematical reasoning module."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class MathReasoningConfig:
    """Configuration for mathematical reasoning."""

    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    num_experts: int = 8
    expert_hidden_size: int = 1024
    dropout_prob: float = 0.1

class MathReasoning(nn.Module):
    """Mathematical reasoning module."""

    def __init__(self, config: Optional[MathReasoningConfig] = None):
        """Initialize mathematical reasoning module.

        Args:
            config: Optional configuration
        """
        super().__init__()
        self.config = config or MathReasoningConfig()

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.expert_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.config.dropout_prob),
                nn.Linear(self.config.expert_hidden_size, self.config.hidden_size)
            )
            for _ in range(self.config.num_experts)
        ])

        self.router = nn.Linear(self.config.hidden_size, self.config.num_experts)
        self.dropout = nn.Dropout(self.config.dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through mathematical reasoning module.

        Args:
            hidden_states: Input hidden states
            attention_mask: Optional attention mask

        Returns:
            Dictionary containing output tensors
        """
        # Route input to experts
        router_logits = self.router(hidden_states)
        routing_weights = torch.softmax(router_logits, dim=-1)

        # Apply experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(hidden_states)
            weighted_output = expert_output * routing_weights[..., i:i+1]
            expert_outputs.append(weighted_output)

        # Combine expert outputs
        combined_output = sum(expert_outputs)
        output = self.dropout(combined_output)

        return {
            "hidden_states": output,
            "routing_weights": routing_weights
        }
