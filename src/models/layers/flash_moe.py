"""Flash Mixture of Experts layer implementation.."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class FlashMoEConfig:
    """Configuration for Flash MoE layer.."""

    hidden_size: int = 768
    num_experts: int = 4
    expert_capacity: int = 128
    dropout: float = 0.1
    activation: str = "gelu"

class FlashMoE(nn.Module):
    """Flash Mixture of Experts layer.."""

    def __init__(self, config: Optional[FlashMoEConfig] = None):
        """Initialize Flash MoE layer.

        Args:
            config: Optional layer configuration"""
        super().__init__()
        self.config = config or FlashMoEConfig()
        self.setup_experts()

    def setup_experts(self):
        """Set up expert networks.."""
        self.gate = nn.Linear(self.config.hidden_size, self.config.num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.config.hidden_size, 4 * self.config.hidden_size),
                nn.GELU() if self.config.activation == "gelu" else nn.ReLU(),
                nn.Linear(4 * self.config.hidden_size, self.config.hidden_size),
                nn.Dropout(self.config.dropout)
            )
            for _ in range(self.config.num_experts)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Process input through Flash MoE layer.

        Args:
            hidden_states: Input hidden states
            attention_mask: Optional attention mask

        Returns:
            Dictionary containing processed hidden states"""
        # Gate computation
        gate_logits = self.gate(hidden_states)
        expert_weights = torch.softmax(gate_logits, dim=-1)

        # Expert computation
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(hidden_states)
            weighted_output = expert_output * expert_weights[..., i].unsqueeze(-1)
            expert_outputs.append(weighted_output)

        # Combine expert outputs
        combined_output = sum(expert_outputs)

        return {"hidden_states": combined_output}
