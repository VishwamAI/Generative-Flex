import re

def fix_math_head():
    # Create proper class structure with fixed imports
    new_content = '''"""Math head implementation."""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers.enhanced_transformer import EnhancedTransformer
from src.models.reasoning.math_head_config import MathHeadConfig


class MathHead(nn.Module):
    """Math reasoning head implementation."""

    def __init__(
        self,
        config: MathHeadConfig,
        hidden_size: int = 768,
        num_experts: int = 4,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, config.expert_hidden_size),
                nn.GELU(),
                nn.Linear(config.expert_hidden_size, hidden_size),
                nn.Dropout(config.expert_dropout)
            )
            for _ in range(num_experts)
        ])

        self.router = nn.Linear(hidden_size, num_experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through math head.

        Args:
            hidden_states: Input hidden states
            attention_mask: Optional attention mask

        Returns:
            Tuple of output tensor and auxiliary losses dict
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Get router logits and probabilities
        router_logits = self.router(hidden_states)
        router_probs = F.softmax(router_logits, dim=-1)

        # Add router z-loss
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        aux_loss = self.config.router_z_loss_coef * z_loss

        # Get top-k routing weights
        k = 2 if self.config.router_type == "top_2" else 1
        top_k = torch.topk(router_probs, k=k, dim=-1)
        routing_weights = top_k.values
        routing_indices = top_k.indices

        # Normalize routing weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Dispatch to experts
        final_output = torch.zeros_like(hidden_states)
        for i in range(k):
            expert_index = routing_indices[..., i]
            expert_mask = F.one_hot(expert_index, num_classes=self.num_experts)
            for j, expert in enumerate(self.experts):
                expert_mask_j = expert_mask[..., j].unsqueeze(-1)
                expert_input = hidden_states * expert_mask_j
                expert_output = expert(expert_input)
                final_output += expert_output * routing_weights[..., i].unsqueeze(-1)

        aux_losses = {"router_z_loss": aux_loss}
        return final_output, aux_losses
'''

    # Write the new content
    with open('src/models/reasoning/math_head.py', 'w') as f:
        f.write(new_content)

if __name__ == '__main__':
    fix_math_head()
