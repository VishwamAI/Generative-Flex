"""
Mixture of Experts Implementation for Generative-Flex
Implements conditional computation paths for specialized processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ExpertLayer(nn.Module):
    """
    Individual expert network implementing a specialized computation path
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(self.activation(self.w1(x))))


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer with load balancing and capacity factor
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        k: int = 2,  # Top-k experts to route to
        capacity_factor: float = 1.25,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor

        # Create experts
        self.experts = nn.ModuleList(
            [ExpertLayer(d_model, d_ff, dropout) for _ in range(num_experts)]
        )

        # Router network
        self.router = nn.Linear(d_model, num_experts)
        self.dropout = nn.Dropout(dropout)

    def _compute_routing_weights(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute routing probabilities and expert assignments"""
        # Shape: [batch_size, seq_len, num_experts]
        router_logits = self.router(x)

        if mask is not None:
            router_logits = router_logits.masked_fill(
                ~mask.unsqueeze(-1), float("-inf")
            )

        # Get top-k experts
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1), self.k, dim=-1
        )

        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        return routing_weights, selected_experts

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        # Compute routing weights and expert assignments
        routing_weights, selected_experts = self._compute_routing_weights(x, mask)

        # Initialize output tensor
        output = torch.zeros_like(x)

        # Compute capacity
        capacity = int(self.capacity_factor * (batch_size * seq_len) / self.num_experts)

        # Process tokens through selected experts
        for i in range(self.k):
            # Get expert indices and corresponding weights
            expert_indices = selected_experts[..., i]
            expert_weights = routing_weights[..., i].unsqueeze(-1)

            # Process each expert
            for expert_idx in range(self.num_experts):
                # Find tokens routed to this expert
                expert_mask = expert_indices == expert_idx
                if not expert_mask.any():
                    continue

                # Select tokens for this expert
                expert_input = x[expert_mask]

                # Apply capacity constraint
                if expert_input.size(0) > capacity:
                    # Randomly drop tokens that exceed capacity
                    perm = torch.randperm(expert_input.size(0), device=x.device)
                    expert_input = expert_input[perm[:capacity]]
                    expert_mask[expert_mask.clone()] = False
                    expert_mask[expert_mask.clone()][perm[:capacity]] = True

                # Process tokens through expert
                expert_output = self.experts[expert_idx](expert_input)

                # Combine expert output with routing weights
                output[expert_mask] += expert_output * expert_weights[expert_mask]

        return self.dropout(output)
