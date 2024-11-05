import torch
import torch.nn as nn

"""Mixture of Experts Implementation for Generative-Flex."""


class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer implementation."""

    def __init__(self, num_experts, input_size, output_size) -> None:
        """Initialize the MoE layer.

        Args:
        num_experts: Number of expert networks
        input_size: Size of input features
        output_size: Size of output features
        """
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(input_size, num_experts)

        def forward(self, x) -> None:
            """Forward pass through the MoE layer."""
            # Get expert weights
            expert_weights = torch.softmax(self.gate(x), dim=-1)

            # Get expert outputs
            expert_outputs = torch.stack([expert(x) for expert in self.experts])

            # Combine expert outputs
            output = torch.sum(expert_weights.unsqueeze(-1) * expert_outputs, dim=0)
            return output
