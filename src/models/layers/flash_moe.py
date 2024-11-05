from typing import Optional, Tuple
import torch
import torch.nn as nn

"""
Flash Mixture of Experts implementation.
"""



class FlashMoELayer(nn.Module):
    """
    Flash Mixture of Experts layer implementation.
    """

    def __init__():
    self,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int  = 8,
    dropout_rate: float  = 0.1,
    ):
        """
        Initialize the FlashMoE layer.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.dropout = nn.Dropout(dropout_rate)

        # Expert network
        self.experts = nn.ModuleList([
        nn.Sequential(
        nn.Linear(hidden_size, intermediate_size),
        nn.GELU(),
        nn.Linear(intermediate_size, hidden_size),
        nn.Dropout(dropout_rate),
        )
        for _ in range(num_experts)
        ]
        )

        # Router network
        self.router = nn.Linear(hidden_size, num_experts)

    def __init__(self, forward():
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor]  = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the FlashMoE layer.
        """
        batch_size, seq_length, hidden_size = hidden_states.shape

        # Get routing weights
        routing_weights = torch.softmax(self.router(hidden_states), dim=-1)

        # Initialize output tensor
        combined_output = torch.zeros_like(hidden_states)

        # Apply each expert
        for i, expert in enumerate(self.experts):
            expert_output = expert(hidden_states)
            combined_output += routing_weights[..., i : i + 1] * expert_output

            return combined_output, routing_weights
