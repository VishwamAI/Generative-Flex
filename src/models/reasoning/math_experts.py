from typing import Tuple
import torch
import torch.nn as nn

    """Specialized experts for mathematical reasoning."""


class MathematicalExpert(nn.Module):
    """Expert module specialized for mathematical operations."""

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.dense = nn.Linear(self.hidden_size, self.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.dense_output = nn.Linear(self.intermediate_size, self.hidden_size)

        def forward(
        self, hidden_states: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass through the mathematical expert."""
        intermediate_output = self.dense(hidden_states)
        intermediate_output = self.intermediate_act_fn(intermediate_output)

        layer_output = self.dense_output(intermediate_output)
        layer_output = self.dropout(layer_output)

        return layer_output, torch.mean(intermediate_output, dim=-1)
