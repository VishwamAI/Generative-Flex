import torch.nn as nn

"""Symbolic mathematics processing module."""


class SymbolicMathProcessor(nn.Module):
    """Processes symbolic mathematics expressions."""

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dropout_prob = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(self.dropout_prob)

        def forward(self, x) -> None:
            """Forward pass for symbolic math processing."""
            return self.dropout(x)
