import math
import torch
import torch.nn as nn

"""Flash Attention Implementation for Generative-Flex."""


class FlashAttention(nn.Module):
    """Efficient attention implementation using flash attention algorithm."""

    def __init__(self, hidden_size, num_heads, dropout_prob=0.1) -> None:
        """Initialize the Flash Attention module.

        Args:
        hidden_size: Size of hidden dimension
        num_heads: Number of attention heads
        dropout_prob: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.dropout = nn.Dropout(dropout_prob)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        def forward(self, x, mask=None) -> None:
            """Forward pass implementing flash attention algorithm."""
            batch_size = x.size(0)

            # Project queries, keys, and values
            q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.head_size)
            k = self.k_proj(x).view(batch_size, -1, self.num_heads, self.head_size)
            v = self.v_proj(x).view(batch_size, -1, self.num_heads, self.head_size)

            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))

                # Apply softmax and dropout
                attn = self.dropout(torch.softmax(scores, dim=-1))

                # Get output
                out = torch.matmul(attn, v)
                out = out.view(batch_size, -1, self.hidden_size)

                return self.out_proj(out)
