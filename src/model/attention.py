"""
Flash Attention Implementation for Generative-Flex
Optimized attention mechanism with O(N) memory complexity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class FlashAttention(nn.Module):
    """
    Flash Attention implementation with optimized memory usage and computation
    Based on "Flash Attention: Fast and Memory-Efficient Exact Attention"
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        block_size: int = 1024,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.block_size = block_size
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split heads and reshape: (B, L, D) -> (B, H, L, D//H)"""
        B, L, D = x.shape
        x = x.view(B, L, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge heads: (B, H, L, D//H) -> (B, L, D)"""
        B, H, L, D = x.shape
        x = x.transpose(1, 2)
        return x.reshape(B, L, H * D)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = q.shape

        # Project and split heads
        q = self._split_heads(self.q_proj(q))
        k = self._split_heads(self.k_proj(k))
        v = self._split_heads(self.v_proj(v))

        # Initialize output tensor
        output = torch.zeros_like(q)

        # Process attention in blocks for memory efficiency
        for i in range(0, L, self.block_size):
            j_end = min(i + self.block_size, L)
            q_block = q[:, :, i:j_end]
            scores = torch.matmul(q_block, k.transpose(-2, -1)) * self.scale

            if mask is not None:
                mask_block = (
                    mask[:, i:j_end, :] if mask.dim() == 3 else mask[i:j_end, :]
                )
                scores = scores.masked_fill(~mask_block.unsqueeze(1), float("-inf"))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            output[:, :, i:j_end] = torch.matmul(attn_weights, v)

        # Merge heads and project output
        output = self._merge_heads(output)
        return self.out_proj(output)
