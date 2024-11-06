"""
Enhanced transformer layer implementation with advanced features.
"""
from typing import Optional, Dict, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedTransformer(nn.Module):
    """
Enhanced transformer layer with advanced attention mechanisms.
"""

    def __init__(self,
        hidden_size: int,
        num_attention_heads: int,
        dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.attention_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
Transpose and reshape tensor for attention computation.
"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,) -> Tuple[torch.Tensor, torch.Tensor]:
        """
Forward pass of the enhanced transformer layer.
"""
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(self.attention_head_size, dtype=attention_scores.dtype)
        )

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        output = self.dropout(context_layer)
        output = self.layer_norm(output + hidden_states)
        return output, attention_probs
