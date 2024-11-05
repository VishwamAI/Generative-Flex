from transformers import PretrainedConfig
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F




class EnhancedTransformerBlock(nn.Module):
    """Enhanced transformer block with additional capabilities for mathematical reasoning"""


    def __init__(self, config: PretrainedConfig, dropout: float = 0.1):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.num_attention_heads = config.num_attention_heads
    self.dropout_prob = dropout

    # Multi-head attention
    self.attention = nn.MultiheadAttention(
    embed_dim=config.hidden_size,
    num_heads=config.num_attention_heads,
    dropout=dropout,
    batch_first=True,
    )

    # Feed-forward network
    self.feed_forward = nn.Sequential(
    nn.Linear(config.hidden_size, config.hidden_size * 4),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(config.hidden_size * 4, config.hidden_size),
    nn.Dropout(dropout),
    )

    # Layer normalization
    self.attention_norm = nn.LayerNorm(config.hidden_size)
    self.feed_forward_norm = nn.LayerNorm(config.hidden_size)

    # Optional: Flash Attention support
    self.use_flash_attention = getattr(config, "flash_attention", False)

    # Gradient checkpointing
    self.gradient_checkpointing = False

    def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
    """Forward pass for the enhanced transformer block
    Args:
    hidden_states: Input tensor of shape(
    batch_size,
    seq_length,
    hidden_size
    )
    attention_mask: Optional attention mask
    layer_past: Optional past key/value states for incremental decoding
    use_cache: Whether to use past key/value states
    Returns:
    Tuple of (output tensor, auxiliary info dictionary)"""

    # Prepare attention mask
    if attention_mask is not None:
    # Convert mask to float and proper shape for nn.MultiheadAttention
    # MultiheadAttention expects mask of shape(
    batch_size,
    seq_length,
    seq_length
    )
    batch_size, seq_length = hidden_states.shape[:2]

    # Handle different input mask shapes
    if attention_mask.dim() == 2:
    attention_mask = attention_mask.unsqueeze(1)
    elif attention_mask.dim() > 3:
    attention_mask = attention_mask.squeeze()
    while attention_mask.dim() > 3:
    attention_mask = attention_mask.squeeze(1)

    # Ensure proper size by truncating or padding
    if attention_mask.size(-1) != seq_length:
    if attention_mask.size(-1) > seq_length:
    attention_mask = attention_mask[...,:seq_length]
    else:
    pad_size = seq_length - attention_mask.size(-1)
    attention_mask = F.pad(
    attention_mask, (0, pad_size), value=0
    )

    # Create causal mask of matching size
    causal_mask = torch.triu(
    torch.ones(
    seq_length, seq_length, device=hidden_states.device
    ),
    diagonal=1,
    ).bool()

    # Ensure attention mask has correct shape before applying causal mask
    attention_mask = attention_mask.expand(
    batch_size, seq_length, seq_length
    )
    attention_mask = attention_mask.masked_fill(
    causal_mask,
    0
    )

    # Convert to float and proper values for attention
    attention_mask = attention_mask.to(dtype=hidden_states.dtype)
    attention_mask = (1.0 - attention_mask) * -10000.0

    # Reshape attention mask for multi-head attention
    attention_mask = attention_mask.repeat(
    self.num_attention_heads, 1, 1
    )

    # Self-attention
    residual = hidden_states
    hidden_states = self.attention_norm(hidden_states)

    if self.gradient_checkpointing and self.training:

    def create_custom_forward(module):
    def custom_forward(*inputs):
    return module(*inputs)

    return custom_forward

    attn_output, attn_weights = torch.utils.checkpoint.checkpoint(
    create_custom_forward(self.attention),
    hidden_states,
    hidden_states,
    hidden_states,
    attention_mask,
    None,
    )
    else:
    attn_output, attn_weights = self.attention(
    query=hidden_states,
    key=hidden_states,
    value=hidden_states,
    attn_mask=(
    attention_mask if attention_mask is not None else None
    ),
    need_weights=True,  # Get attention weights
    is_causal=True,  # Enable causal masking
    )

    hidden_states = residual + attn_output

    # Feed-forward network
    residual = hidden_states
    hidden_states = self.feed_forward_norm(hidden_states)

    if self.gradient_checkpointing and self.training:
    hidden_states = torch.utils.checkpoint.checkpoint(
    create_custom_forward(
    self.feed_forward),
    hidden_states
    )
    )
    else:
    hidden_states = self.feed_forward(hidden_states)

    hidden_states = residual +
    hidden_states

    # Return both hidden states and auxiliary information
    auxiliary_info = {
    "attention_weights": attn_weights,
    "attention_output": attn_output,
    "intermediate_states": hidden_states.detach(
    ),

    )
    "layer_past": layer_past,
    }

    return hidden_states, auxiliary_info
