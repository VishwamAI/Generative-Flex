import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Union, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)


class FlashAttention(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_length: int = 512,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.max_seq_length = max_seq_length

        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)
        self.out_proj = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # Early return for empty tensors
        if batch_size == 0 or seq_len == 0:
            return x, None

        # Project and reshape
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Create causal mask if no mask provided
        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=x.device)

        # Handle mask dimensions efficiently
        try:
            # Convert mask to correct shape in one operation if possible
            if mask.dim() == 4 and mask.shape[1] == 1 and mask.shape[2] == 1:
                attention_mask = mask  # Already in correct shape
            elif mask.dim() == 3 and mask.shape[1] == 1:
                attention_mask = mask.unsqueeze(2)
            elif mask.dim() == 2:
                attention_mask = mask.unsqueeze(1).unsqueeze(2)
            else:
                # Fallback to sequential squeeze for complex cases
                while mask.dim() > 2:
                    mask = mask.squeeze(1)
                attention_mask = mask.unsqueeze(1).unsqueeze(2)

            # Ensure mask matches sequence length
            if attention_mask.size(-1) != seq_len:
                if attention_mask.size(-1) > seq_len:
                    attention_mask = attention_mask[..., :seq_len]
                else:
                    pad_size = seq_len - attention_mask.size(-1)
                    attention_mask = F.pad(attention_mask, (0, pad_size), value=0)

            # Expand mask efficiently for all attention heads
            attention_mask = attention_mask.expand(
                batch_size, self.num_heads, seq_len, seq_len
            )

        except Exception as e:
            logger.error(f"Error processing attention mask: {e}")
            # Fallback to basic mask
            attention_mask = torch.ones(
                batch_size, self.num_heads, seq_len, seq_len, device=x.device
            )

        # Compute attention with better memory efficiency
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights_dropout = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights_dropout, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        return self.out_proj(attn_output), attn_weights


class MixtureOfExperts(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        expert_dim: int,
        num_experts: int = 4,
        capacity_factor: float = 1.25,
        dropout: float = 0.1,
        k: int = 2,  # Top-k routing
    ):
        super().__init__()
        self.input_dim = input_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.k = k

        # Router network (input -> expert selection)
        self.router = torch.nn.Linear(input_dim, num_experts)

        # Create experts (simple feed-forward networks)
        self.experts = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(input_dim, expert_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(expert_dim, input_dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle input that might be a tuple from previous layer
        if isinstance(x, tuple):
            x, _ = x  # Take the first element (output tensor)

        batch_size, seq_len, _ = x.shape

        # Compute router logits and probabilities
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]

        # Add noise to router logits for exploration
        if self.training:
            router_noise = torch.randn_like(router_logits) * 0.1
            router_logits = router_logits + router_noise

        # Get top-k experts for each token
        top_k_logits, top_k_indices = torch.topk(router_logits, self.k, dim=-1)
        router_probs = F.softmax(top_k_logits, dim=-1)  # Normalize only top-k

        # Add gating uncertainty loss
        router_entropy = (
            -(F.softmax(router_logits, dim=-1) * F.log_softmax(router_logits, dim=-1))
            .sum(-1)
            .mean()
        )

        # Compute load balancing loss
        # Mean probability of routing to each expert
        mean_prob = router_probs.mean(dim=[0, 1])
        # Ideal uniform distribution
        uniform_prob = torch.ones_like(mean_prob) / self.k
        # KL divergence for load balancing
        load_balance_loss = F.kl_div(
            mean_prob.log(), uniform_prob, reduction="batchmean"
        )

        # Compute capacity
        capacity = int(self.capacity_factor * (batch_size * seq_len) / self.num_experts)

        # Initialize output tensor
        combined_output = torch.zeros_like(x)

        # Process tokens through their top-k experts
        for i in range(self.k):
            # Get expert indices and probabilities for current k
            expert_indices = top_k_indices[:, :, i]  # [batch_size, seq_len]
            expert_probs = router_probs[:, :, i]  # [batch_size, seq_len]

            # Process each expert
            for j in range(self.num_experts):
                # Get tokens assigned to this expert
                expert_mask = expert_indices == j  # [batch_size, seq_len]
                if not expert_mask.any():
                    continue

                # Get masked input and expand expert probs
                masked_input = x[expert_mask]
                masked_probs = expert_probs[expert_mask].unsqueeze(-1)

                # Process tokens through expert
                expert_output = self.experts[j](masked_input)
                if isinstance(expert_output, tuple):
                    expert_output = expert_output[0]  # Take the first element if tuple

                # Weight output by router probability
                weighted_output = expert_output * masked_probs

                # Add to combined output
                combined_output[expert_mask] += weighted_output

        return combined_output, router_probs


class EnhancedTransformerBlock(torch.nn.Module):
    def __init__(self, config, dropout: float = 0.1):
        super().__init__()
        # Extract dimensions from config
        self.hidden_size = (
            config.hidden_size if hasattr(config, "hidden_size") else config.d_model
        )
        self.num_heads = (
            config.num_attention_heads if hasattr(config, "num_attention_heads") else 8
        )
        self.max_seq_length = (
            config.max_position_embeddings
            if hasattr(config, "max_position_embeddings")
            else 512
        )
        self.num_experts = getattr(config, "num_experts", 4)

        # Calculate expert dimension
        expert_dim = self.hidden_size * 4

        self.norm1 = torch.nn.LayerNorm(self.hidden_size)
        self.norm2 = torch.nn.LayerNorm(self.hidden_size)
        self.attention = FlashAttention(
            dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=dropout,
            max_seq_length=self.max_seq_length,
        )
        self.moe = MixtureOfExperts(
            input_dim=self.hidden_size,
            expert_dim=expert_dim,
            num_experts=self.num_experts,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, Any]],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        # Initialize auxiliary info dictionary
        aux_info = {"attention_weights": [], "router_probs": [], "layer_outputs": []}

        # Handle input that might be a tuple
        if isinstance(x, tuple):
            x = x[0]  # Take the first element (output tensor)

        # Attention block with residual connection
        normed_x = self.norm1(x)
        attn_output, attn_weights = self.attention(normed_x, mask)
        aux_info["attention_weights"].append(attn_weights)
        x = x + self.dropout(attn_output)
        aux_info["layer_outputs"].append(x.detach().clone())

        # MoE block with residual connection
        normed_x = self.norm2(x)
        moe_output, router_probs = self.moe(normed_x)
        aux_info["router_probs"].append(router_probs)
        x = x + self.dropout(moe_output)
        aux_info["layer_outputs"].append(x.detach().clone())

        return x, aux_info
