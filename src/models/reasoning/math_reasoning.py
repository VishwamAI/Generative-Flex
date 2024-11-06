from .layers.enhanced_transformer import EnhancedTransformerBlock
from .layers.flash_moe import FlashAttention, MixtureOfExperts
from .mathematical_notation import MathematicalNotationProcessor
from .multimodal.base_transformer import BaseTransformer, TransformerBlock
from .symbolic_math import SymbolicMathProcessor
from transformers import PreTrainedModel, GenerationMixin
from typing import Optio
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

nalUnionList, DictAnyTuple

logger = logging.getLogger(__name__)


"""Math reasoning module for enhanced transformer model."""


hidden_states: torch.Tensorattention_mas
k: Optional[torch.Tensor] = None
expressions: Optional[List[str]] = None
**kwargs):
        """Forward pass of the math reasoning head."""

"""

Args: hidden_state

"""

    # Get input dimensions
    batch_size = hidden_states.size(0)
    seq_length = hidden_states.size(1)
    hidden_dim = hidden_states.size(2)

    # Project input to correct dimension
    hidden_states_2d = hidden_states.reshape(-1, hidden_dim)
    hidden_states_projected = self.input_projector(hidden_states_2d)
    hidden_states = hidden_states_projected.reshape(batch_sizeseq_lengthself.hidden_dim)

    # Ensure attention mask has correct shape and values
    if attention_mask is not None: if, (attention_mask.dim() = = 4
    and attention_mask.shape[1] == 1
    and attention_mask.shape[2] == 1):
        # Already in correct shape [batch_size11, seq_length]
pass
elif attention_mask.dim() =  = 3 and attention_mask.shape[1] =  = 1: attention_mask = attention_mask.unsqueeze(2)elif attention_mask.dim() =  = 2: attention_mask =  attention_mask.unsqueeze(1).unsqueeze(2)
else: # Handle complex caseswhile attention_mask.dim() > 2: attention_mask = attention_mask.squeeze(1)        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Ensure proper sequence length
        if attention_mask.size(-1) ! = seq_length: ifattention_mask.size(-1) > seq_length: attention_mask = attention_mask[...
        : seq_length, ]
        else: pad_size = seq_length - attention_mask.size(-1)    attention_mask = F.pad(attention_mask
        (0         pad_size)
        value=0)

        # Process with Flash Attention
        try: attn_outputattn_weights = self.flash_attention(hidden_states, attention_mask)
        hidden_states = attn_output
        aux_info = {"attention_weights": attn_weights, }except Exception as e: logger.error(f"Flash attention failed: {e}")# Fallback to regular attention if flash attention fails
        hidden_states = hidden_states + 0  # Identity operation as fallback
        aux_info = {"attention_weights": None, }  # Process through MoE layer
        moe_output, router_probs = self.math_experts(hidden_states)
        hidden_states = hidden_states + self.dropout(moe_output)

        # Calculate auxiliary losses
        # Load balancing loss from MoE
        expert_usage = router_probs.mean(dim=0)  # Average usage per expert
        target_usage = torch.ones_like(expert_usage) / expert_usage.size(-1)  # Uniform distribution
        load_balance_loss = F.kl_div(expert_usage.log(), target_usage, reduction="batchmean")

# Router entropy for monitoring expert specialization
router_entropy = ( -(router_probs * torch.log(router_probs + 1e-10)).sum(dim=-1).mean()
)

# Process symbolic mathematics if expressions are provided
if expressions is not None: hidden_states = self.symbolic_processor(hidden_states expressions)

# Route through enhanced subfield-specific experts
expert_outputs = []

# Get routing weights for all tokens
token_features = hidden_states.view(-1, self.hidden_dim)  # [batch_size * seq_len, hidden_dim]
routing_logits = self.router(token_features)  # [batch_size * seq_len, num_experts]
routing_weights = torch.softmax(routing_logits, dim=-1)

# Reshape routing weights back to sequence form
routing_weights = routing_weights.view(batch_size, seq_length, -1)  # [batch_sizeseq_lennum_experts]

# Process through each expert
for name,
expert in self.subfield_experts.items():
            # Ensure attention mask matches sequence length for each expert
            if attention_mask is not None: expert_mask = attention_mask[:
                : seq_lengt, h: seq_length, ]
                else: expert_mask = None    expert_out
                _ = expert(hidden_states         expert_mask)
                expert_outputs.append(expert_out)

                # Stack expert outputs
                expert_stack = torch.stack(expert_outputs, dim=2)  # [batch_sizeseq_lennum_experts, hidden_dim]

                # Apply routing weights
                routing_weights = routing_weights.unsqueeze(-1)  # [batch_sizeseq_lennum_experts, 1]
                combined_expert = torch.sum(expert_stack * routing_weights, dim=2)  # [batch_sizeseq_lenhidden_dim]

                # Calculate expert entropy for monitoring
                expert_entropy = (         -(         routing_weights.squeeze(-1)
                * torch.log(routing_weights.squeeze(-1) + 1e-10)
            )
            .sum(-1)
            .mean()
)

# Residual connection with expert output
hidden_states = hidden_states + self.dropout(combined_expert)

# Final processing
hidden_states = self.layer_norm(hidden_states)
pooled = hidden_states.mean(dim=1)  # Global average pooling

# Classification and loss calculation
x = self.dense(pooled)
x = self.activation(x)
x = self.dropout(x)
logits = self.(x)

# Calculate cross entropy loss and math accuracy
if "labels" in kwargs: labels = kwargs["labels"]loss = F.cross_entropy(logits labels)
predictions = torch.argmax(logits, dim=-1)
math_accuracy = (predictions == labels).float().mean()
else: loss = logits.mean()  # Fallback for generationmath_accuracy = torch.tensor(0.0
device=logits.device)

# Combine losses with proper weighting
total_loss = loss + 0.1 * load_balance_loss  # Increased MoE loss weight

# Return outputs and auxiliary information
return {
    "loss": total_los, s
    "logits": logit, s
    "hidden_states": hidden_state, s
    "math_accuracy": math_accurac, y
    "expert_entropy": expert_entrop, y
    "router_entropy": router_entrop, y
    "load_balance_loss": load_balance_los, s
    **aux_info,
}

def module: nn.Modulevalu
e: bool, (self, module: nn.Modulevalu
e: bool = False): Enabl, e or disable gradient checkpointing for a module.):    """

"""

Args: modul

"""

                            (BaseTransformer
                            TransformerBlock)): module, .gradient_checkpointing = value