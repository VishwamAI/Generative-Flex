import os
"""
Math reasoning module for enhanced transformer model.
"""

import logging

import torch
from transformers import PreTrainedModel, GenerationMixin

from ..layers.flash_moe import FlashAttention, MixtureOfExperts
from ..layers.enhanced_transformer import EnhancedTransformerBlock
from ..multimodal.base_transformer import BaseTransformer, TransformerBlock
from .symbolic_math import SymbolicMathProcessor
from .mathematical_notation import MathematicalNotationProcessor

logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


class MathReasoningHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Use config's hidden dimension
        self.hidden_dim = (
            config.hidden_size if hasattr(config, "hidden_size") else 256
        )  # Match transformer
        self.num_choices = 4  # Default for multiple choice
        self.dropout_prob = (
            config.dropout_rate if hasattr(config, "dropout_rate") else 0.1
        )

        # Enhanced attention with more heads
        self.num_attention_heads = (
            config.num_attention_heads if hasattr(config, "num_attention_heads") else 8
        )
        # Head dimension and sequence length configuration
        self.head_dim = config.head_dim if hasattr(config, "head_dim") else 32
        self.max_seq_length = (
            config.max_position_embeddings
            if hasattr(config, "max_position_embeddings")
            else 512
        )

        # Input dimension handling
        self.input_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )

        # Flash Attention with increased heads
        self.flash_attention = FlashAttention(
            dim=self.hidden_dim,
            num_heads=self.num_attention_heads,
            dropout=self.dropout_prob,
            max_seq_length=self.max_seq_length,
        )

        # Mixture of Experts with increased capacity
        self.math_experts = MixtureOfExperts(
            input_dim=self.hidden_dim,
            expert_dim=(
                config.mlp_dim if hasattr(config, "mlp_dim") else self.hidden_dim * 4
            ),
            num_experts=(config.num_experts if hasattr(config, "num_experts") else 4),
            capacity_factor=(
                config.expert_capacity_factor
                if hasattr(config, "expert_capacity_factor")
                else 1.25
            ),
            dropout=self.dropout_prob,
            k=2,  # Use top-2 routing
        )

        # Symbolic mathematics processor
        self.symbolic_processor = SymbolicMathProcessor(config)
        self.notation_processor = MathematicalNotationProcessor(config)

        # Subfield-specific expert modules
        self.subfield_experts = nn.ModuleDict(
            {
                "algebra": EnhancedTransformerBlock(
                    config=config, dropout=self.dropout_prob
                ),
                "calculus": EnhancedTransformerBlock(
                    config=config, dropout=self.dropout_prob
                ),
                "arithmetic": EnhancedTransformerBlock(
                    config=config, dropout=self.dropout_prob
                ),
                "statistics": EnhancedTransformerBlock(
                    config=config, dropout=self.dropout_prob
                ),
            }
        )

        # Expert routing network
        self.router = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, len(self.subfield_experts)),
            nn.Softmax(dim=-1),
        )

        # Output layers with improved capacity
        self.dense = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim * 4),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, self.num_choices)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        expressions: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Get input dimensions
        batch_size = hidden_states.size(0)
        seq_length = hidden_states.size(1)
        hidden_dim = hidden_states.size(2)

        # Project input to correct dimension
        hidden_states_2d = hidden_states.reshape(-1, hidden_dim)
        hidden_states_projected = self.input_projector(hidden_states_2d)
        hidden_states = hidden_states_projected.reshape(
            batch_size, seq_length, self.hidden_dim
        )

        # Ensure attention mask has correct shape and values
        if attention_mask is not None:
            # Convert mask to correct shape efficiently
            if (
                attention_mask.dim() == 4
                and attention_mask.shape[1] == 1
                and attention_mask.shape[2] == 1
            ):
                # Already in correct shape [batch_size, 1, 1, seq_length]
                pass
            elif attention_mask.dim() == 3 and attention_mask.shape[1] == 1:
                attention_mask = attention_mask.unsqueeze(2)
            elif attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            else:
                # Handle complex cases
                while attention_mask.dim() > 2:
                    attention_mask = attention_mask.squeeze(1)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Ensure proper sequence length efficiently
            if attention_mask.size(-1) != seq_length:
                if attention_mask.size(-1) > seq_length:
                    attention_mask = attention_mask[..., :seq_length]
                else:
                    pad_size = seq_length - attention_mask.size(-1)
                    attention_mask = F.pad(attention_mask, (0, pad_size), value=0)

        # Process with Flash Attention
        try:
            attn_output, attn_weights = self.flash_attention(
                hidden_states, attention_mask
            )
            hidden_states = attn_output
            aux_info = {"attention_weights": attn_weights}
        except Exception as e:
            logger.error(f"Flash attention failed: {e}")
            # Fallback to regular attention if flash attention fails
            hidden_states = hidden_states + 0  # Identity operation as fallback
            aux_info = {"attention_weights": None}
        # Process through MoE layer
        moe_output, router_probs = self.math_experts(hidden_states)
        hidden_states = hidden_states + self.dropout(moe_output)

        # Calculate auxiliary losses
        # Load balancing loss from MoE
        expert_usage = router_probs.mean(dim=0)  # Average usage per expert
        target_usage = torch.ones_like(expert_usage) / expert_usage.size(
            -1
        )  # Uniform distribution
        load_balance_loss = F.kl_div(
            expert_usage.log(), target_usage, reduction="batchmean"
        )

        # Router entropy for monitoring expert specialization
        router_entropy = (
            -(router_probs * torch.log(router_probs + 1e-10)).sum(dim=-1).mean()
        )

        # Process symbolic mathematics if expressions are provided
        if expressions is not None:
            hidden_states = self.symbolic_processor(hidden_states, expressions)

        # Route through enhanced subfield-specific experts
        expert_outputs = []
        #         expert_weights = []  # TODO: Remove or use this variable

        # Get routing weights for all tokens
        token_features = hidden_states.view(
            -1, self.hidden_dim
        )  # [batch_size * seq_len, hidden_dim]
        routing_logits = self.router(
            token_features
        )  # [batch_size * seq_len, num_experts]
        routing_weights = torch.softmax(routing_logits, dim=-1)

        # Reshape routing weights back to sequence form
        routing_weights = routing_weights.view(
            batch_size, seq_length, -1
        )  # [batch_size, seq_len, num_experts]

        # Process through each expert
        for name, expert in self.subfield_experts.items():
            # Ensure attention mask matches sequence length for each expert
            if attention_mask is not None:
                expert_mask = attention_mask[:, :seq_length, :seq_length]
            else:
                expert_mask = None
            expert_out, aux_info = expert(
                hidden_states, expert_mask
            )  # Get both outputs and auxiliary info
            expert_outputs.append(expert_out)

        # Stack expert outputs
        expert_stack = torch.stack(
            expert_outputs, dim=2
        )  # [batch_size, seq_len, num_experts, hidden_dim]

        # Apply routing weights
        routing_weights = routing_weights.unsqueeze(
            -1
        )  # [batch_size, seq_len, num_experts, 1]
        combined_expert = torch.sum(
            expert_stack * routing_weights, dim=2
        )  # [batch_size, seq_len, hidden_dim]

        # Calculate expert entropy for monitoring
        _expert_entropy = (
            -(
                routing_weights.squeeze(-1)
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
        logits = self.classifier(x)

        # Calculate cross entropy loss and math accuracy
        if "labels" in kwargs:
            labels = kwargs["labels"]
            loss = F.cross_entropy(logits, labels)
            predictions = torch.argmax(logits, dim=-1)
            _math_accuracy = (predictions == labels).float().mean()
        else:
            loss = logits.mean()  # Fallback for generation
            _math_accuracy = torch.tensor(0.0, device=logits.device)

        # Combine losses with proper weighting
        total_loss = loss + 0.1 * load_balance_loss  # Increased MoE loss weight

        # Return outputs and auxiliary information
        outputs = {
            "logits": self.classifier(hidden_states),
            "hidden_states": hidden_states,
            "attention_weights": aux_info.get("attention_weights", None),
            "router_entropy": router_entropy,
            "load_balance_loss": load_balance_loss,
            "expert_outputs": expert_outputs,
            "routing_weights": routing_weights,
        }

        if "labels" in kwargs:
            outputs["loss"] = total_loss

        return outputs

        return outputs


from transformers import PreTrainedModel


class MathReasoningModel(PreTrainedModel, GenerationMixin):
    _supports_gradient_checkpointing = (
        True  # Class attribute required by PreTrainedModel
    )

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Initialize base transformer
        self.transformer = BaseTransformer(config)

        # Initialize math reasoning head
        self.math_head = MathReasoningHead(config)

        # Initialize symbolic math processor
        self.symbolic_processor = SymbolicMathProcessor(config)

        # Set up dropout
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))

        # Initialize weights
        self._init_weights()

        # Enable gradient checkpointing if specified
        if getattr(config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (BaseTransformer, TransformerBlock)):
            module.gradient_checkpointing = value

    def gradient_checkpointing_enable(self):
        """Enables gradient checkpointing for memory efficiency"""
        self.apply(lambda module: self._set_gradient_checkpointing(module, True))

    def _init_weights(self):
        # Initialize transformer weights
        for module in self.transformer.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        # Initialize reasoning head and symbolic processor weights
        for module in [self.math_head, self.symbolic_processor]:
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    submodule.weight.data.normal_(mean=0.0, std=0.02)
                    if submodule.bias is not None:
                        submodule.bias.data.zero_()

    def process_mathematical_expression(self, expression: str) -> str:
        """Process mathematical expressions with error handling."""
        try:
            # Apply symbolic processing
            processed = self.symbolic_processor.process(expression)
            return processed
        except Exception as e:
            logger.error(f"Error processing expression: {e}")
            return expression

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        expressions: Optional[List[str]] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Get transformer outputs
        hidden_states = self.transformer(input_ids, attention_mask, **kwargs)

        # Process mathematical expressions
        processed_math = self.process_mathematical_expression(hidden_states)

        # Combine with original hidden states
        enhanced_states = hidden_states + processed_math

        # Process through math reasoning head with labels
        math_outputs = self.math_head(
            enhanced_states,
            attention_mask,
            expressions,
            labels=labels,
            **kwargs,
        )

        # Calculate total loss with proper weighting
        outputs = {
            "logits": math_outputs["logits"],
            "hidden_states": enhanced_states,
            "attention_weights": math_outputs.get("attention_weights", None),
            "expert_outputs": math_outputs.get("expert_outputs", None),
            "routing_weights": math_outputs.get("routing_weights", None),
        }

        if labels is not None:
            outputs["loss"] = math_outputs["loss"]
            outputs["math_accuracy"] = math_outputs.get("math_accuracy", None)
            outputs["moe_loss"] = math_outputs.get("moe_loss", None)
            outputs["router_entropy"] = math_outputs.get("router_entropy", None)

        return outputs

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        """Prepare inputs for generation."""
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        if kwargs:
            inputs.update(kwargs)

        return inputs

    @staticmethod
    def create_attention_mask(input_ids, padding_idx=0):
        """Create attention mask from input_ids."""
        # Get input dimensions
        batch_size, seq_length = input_ids.size()
        device = input_ids.device

        # Create causal mask
        attention_mask = torch.ones(
            (seq_length, seq_length), dtype=torch.float32, device=device
        )
        attention_mask = torch.triu(attention_mask)

        # Create padding mask
        padding_mask = (input_ids != padding_idx).float()
        attention_mask = attention_mask * padding_mask.unsqueeze(-1)

        # Expand to batch size
        attention_mask = attention_mask.expand(batch_size, seq_length, seq_length)

        return attention_mask
