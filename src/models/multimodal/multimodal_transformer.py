"""
MultiModal Transformer implementation with features inspired by Gemma and LLaMA.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from ..layers.flash_moe import EnhancedTransformerBlock
from ..enhanced_transformer import EnhancedTransformer
from .image_processor import ImageProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalTransformer(nn.Module):
    """
    MultiModal Transformer with enhanced capabilities for mathematical reasoning.
    Incorporates features from Gemma and LLaMA architectures.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Text embedding components
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Image processing components
        self.image_processor = ImageProcessor(config.hidden_size)

        # Image feature projection
        self.image_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
        )

        # Enhanced transformer blocks with expert routing
        self.transformer_blocks = nn.ModuleList(
            [EnhancedTransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )

        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=getattr(
                config, "dropout", 0.1
            ),  # Use OPT's dropout or default to 0.1
        )

        # Output components
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(
            getattr(config, "dropout", 0.1)
        )  # Use OPT's dropout or default to 0.1

        # Mathematical reasoning specific components
        self.math_gate = nn.Linear(config.hidden_size, 1)
        self.math_transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
        )

        self.init_weights()

    def init_weights(self):
        """Initialize weights with specific initialization for mathematical operations."""

        def _init_math_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        self.apply(_init_math_weights)

    def _get_position_embeddings(self, position_ids, seq_length):
        """Get position embeddings with support for relative positions."""
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=self.word_embeddings.weight.device
            )
            position_ids = position_ids.unsqueeze(0)
        return self.position_embeddings(position_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with support for text and image inputs.
        """
        batch_size = (
            input_ids.size(0) if input_ids is not None else image_features.size(0)
        )
        device = next(self.parameters()).device
        embeddings = None
        total_sequence_length = 0

        # Process text inputs
        if input_ids is not None:
            text_embeddings = self.word_embeddings(
                input_ids
            )  # [batch_size, seq_len, hidden_size]
            total_sequence_length += text_embeddings.size(1)
            embeddings = text_embeddings

        # Process image inputs
        if image_features is not None:
            try:
                # Process images through ImageProcessor
                processed_images = self.image_processor(
                    image_features
                )  # [batch_size, num_images, hidden_size]

                # Project image features
                image_embeddings = self.image_projection(
                    processed_images
                )  # [batch_size, num_images, hidden_size]

                total_sequence_length += image_embeddings.size(1)

                if embeddings is not None:
                    # Combine text and image embeddings along sequence dimension
                    embeddings = torch.cat([embeddings, image_embeddings], dim=1)
                else:
                    embeddings = image_embeddings

            except Exception as e:
                logger.error(f"Error processing images in transformer: {str(e)}")
                if embeddings is None:
                    embeddings = torch.zeros(
                        batch_size, 1, self.config.hidden_size, device=device
                    )
                    total_sequence_length += 1

        # Add position embeddings
        position_ids = torch.arange(
            total_sequence_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)

        # Add token type embeddings (0 for text, 1 for image)
        token_type_ids = torch.zeros(
            (batch_size, total_sequence_length), dtype=torch.long, device=device
        )
        if input_ids is not None and image_features is not None:
            token_type_ids[:, input_ids.size(1) :] = 1
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # Combine all embeddings
        embeddings = embeddings + position_embeddings + token_type_embeddings
        embeddings = self.dropout(embeddings)
        hidden_states = embeddings

        # Apply transformer blocks
        router_probs_list = []
        for block in self.transformer_blocks:
            hidden_states, router_probs = block(hidden_states, attention_mask)
            router_probs_list.append(router_probs)

        # Apply mathematical reasoning enhancement
        math_gate = torch.sigmoid(self.math_gate(hidden_states))
        math_hidden = self.math_transform(hidden_states)
        hidden_states = math_gate * math_hidden + (1 - math_gate) * hidden_states

        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)

        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "pooler_output": hidden_states[:, 0],  # Use first token for pooling
                "math_gate": math_gate,
                "router_probs": router_probs_list,
            }
        return hidden_states

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare inputs for text generation."""
        position_ids = kwargs.get("position_ids", None)
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "image_features": kwargs.get("image_features", None),
        }
