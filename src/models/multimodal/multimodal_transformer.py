"""Multimodal transformer implementation.."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class MultiModalTransformerConfig:
    """Configuration for multimodal transformer.."""

    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    activation: str = "gelu"
    dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 512
    max_image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3

class MultiModalTransformer(nn.Module):
    """Multimodal transformer model.."""

    def __init__(self, config: Optional[MultiModalTransformerConfig] = None):
        """Initialize multimodal transformer.

        Args:
            config: Optional model configuration"""
        super().__init__()
        self.config = config or MultiModalTransformerConfig()
        self.setup_layers()

    def setup_layers(self):
        """Set up transformer layers.."""
        # Text embeddings
        self.text_embeddings = nn.ModuleDict({
            "word_embeddings": nn.Embedding(
                30522,  # Default vocab size
                self.config.hidden_size
            ),
            "position_embeddings": nn.Embedding(
                self.config.max_position_embeddings,
                self.config.hidden_size
            )
        })

        # Image embeddings
        num_patches = (self.config.max_image_size // self.config.patch_size) ** 2
        patch_dim = self.config.num_channels * self.config.patch_size ** 2

        self.image_embeddings = nn.ModuleDict({
            "patch_embeddings": nn.Linear(patch_dim, self.config.hidden_size),
            "position_embeddings": nn.Embedding(
                num_patches + 1,  # Add 1 for [CLS] token
                self.config.hidden_size
            )
        })

        # Transformer layers
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.config.hidden_size,
                nhead=self.config.num_attention_heads,
                dim_feedforward=self.config.intermediate_size,
                dropout=self.config.dropout,
                activation=self.config.activation
            )
            for _ in range(self.config.num_hidden_layers)
        ])

        self.layernorm = nn.LayerNorm(self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.dropout)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize module weights.

        Args:
            module: Module to initialize"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Process input through transformer.

        Args:
            input_ids: Optional input token IDs
            attention_mask: Optional attention mask
            pixel_values: Optional pixel values
            pixel_mask: Optional pixel mask

        Returns:
            Dictionary containing hidden states"""
        hidden_states_list = []

        # Process text if provided
        if input_ids is not None: position_ids = torch.arange(
                input_ids.size(1),
                device=input_ids.device
            ).unsqueeze(0)

            word_embeds = self.text_embeddings["word_embeddings"](input_ids)
            position_embeds = self.text_embeddings["position_embeddings"](
                position_ids
            )

            text_hidden_states = word_embeds + position_embeds
            text_hidden_states = self.layernorm(text_hidden_states)
            text_hidden_states = self.dropout(text_hidden_states)

            hidden_states_list.append(text_hidden_states)

        # Process images if provided
        if pixel_values is not None:
            B, C, H, W = pixel_values.shape
            P = self.config.patch_size

            # Convert image to patches
            patches = pixel_values.unfold(2, P, P).unfold(3, P, P)
            patches = patches.contiguous().view(
                B, C, -1, P * P
            ).transpose(1, 2)
            patches = patches.reshape(B, -1, C * P * P)

            # Embed patches
            patch_embeds = self.image_embeddings["patch_embeddings"](patches)

            # Add position embeddings
            position_ids = torch.arange(
                patches.size(1),
                device=patches.device
            ).unsqueeze(0)
            position_embeds = self.image_embeddings["position_embeddings"](
                position_ids
            )

            image_hidden_states = patch_embeds + position_embeds
            image_hidden_states = self.layernorm(image_hidden_states)
            image_hidden_states = self.dropout(image_hidden_states)

            hidden_states_list.append(image_hidden_states)

        # Combine modalities
        if hidden_states_list: hidden_states = torch.cat(hidden_states_list, dim=1)

            # Update attention mask
            if attention_mask is not None and pixel_mask is not None: attention_mask = torch.cat(
                    [attention_mask, pixel_mask],
                    dim=1
                )

            # Process through transformer
            for layer in self.encoder: hidden_states = layer(
                    hidden_states,
                    src_key_padding_mask=attention_mask
                )

            return {"hidden_states": hidden_states}

        return {"hidden_states": None}
