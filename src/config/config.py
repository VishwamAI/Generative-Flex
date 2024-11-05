import os
from typing import Tuple
"""Centralized configuration management for Generative-Flex."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration."""

    model_type: str  # 'language', 'image', 'audio', 'video'
    vocab_size: Optional[int] = 50257  # For language models
    hidden_dim: int = 768  # Reduced from 1024 for memory efficiency
    num_heads: int = 12  # Reduced from 16 for memory efficiency
    num_layers: int = 8  # Reduced from 12 for memory efficiency
    head_dim: int = 64
    mlp_dim: int = 3072  # Reduced from 4096 for memory efficiency
    dropout_rate: float = 0.1
    max_seq_length: int = 512  # Reduced from 1024 for memory efficiency
    attention_block_size: int = 256  # Reduced from 512 for memory efficiency
    num_experts: int = 4  # Reduced from 8 for memory efficiency
    expert_capacity_factor: float = 1.0  # Reduced from 1.25 for memory efficiency
    use_flash_attention: bool = True
    use_mixture_of_experts: bool = True
    gradient_checkpointing: bool = True

    # Model-specific parameters
    image_size: Optional[Tuple[int, int]] = None  # For image models
    patch_size: Optional[Tuple[int, int]] = None  # For image models
    audio_sample_rate: Optional[int] = None  # For audio models
    frame_size: Optional[int] = None  # For audio models
    video_size: Optional[Tuple[int, int, int]] = None  # For video models
    video_patch_size: Optional[Tuple[int, int, int]] = None  # For video models

    @property
    def hidden_size(self) -> int:
        """Compatibility property for models expecting hidden_size."""
        return self.hidden_dim

    @property
    def num_attention_heads(self) -> int:
        """Compatibility property for models expecting num_attention_heads."""
        return self.num_heads

    @property
    def num_hidden_layers(self) -> int:
        """Compatibility property for models expecting num_hidden_layers."""
        return self.num_layers

    @property
    def intermediate_size(self) -> int:
        """Compatibility property for models expecting intermediate_size."""
        return self.mlp_dim

    @property
    def attention_dropout(self) -> float:
        """Compatibility property for models expecting attention_dropout."""
        return self.dropout_rate

    @property
    def hidden_dropout(self) -> float:
        """Compatibility property for models expecting hidden_dropout."""
        return self.dropout_rate

    @property
    def max_position_embeddings(self) -> int:
        """Compatibility property for models expecting max_position_embeddings."""
        return self.max_seq_length


@dataclass
class TrainingConfig:
    """Training configuration."""

    batch_size: int = 2  # Reduced for larger model
    learning_rate: float = 1e-4  # Increased for better optimization
    weight_decay: float = 0.1  # Increased for better regularization
    num_epochs: int = 10  # Increased for better convergence
    warmup_steps: int = 500  # Increased for better initialization
    max_grad_norm: float = 0.5  # Reduced for stability
    fp16: bool = False  # Disabled for CPU training
    distributed_training: bool = False  # Disabled for single CPU
    save_steps: int = 100  # More frequent checkpoints
    eval_steps: int = 50  # More frequent evaluation
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    seed: int = 42


@dataclass
class Config:
    """Complete configuration."""

    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)

        model_config = ModelConfig(**config_dict["model"])
        training_config = TrainingConfig(**config_dict["training"])

        return cls(_model=model_config, _training=training_config)

    def to_json(self, path: str):
        """Save configuration to JSON file."""
        config_dict = {
            "model": {k: v for k, v in self.model.__dict__.items() if v is not None},
            "training": self.training.__dict__,
        }

        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def get_config(model_type: str, config_path: Optional[str] = None) -> "Config":
        """Get configuration for a specific model type."""
        if config_path and Path(config_path).exists():
            return Config.from_json(config_path)

        valid_model_types = {"language", "image", "audio", "video"}
        if model_type not in valid_model_types:
            raise ValueError(
                f"Invalid model type: {model_type}. Must be one of {valid_model_types}"
            )

        # Default configurations for different model types
        model_config = ModelConfig(model_type=model_type)

        if model_type == "image":
            model_config._image_size = (256, 256)
            model_config._patch_size = (16, 16)
        elif model_type == "audio":
            model_config._audio_sample_rate = 16000
            model_config._frame_size = 1024
        elif model_type == "video":
            model_config._video_size = (16, 256, 256)
            model_config._video_patch_size = (2, 16, 16)

        return Config(_model=model_config, _training=TrainingConfig())
