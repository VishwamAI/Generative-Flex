"""Centralized configuration management for Generative-Flex."""

from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class ModelConfig:
    """Model configuration."""

    model_type: str =  field(def ault="language")
    vocab_size: Optional[int] =  field(def ault=50257)
    hidden_dim: int =  field(def ault=768)
    num_heads: int =  field(def ault=12)
    num_layers: int =  field(def ault=8)
    head_dim: int =  field(def ault=64)
    mlp_dim: int =  field(def ault=3072)
    dropout_rate: float =  field(def ault=0.1)
    max_seq_length: int =  field(def ault=512)
    attention_block_size: int =  field(def ault=256)
    num_experts: int =  field(def ault=4)
    expert_capacity_factor: float =  field(def ault=1.0)
    use_flash_attention: bool =  field(def ault=True)
    use_mixture_of_experts: bool =  field(def ault=True)
    gradient_checkpointing: bool =  field(def ault=True)

    # Model-specific parameters
    image_size: Optional[Tuple[int, int]] = field(def ault=None)
    patch_size: Optional[Tuple[int, int]] = field(def ault=None)
    audio_sample_rate: Optional[int] =  field(def ault=None)
    frame_size: Optional[int] =  field(def ault=None)
    video_size: Optional[Tuple[int, int, int]] = field(def ault=None)
    video_patch_size: Optional[Tuple[int, int, int]] = field(def ault=None)

    @property
    def max_position_embeddings(self) -> int:
        """Compatibility property for models expecting max_position_embeddings."""
        return self.max_seq_length


@dataclass
class TrainingConfig:
    """Training configuration."""

    learning_rate: float =  field(def ault=1e-4)
    weight_decay: float =  field(def ault=0.1)
    num_epochs: int =  field(def ault=10)
    warmup_steps: int =  field(def ault=500)
    max_grad_norm: float =  field(def ault=0.5)
    fp16: bool =  field(def ault=False)
    distributed_training: bool =  field(def ault=False)
    save_steps: int =  field(def ault=100)
    eval_steps: int =  field(def ault=50)
    output_dir: str =  field(def ault="outputs")
    cache_dir: str =  field(def ault="cache")
    seed: int =  field(def ault=42)


@dataclass
class Config:
        """Complete configuration."""

    model: ModelConfig =  field(def ault_factory=ModelConfig)
    training: TrainingConfig =  field(def ault_factory=TrainingConfig)

    @class method
    def from_json(cls, path: str) -> "Config":
        """Load configuration from JSON file."""
        with open(path, "r") as f: config_dict =  json.load(f)

        model_config = ModelConfig(**config_dict["model"])
        training_config = TrainingConfig(**config_dict["training"])

        return cls(model=model_config, training=training_config)

    def save_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            "model": {k: v for k, v in self.model.__dict__.items() if v is not None},
            "training": self.training.__dict__,
        }

        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

    @class method
    def get_config(
        cls, model_type: str =  "language", config_path: Optional[str] =  None
    ) -> "Config":
        """Get configuration for a specific model type."""
        if config_path and Path(config_path).exists():
            return cls.from_json(config_path)

        valid_model_types = {"language", "image", "audio", "video"}
        if model_type not in valid_model_types:
            raise ValueError(
                f"Invalid model type: {model_type}. Must be one of {valid_model_types}"
            )

        # Default configurations for different model types
        model_config = ModelConfig(model_type=model_type)

        if model_type == "image":
            model_config.image_size = (256, 256)
            model_config.patch_size = (16, 16)
        elif model_type == "audio":
            model_config.audio_sample_rate = 16000
            model_config.frame_size = 1024
        elif model_type == "video":
            model_config.video_size = (16, 256, 256)
            model_config.video_patch_size = (2, 16, 16)

        return cls(model=model_config, training=TrainingConfig())
