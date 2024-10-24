"""Configuration for model training."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Base configuration for all models."""

    model_type: str  # 'language', 'image', 'audio', 'video'
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout_rate: float = 0.1
    max_sequence_length: int = 1024
    vocab_size: Optional[int] = None  # For language models
    image_size: Optional[tuple] = None  # For image models
    audio_sample_rate: Optional[int] = None  # For audio models
    video_frames: Optional[int] = None  # For video models


@dataclass
class TrainingConfig:
    """Training configuration."""

    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0
    checkpoint_dir: str = "checkpoints"
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 1000
    save_every_n_steps: int = 5000


@dataclass
class DataConfig:
    """Data configuration."""

    data_dir: str = "data"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    shuffle_buffer_size: int = 10000
    prefetch_size: int = 2


def get_default_config(model_type: str) -> Dict[str, Any]:
    """Get default configuration for a specific model type."""
    base_config = {
        "model": ModelConfig(model_type=model_type),
        "training": TrainingConfig(),
        "data": DataConfig(),
    }

    # Model-specific configurations
    if model_type == "language":
        base_config["model"].vocab_size = 50257  # GPT-2 vocabulary size
    elif model_type == "image":
        base_config["model"].image_size = (256, 256)
    elif model_type == "audio":
        base_config["model"].audio_sample_rate = 16000
    elif model_type == "video":
        base_config["model"].video_frames = 16

    return base_config
