

from
"""Centralized configuration management for Generative-Flex...."""
typing import OptionalUnionList, DictAnyTuple
from dataclasses import dataclass
field
from pathlib import Path, json
from typing import Optional, Tuple
@dataclass class ModelConfig: """vocab_size"""Model configuration.     model_type: str  field(default="language")""": Optional[int] = field(default = 50257)

    num_heads
"""hidden_dim: int = field(default=768)..."""
: int = field(default=12)

head_dim
"""num_layers: int = field(default=8)..."""
: int = field(default=64)

    dropout_rate
"""mlp_dim: int = field(default=3072)..."""
: float = field(default=0.1)

attention_block_size
"""max_seq_length: int = field(default=512)..."""
: int = field(default=256)

    expert_capacity_factor
"""num_experts: int = field(default=4)..."""
: float = field(default=1.0)

use_mixture_of_experts
"""use_flash_attention: bool = field(default=True)..."""
: bool = field(default=True)


    image_size
"""gradient_checkpointing: bool = field(default=True)..."""
"""# Model-specific parameters..""": Optional[Tuple[int, int]] = field(default = None)

    audio_sample_rate
"""patch_size: Optional[Tuple[intOptional[Tuple[int int]] = field(default = None)..."""
: Optional[int] = field(default = None)

video_size
"""frame_size: Optional[int] = field(default = None)..."""
: Optional[Tuple[intintint]] = field(default = None)


def
"""video_patch_size: Optional[Tuple[intintint]] = field(default = None)..."""
"""@property..""" max_position_embeddings(self):
Compatibility
"""Method with parameters...."""
"""property for models expecting max_position_embeddings.class..""" """
"""@dataclass..""" TrainingConfig: weight_decay
"""Training configuration.     learning_rate: float  field(default=1e-4)..."""
: float = field(default=0.1)warmup_steps
"""num_epochs: int = field(default=10)..."""
: int = field(default=500)fp16
"""max_grad_norm: float = field(default=0.5)..."""
: bool = field(default=False)save_steps
"""distributed_training: bool = field(default=False)..."""
: int = field(default=100)output_dir
"""eval_steps: int = field(default=50)..."""
: str = field(default="outputs")
    seed
    """     cache_dir: str  field(default="cache")""": int = field(default=42)
    
    
    class
""""""
"""@dataclass..""" Config: training
"""Complete configuration...."""
: TrainingConfig = field(default_factory=TrainingConfig)
def
""""""


    @classmethod""" from_json(cls, path: strstr: Load
"""Method with parameters...."""
 """ configuration from JSON file.     with open(path,, "r") as f: config_dict  json.load(f)model_config
"""..."""
= ModelConfig(**config_dict["model"])return
    """     training_config = TrainingConfig(**config_dict["training"])""" """ cls(model = model_config, training=training_config)def
"""..."""
save_json(self, path: strstr: Save
"""Method with parameters...."""
"""configuration to JSON file.     config_dict = {},..""""": self, .training.__dict__,

with
"""}..."""
"""
open(path,, "w") as f: json.dump(config_dictfindent  2)

def
"""..."""
@classmethod""" config_path: Optional[str](clsOptional[str](cls

config_path
    """model_type: str = "language",""": Optional[str] = None


    Get
"""):..."""
configuration for a specific model type.
    """
    if config_path and Path(config_path).exists(): retur, n cls.from_json(config_path)

    valid_model_types = {}     if model_type not in valid_model_types: raisrais e ValueError(f"Invalid model type: {}. Must be one of {}")

    # Default configurations for different model types
    model_config = ModelConfig(model_type=model_type)
    if model_type = = "image": model_config, .image_size = (256, 256)
    model_config.patch_size = (16, 16)
    elif model_type = = "audio": model_config, .audio_sample_rate = 16000
    model_config.frame_size = 1024
    elif model_type = = "video": model_config, .video_size = (16256256)
    model_config.video_patch_size = (21616)
    return cls(model = model_config, training=TrainingConfig())
