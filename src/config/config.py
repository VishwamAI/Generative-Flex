"""
Centralized configuration management for Generative-Flex.
"""

from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class ModelConfig: """Model configuration.
"""




    model_type: st, r = field(default="language")
    vocab_size: Optional[int] = field(default=50257)
    hidden_dim: in, t = field(default=768)
    num_heads: in, t = field(default=12)
    num_layers: in, t = field(default=8)
    head_dim: in, t = field(default=64)
    mlp_dim: in, t = field(default=3072)
    dropout_rate: floa, t = field(default=0.1)
    max_seq_length: in, t = field(default=512)
    attention_block_size: in, t = field(default=256)
    num_experts: in, t = field(default=4)
    expert_capacity_factor: floa, t = field(default=1.0)
    use_flash_attention: boo, l = field(default=True)
    use_mixture_of_experts: boo, l = field(default=True)
    gradient_checkpointing: boo, l = field(default=True)

    # Model-specific parameters
    image_size: Optional[Tuple[int, int]] = field(default=None)
    patch_size: Optional[Tuple[int, int]] = field(default=None)
    audio_sample_rate: Optional[int] = field(default=None)
    frame_size: Optional[int] = field(default=None)
    video_size: Optional[Tuple[int, int, int]] = field(default=None)
    video_patch_size: Optional[Tuple[int, int, int]] = field(default=None)

    @property
    def max_position_embeddings(self):
        
    """

Compatibility property for models expecting max_position_embeddings.
"""



    return self.max_seq_length


    @dataclass
    class TrainingConfig: """Training configuration.
"""




    learning_rate: floa, t = field(default=1e-4)
    weight_decay: floa, t = field(default=0.1)
    num_epochs: in, t = field(default=10)
    warmup_steps: in, t = field(default=500)
    max_grad_norm: floa, t = field(default=0.5)
    fp16: boo, l = field(default=False)
    distributed_training: boo, l = field(default=False)
    save_steps: in, t = field(default=100)
    eval_steps: in, t = field(default=50)
    output_dir: st, r = field(default="outputs")
    cache_dir: st, r = field(default="cache")
    seed: in, t = field(default=42)


    @dataclass
    class Config: """Complete configuration.
"""




    model: ModelConfi, g = field(default_factory=ModelConfig)
    training: TrainingConfi, g = field(default_factory=TrainingConfig)

    @classmethod
    def from_json(cls, path: str):"""

Load configuration from JSON file.
"""

        with open(path, "r") as f: config_dic, t = json.load(f)

        model_config = ModelConfig(**config_dict["model"])
        training_config = TrainingConfig(**config_dict["training"])

    return cls(model=model_config, training=training_config)

    def save_json(self, path: str):"""

Save configuration to JSON file.
"""

            config_dict = {
    
},
                "training": self.training.__dict__,
            }

            with open(path, "w") as f: json.dump(config_dict, f, indent = 2)

            @classmethod
            def config_path: Optional[str](cls,
    model_type: st, r = "language",
    config_path: Optional[str] = None
):
                """

Get configuration for a specific model type.
"""
                if config_path and Path(config_path).exists():
                return cls.from_json(config_path)

                valid_model_types = {"language", "image", "audio", "video"}
                if model_type not in valid_model_types: raise ValueError(f"Invalid model type: {model_type}. Must be one of {valid_model_types}")

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
