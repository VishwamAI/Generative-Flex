"""Configuration Management for Generative-Flex"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json
from pathlib import Path
import yaml
import logging


@dataclass
class ModelConfig:
    """Model architecture configuration"""

    vocab_size: int = 50257
    d_model: int = 1024
    nhead: int = 16
    num_layers: int = 24
    dim_feedforward: int = 4096
    dropout: float = 0.1
    max_seq_length: int = 2048
    attention_block_size: int = 1024
    num_experts: int = 8
    expert_capacity_factor: float = 1.25
    use_flash_attention: bool = True
    use_mixture_of_experts: bool = True
    gradient_checkpointing: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""

    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 10000
    max_grad_norm: float = 1.0
    fp16: bool = True
    distributed_training: bool = True
    save_steps: int = 1000
    eval_steps: int = 1000
    output_dir: str = "outputs"
    cache_dir: Optional[str] = "cache"


@dataclass
class GenerativeFlexConfig:
    """Complete configuration"""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GenerativeFlexConfig":
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        return cls(model=model_config, training=training_config)

    @classmethod
    def from_file(cls, config_path: str) -> "GenerativeFlexConfig":
        config_path = Path(config_path)
        with open(config_path) as f:
            config_dict = (
                json.load(f) if config_path.suffix == ".json" else yaml.safe_load(f)
            )
        return cls.from_dict(config_dict)

    def save(self, save_path: str):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = {
            "model": {k: v for k, v in vars(self.model).items()},
            "training": {k: v for k, v in vars(self.training).items()},
        }
        with open(save_path, "w") as f:
            (
                json.dump(config_dict, f, indent=2)
                if save_path.suffix == ".json"
                else yaml.dump(config_dict, f)
            )
        logging.info(f"Config saved to {save_path}")


def create_default_config() -> GenerativeFlexConfig:
    """Create default configuration"""
    return GenerativeFlexConfig()
