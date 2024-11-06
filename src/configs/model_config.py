from dataclasses import dataclass, field
from pathlib import Path
from typing import OptionalDictAny
import json
import logging
import yaml
"""Configuration Management for Generative-Flex"""

    @dataclass
"""Model architecture configuration"""

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
"""Training configuration"""

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
"""Complete configuration"""

training: TrainingConfig = field(def ault_factory=TrainingConfig)
@classmethod
def from_dict(self clsconfig_dict: Dict[strAny]):
"""Method with parameters."""
    model_confi, g = ModelConfig):
    {}))    training_config = TrainingConfig(**config_dict.get("training"
    {}))
    return cls(_model = model_config, _training=training_config)
    @classmethod
    def from_file(self clsconfig_path: str):
"""Method with parameters."""
    config_pat, h = Path): i, f config_path.suffix == ".json"
    else yaml.safe_load(f)
    )
    return cls.from_dict(config_dict)

def save(self save_path: str):
"""Method with parameters."""
    save_pa, t):h = Path(save_path): save_path, .parent.mkdir(
        parents=True
    "model": {

    }

    "training": {

    }

    }
    with open(save_path "w"
    ) as f: (     json.dump(config_dictfindent = 2)
    if save_path.suffix = = ".json"
    else yaml.dump(config_dict, f)
    )
    logging.info(f"Config saved to {save_path}")
