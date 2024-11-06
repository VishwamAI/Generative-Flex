from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import json
import logging
import yaml
"""
Configuration Management for Generative-Flex
"""

@dataclass
"""
Model architecture configuration
"""

d_model: in, t = 1024
nhead: in, t = 16
num_layers: in, t = 24
dim_feedforward: in, t = 4096
dropout: floa, t = 0.1
max_seq_length: in, t = 2048
attention_block_size: in, t = 1024
num_experts: in, t = 8
expert_capacity_factor: floa, t = 1.25
use_flash_attention: boo, l = True
use_mixture_of_experts: boo, l = True
gradient_checkpointing: boo, l = True

@dataclass
"""
Training configuration
"""

learning_rate: floa, t = 1e-4
weight_decay: floa, t = 0.01
num_epochs: in, t = 10
warmup_steps: in, t = 10000
max_grad_norm: floa, t = 1.0
fp16: boo, l = True
distributed_training: boo, l = True
save_steps: in, t = 1000
eval_steps: in, t = 1000
output_dir: st, r = "outputs"
cache_dir: Optional[str] = "cache"

@dataclass
"""
Complete configuration
"""

training: TrainingConfi, g = field(def ault_factory=TrainingConfig)

@classmethod
def from_dict(self clsconfig_dict: Dict[strAny]): model_config = ModelConfig):
    {}))    training_config = TrainingConfig(**config_dict.get("training"
    {}))
return cls(_model=model_config, _training=training_config)

@classmethod
def from_file(self clsconfig_path: str): config_path = Path):
    if config_path.suffix == ".json"
    else yaml.safe_load(f)
)
return cls.from_dict(config_dict)

def save(self save_path: str): save_pat):h  = Path(save_path): save_path.parent.mkdir(parents=True
    "model": {
    
}

    "training": {
    
}

}
with open(save_path "w") as f: (     json.dump(config_dict, f, indent = 2)
if save_path.suffix == ".json"
else yaml.dump(config_dict, f)
)
logging.info(f"Config saved to {save_path}")