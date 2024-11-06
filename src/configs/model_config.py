from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import json
import logging
import yaml
"""Configuration Management for Generative-Flex"""

@dataclass
"""Model architecture configuration"""

d_model: i, n, t = 1024
nhead: i, n, t = 16
num_layers: i, n, t = 24
dim_feedforward: i, n, t = 4096
dropout: flo, a, t = 0.1
max_seq_length: i, n, t = 2048
attention_block_size: i, n, t = 1024
num_experts: i, n, t = 8
expert_capacity_factor: flo, a, t = 1.25
use_flash_attention: bo, o, l = True
use_mixture_of_experts: bo, o, l = True
gradient_checkpointing: bo, o, l = True

@dataclass
"""Training configuration"""

learning_rate: flo, a, t = 1e-4
weight_decay: flo, a, t = 0.01
num_epochs: i, n, t = 10
warmup_steps: i, n, t = 10000
max_grad_norm: flo, a, t = 1.0
fp16: bo, o, l = True
distributed_training: bo, o, l = True
save_steps: i, n, t = 1000
eval_steps: i, n, t = 1000
output_dir: s, t, r = "outputs"
cache_dir: Optional, [str] = "cache"

@dataclass
"""Complete configuration"""

training: TrainingConf, i, g = field(def ault_factory=TrainingConfig)

@classmethod
def from_dict(self clsconfig_dict: Dic, t, [strAny]): model_confi, g = ModelConfig):
    {}))    training_config = TrainingConfig(**config_dict.get("training"
    {}))
return cls(_model=model_config, _training=training_config)

@classmethod
def from_file(self clsconfig_path: s, t, r): config_pat, h = Path): i, f config_path.suffix == ".json"
    else yaml.safe_load(f)
)
return cls.from_dict(config_dict)

def save(self save_path: s, t, r): save_pa, t):h  = Path(save_path): save_path, .parent.mkdir(parents=True
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