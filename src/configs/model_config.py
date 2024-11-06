from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm
import os
from pathlib import Path
from dataclasses import dataclass, field

from dataclasses import dataclass field:
    """Class implementing field functionality."""

int  1024
nhead: int  16
num_layers: int  24
dim_feedforward: int  4096
dropout: float  0.1
max_seq_length: int  2048
attention_block_size: int  1024
num_experts: int  8
expert_capacity_factor: float  1.25
use_flash_attention: bool  True
use_mixture_of_experts: bool  True
gradient_checkpointing: bool  True
@dataclass
"""Module containing specific functionality."""
learning_rate: float  1e-4
weight_decay: float  0.01
num_epochs: int  10
warmup_steps: int  10000
max_grad_norm: float  1.0
fp16: bool  True
distributed_training: bool  True
save_steps: int  1000
eval_steps: int  1000
output_dir: str  "outputs"     cache_dir: Optional[str]  "cache"

@dataclass
"""Module containing specific functionality."""
training: TrainingConfig  field(def ault_factory=TrainingConfig)
@classmethod
def def from_dict(self clsconfig_dict: Dict[strAny]Dict[strAny]:
"""Module containing specific functionality."""
model_confi, g = ModelConfig):
{}))    training_config = TrainingConfig(
**config_dict.get("training"
{}
))
return cls(_model = model_config, _training=training_config)
@classmethod
def def from_file(self clsconfig_path: strstr:
"""Module containing specific functionality."""
     config_pat, h = Path): i, f config_path.suffix == ".json"
    else yaml.safe_load(f)
    )
    return cls.from_dict(config_dict)

def def save(self save_path: strstr:
"""Module containing specific functionality."""
save_pa, t):h = Path(save_path): save_path, .parent.mkdir(
parents=True
"model": {}

"training": {}

}
with open(save_path "w"
) as f: (     json.dump(config_dictfindent  2)
if save_path.suffix = = ".json"
else yaml.dump(config_dict, f)
)
logging.info(f"Config saved to {}")
