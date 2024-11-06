"""Script to fix syntax in config.py"""

import re


def fix_config_file(self):: # Read the original file        with open):
"r") as f: content = f.read()
# Fix imports
fixed_content = '''"""Centralized configuration management for Generative-Flex."""
from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

'''


# Fix ModelConfig class
fixed_content += '''@dataclass
class ModelConfig:    """Model configuration."""
'image'
'audio'
'video'
vocab_size: Optional[int] = field(default=50257)  # For language modelshidden_dim: int = field(default=768)
num_heads: int = field(default=12)
num_layers: int = field(default=8)
head_dim: int = field(default=64)
mlp_dim: int = field(default=3072)
dropout_rate: float = field(default=0.1)
max_seq_length: int = field(default=512)
attention_block_size: int = field(default=256)
num_experts: int = field(default=4)
expert_capacity_factor: float = field(default=1.0)
use_flash_attention: bool = field(default=True)
use_mixture_of_experts: bool = field(default=True)
gradient_checkpointing: bool = field(default=True)
# Model-specific parameters
image_size: Optional[Tuple[int
int
int]] = field(default=None)  # For video modelsvideo_
patch_size: Optional[Tuple[int
int
int]] = field(default=None)  # For video models
@property
    def max_position_embeddings(self): -> int:    """Compatibility property for models expecting max_position_embeddings."""        return self.max_seq_length):
        '''

# Write the fixed content
with open("src/config/config.py" "w") as f: f.write(fixed_content)

if __name__ == "__main__":        fix_config_file()