import re

def fix_math_config():
    # Create proper dataclass structure
    new_content = '''"""Math configuration module."""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import torch


@dataclass
class MathConfig:
    """Configuration for math reasoning module."""
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_experts: int = 4
    expert_hidden_size: int = 1024
    dropout_rate: float = 0.1
    activation_fn: str = "gelu"
    layer_norm_eps: float = 1e-12
    use_cache: bool = True
    output_attentions: bool = False
    output_hidden_states: bool = False
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    vocab_size: int = 50257
    initializer_range: float = 0.02
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    expert_capacity: int = 64
    expert_dropout: float = 0.1
    expert_router_type: str = "top_2"
    router_z_loss_coef: float = 0.01
    router_aux_loss_coef: float = 0.01
    jitter_noise: float = 0.1
    use_expert_choice: bool = True
    num_symbolic_rules: int = 100
    max_rule_depth: int = 5
    use_rule_embeddings: bool = True
    rule_embedding_dim: int = 256
'''

    # Write the new content
    with open('src/models/reasoning/math_config.py', 'w') as f:
        f.write(new_content)

if __name__ == '__main__':
    fix_math_config()
