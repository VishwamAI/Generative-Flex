"""Configuration for mathematical reasoning head."""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class MathHeadConfig:
    """Configuration for mathematical reasoning head."""

    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    num_experts: int = 8
    num_math_tokens: int = 1000
