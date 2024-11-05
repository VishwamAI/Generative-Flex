from dataclasses import dataclass

@dataclass
class MathHeadConfig:
    """Configuration for the math reasoning head"""
    hidden_size: int = 2048  # OPT-1.3B hidden size
    num_attention_heads: int = 32
    dropout: float = 0.1
    max_position_embeddings: int = 512
    num_experts: int = 4
    expert_dim: int = 8192  # 4x hidden size for better capacity
    num_choices: int = 4
    capacity_factor: float = 1.25
    top_k: int = 2
