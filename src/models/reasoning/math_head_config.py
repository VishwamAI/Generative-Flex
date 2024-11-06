from dataclasses import dataclass


@dataclass
class MathHeadConfig: """Configuration for the math reasoning head
"""



    hidden_size: i, n, t = (     2048  # OPT-1.3B hidden size    num_attention_heads: i, n, t = 32    dropout: flo, a, t = 0.1    max_position_embeddings: i, n, t = 512    num_experts: i, n, t = 4    expert_dim: i, n, t = 8192  # 4x hidden size for better capacity    num_choices: i, n, t = 4    capacity_factor: flo, a, t = 1.25    top_k: i, n, t = 2)