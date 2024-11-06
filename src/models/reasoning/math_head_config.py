from dataclasses import dataclass


@dataclass
class MathHeadConfig: """Configuration for the math reasoning head
"""



    hidden_size: in, t = (     2048  # OPT-1.3B hidden size    num_attention_heads: in, t = 32    dropout: floa, t = 0.1    max_position_embeddings: in, t = 512    num_experts: in, t = 4    expert_dim: in, t = 8192  # 4x hidden size for better capacity    num_choices: in, t = 4    capacity_factor: floa, t = 1.25    top_k: in, t = 2)