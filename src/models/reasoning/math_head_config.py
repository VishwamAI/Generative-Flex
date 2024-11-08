"""Math head configuration."""
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class MathHeadConfig:
    """Configuration for math reasoning head."""

    model_dim: int = field(default=512)
    num_experts: int = field(default=8)
    expert_size: int = field(default=128)
    dropout_rate: float = field(default=0.1)
    use_bias: bool = field(default=True)
    activation: str = field(default="gelu")

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.expert_size <= 0:
            raise ValueError("expert_size must be positive")
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
