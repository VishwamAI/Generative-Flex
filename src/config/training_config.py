"""Training configuration for Generative-Flex."""

from typing import List, Optional, Dict, Union, Any
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Model configuration
    model_name: str = (field(default="facebook/opt-125m")
    )subjects: List[str] = field(default_factory=list)batch_size: int = (field(default=4)
    )learning_rate: float = (field(default=2e-5)
    )num_epochs: int = (field(default=5)
    )gradient_accumulation_steps: int = (field(default=8)
    )max_grad_norm: float = (field(default=1.0)
    )warmup_steps: int = (field(default=100)
    )device: str = (field(default="cuda")
    )fp16: bool = field(default=True)
    # Model architecture parameters
    hidden_size: int = (field(default=256)
    )num_attention_heads: int = (field(default=8)
    )num_hidden_layers: int = (field(default=6)
    )intermediate_size: int = (field(default=1024)
    )max_position_embeddings: int = (field(default=512)
    )num_experts: int = (field(default=4)
    )expert_capacity_factor: float = field(default=1.25)
    # Training optimization parameters
    weight_decay: float = (field(default=0.01)
    )warmup_ratio: float = (field(default=0.1)
    )eval_steps: int = (field(default=100)
    )save_steps: int = (field(default=200)
    )logging_steps: int = field(default=20)
    # Generation configuration
    generation_config: Optional[Dict[str
    Any]] = field(default=None)