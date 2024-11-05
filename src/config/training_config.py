import os
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class TrainingConfig:
    model_name: str = "facebook/opt-125m"
    subjects: List[str] = None
    batch_size: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 5
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    device: str = "cuda"  # Explicitly set device
    fp16: bool = True  # Enable mixed precision

    # Model architecture parameters
    hidden_size: int = 256
    num_attention_heads: int = 8
    num_hidden_layers: int = 6
    intermediate_size: int = 1024
    max_position_embeddings: int = 512
    num_experts: int = 4
    expert_capacity_factor: float = 1.25

    # Training optimization parameters
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    eval_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 20

    generation_config: Dict = None

    def __post_init__(self):
        if self.subjects is None:
            self._subjects = ["Math", "Computer_Science"]

        if self.generation_config is None:
            self._generation_config = {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_length": 512,
            }
