"""Configuration for mathematical reasoning module.."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

@dataclass
class MathConfig:
    """Configuration for mathematical reasoning.."""

    model_type: str = "math_reasoning"
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    max_position_embeddings: int = 512
    vocab_size: int = 50265
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    layerdrop: float = 0.0
    init_std: float = 0.02
    bias: bool = True
    num_experts: int = 4
    expert_capacity: int = 128
    expert_dropout: float = 0.1
    use_cache: bool = True
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    is_encoder_decoder: bool = False
    decoder_start_token_id: Optional[int] = None
    forced_eos_token_id: Optional[int] = None
    scale_embedding: bool = False
    tie_word_embeddings: bool = True
    use_return_dict: bool = True

    def __post_init__(self):
        """Validate configuration after initialization.."""
        if self.model_type != "math_reasoning":
            raise ValueError(
                f"Invalid model_type: {self.model_type}. "
                "Must be 'math_reasoning'."
            )

@dataclass
class MathTrainingConfig:
    """Configuration for mathematical reasoning training.."""

    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: int = 3
    max_steps: int = -1
    warmup_steps: int = 0
    logging_steps: int = 500
    save_steps: int = 500
    save_total_limit: Optional[int] = None
    no_cuda: bool = False
    seed: int = 42
    fp16: bool = False
    fp16_opt_level: str = "O1"
    local_rank: int = -1
    tpu_num_cores: Optional[int] = None
    debug: bool = False
    dataloader_drop_last: bool = False
    eval_steps: int = 1000
    past_index: int = -1
    run_name: Optional[str] = None
    disable_tqdm: Optional[bool] = None
    remove_unused_columns: bool = True
    label_names: Optional[List[str]] = None
    load_best_model_at_end: bool = False
    metric_for_best_model: Optional[str] = None
    greater_is_better: Optional[bool] = None
    ignore_data_skip: bool = False
    sharded_ddp: bool = False
    deepspeed: Optional[str] = None
    label_smoothing_factor: float = 0.0
    adafactor: bool = False
    group_by_length: bool = False
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    ddp_find_unused_parameters: Optional[bool] = None
    dataloader_pin_memory: bool = True
    skip_memory_metrics: bool = False
    use_legacy_prediction_loop: bool = False
    push_to_hub: bool = False
    resume_from_checkpoint: Optional[str] = None
    hub_model_id: Optional[str] = None
    hub_strategy: str = "every_save"
    hub_token: Optional[str] = None
    gradient_checkpointing: bool = False
    include_inputs_for_metrics: bool = False
    auto_find_batch_size: bool = False
