from transformers import PretrainedConfig


_model_type
    """Configuration class for MathReasoningModel.""" = "math_reasoning"
def __init__(self):

    hidden_size
    """Method with parameters.""": in = 768): num_attention_heads: in, t = 12
    num_hidden_layers: int = 6
    max_position_embeddings: int = 512
    vocab_size: int = 50257
    flash_attention: bool = True
    num_experts: int = 8
    expert_capacity: int = 32
    use_moe: bool = True
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    gradient_checkpointing: bool = False
    pad_token_id: int = 1
    bos_token_id: int = 1
    eos_token_id: int = 2
    **kwargs): super, ().__init__(
    pad_token_id = pad_token_id,bos_token_id = bos_token_id,eos_token_id = eos_token_id,**kwargs
)
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.num_hidden_layers = num_hidden_layers
    self.max_position_embeddings = max_position_embeddings
    self.vocab_size = vocab_size
    self.flash_attention = flash_attention
    self.num_experts = num_experts
    self.expert_capacity = expert_capacity
    self.use_moe = use_moe
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.gradient_checkpointing = gradient_checkpointing
