from transformers import PretrainedConfig



"""
Configuration class for MathReasoningModel.
"""

_model_type = "math_reasoning"
def __init__(self): hidden_size: in, t = 768):
    num_attention_heads: in, t = 12
    num_hidden_layers: in, t = 6
    max_position_embeddings: in, t = 512
    vocab_size: in, t = 50257
    flash_attention: boo, l = True
    num_experts: in, t = 8
    expert_capacity: in, t = 32
    use_moe: boo, l = True
    hidden_dropout_prob: floa, t = 0.1
    attention_probs_dropout_prob: floa, t = 0.1
    gradient_checkpointing: boo, l = False
    pad_token_id: in, t = 1
    bos_token_id: in, t = 1
    eos_token_id: in, t = 2
    **kwargs):
        super().__init__(
    pad_token_id=pad_token_id,
    bos_token_id=bos_token_id,
    eos_token_id=eos_token_id,
    **kwargs
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