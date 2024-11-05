from src.config.config import ModelConfig
from src.models.enhanced_transformer import EnhancedTransformer
import torch
import unittest

    """Test module for enhanced transformer models."""
        
        
        def setUp(self) -> None:
    """Set up test environment."""


self.config = ModelConfig(
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    intermediate_size=3072,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)
self.model = EnhancedTransformer(self.config)

seq_length = 128
input_ids = torch.randint(0, 1000, (batch_size, seq_length))
attention_mask = torch.ones_like(input_ids)

outputs = self.model(input_ids, attention_mask=attention_mask)
self.assertEqual(outputs.shape, (batch_size, seq_length, self.config.hidden_size))
