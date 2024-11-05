from typing import Optio

nal, Union, List, Dict, Any, Tuple

from src.config.config import ModelConfig
from src.models.knowledge_retrieval import KnowledgeIntegrator
from src.models.text_to_anything import TextToAnything
import torch
import unittest

"""Comprehensive tests for all model features."""

    
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

    batch_size = 4
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids, attention_mask)
    self.assertEqual(
    outputs.shape, (batch_size, seq_length, self.config.hidden_size)
    )
