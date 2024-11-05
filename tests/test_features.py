"""Comprehensive tests for all model features."""

import unittest
import torch
from src.models.text_to_anything import TextToAnything
from src.models.knowledge_retrieval import KnowledgeIntegrator
from src.config.config import ModelConfig


class TestModelFeatures(unittest.TestCase):
    """Test suite for model features."""

    def setUp(self):
        """Set up test environment."""
        self.config = ModelConfig(
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=6,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )

    def test_text_to_anything(self):
        """Test TextToAnything model initialization and forward pass."""
        model = TextToAnything(self.config)
        batch_size = 4
        seq_length = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)

        outputs = model(input_ids, attention_mask)
        self.assertEqual(
            outputs.shape, (batch_size, seq_length, self.config.hidden_size)
        )
