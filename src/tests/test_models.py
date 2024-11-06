import torch
import torch.nn as nn

from src.config.config import ModelConfig
from src.models.transformer import TransformerModel
import unittest


class TestModels:


    """
Class for TestModels..
"""def setUp(self):
    """
Method for setUp..
"""
    self.config = ModelConfig(
    hidden_size=64,
    num_attention_heads=4,
    num_hidden_layers=2,
    intermediate_size=128
    )

    def test_transformer_model(self):


        """
Method for test_transformer_model..
"""
    model = TransformerModel(self.config)
    self.assertIsInstance(model, nn.Module)

    def test_model_forward(self):


        """
Method for test_model_forward..
"""
    model = TransformerModel(self.config)
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    outputs = model(input_ids)
    self.assertEqual(outputs.shape, (batch_size, seq_length, self.config.hidden_size))

    if __name__ == '__main__':
    unittest.main()
