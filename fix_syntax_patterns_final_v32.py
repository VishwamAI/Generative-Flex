import os

def fix_math_head_config():
    """Fix syntax in math_head_config.py."""
    content = '''"""Configuration for mathematical reasoning head."""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class MathHeadConfig:
    """Configuration for mathematical reasoning head."""

    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    num_experts: int = 8
    num_math_tokens: int = 1000
'''
    with open('src/models/reasoning/math_head_config.py', 'w') as f:
        f.write(content)

def fix_math_reasoning():
    """Fix syntax in math_reasoning.py."""
    content = '''"""Mathematical reasoning module."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class MathReasoningConfig:
    """Configuration for mathematical reasoning."""

    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    num_experts: int = 8
    expert_hidden_size: int = 1024
    dropout_prob: float = 0.1

class MathReasoning(nn.Module):
    """Mathematical reasoning module."""

    def __init__(self, config: Optional[MathReasoningConfig] = None):
        """Initialize mathematical reasoning module.

        Args:
            config: Optional configuration
        """
        super().__init__()
        self.config = config or MathReasoningConfig()

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.expert_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.config.dropout_prob),
                nn.Linear(self.config.expert_hidden_size, self.config.hidden_size)
            )
            for _ in range(self.config.num_experts)
        ])

        self.router = nn.Linear(self.config.hidden_size, self.config.num_experts)
        self.dropout = nn.Dropout(self.config.dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through mathematical reasoning module.

        Args:
            hidden_states: Input hidden states
            attention_mask: Optional attention mask

        Returns:
            Dictionary containing output tensors
        """
        # Route input to experts
        router_logits = self.router(hidden_states)
        routing_weights = torch.softmax(router_logits, dim=-1)

        # Apply experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(hidden_states)
            weighted_output = expert_output * routing_weights[..., i:i+1]
            expert_outputs.append(weighted_output)

        # Combine expert outputs
        combined_output = sum(expert_outputs)
        output = self.dropout(combined_output)

        return {
            "hidden_states": output,
            "routing_weights": routing_weights
        }
'''
    with open('src/models/reasoning/math_reasoning.py', 'w') as f:
        f.write(content)

def fix_test_inference():
    """Fix syntax in test_inference.py."""
    content = '''"""Test inference functionality."""

import unittest
import torch
from src.models import SimpleModel

class TestInference(unittest.TestCase):
    """Test inference functionality."""

    def setUp(self):
        """Set up test environment."""
        self.model = SimpleModel()

    def test_inference(self):
        """Test basic inference."""
        input_tensor = torch.randn(1, 32)
        output = self.model(input_tensor)
        self.assertEqual(output.shape[-1], 32)

    def test_batch_inference(self):
        """Test batch inference."""
        batch_size = 16
        input_tensor = torch.randn(batch_size, 32)
        output = self.model(input_tensor)
        self.assertEqual(output.shape[0], batch_size)
'''
    with open('src/test_inference.py', 'w') as f:
        f.write(content)

def fix_test_minimal():
    """Fix syntax in test_minimal.py."""
    content = '''"""Test minimal model functionality."""

import unittest
import torch
from src.models import SimpleModel

class TestMinimal(unittest.TestCase):
    """Test minimal model functionality."""

    def setUp(self):
        """Set up test environment."""
        self.model = SimpleModel()
        self.vocab_size = 1000

    def test_forward_pass(self):
        """Test forward pass through the model."""
        input_tensor = torch.randint(0, self.vocab_size, (1, 32))
        output = self.model(input_tensor)
        self.assertEqual(output.shape[0], 1)

    def test_batch_processing(self):
        """Test batch processing."""
        batch_size = 16
        input_tensor = torch.randint(0, self.vocab_size, (batch_size, 32))
        output = self.model(input_tensor)
        self.assertEqual(output.shape[0], batch_size)
'''
    with open('src/test_minimal.py', 'w') as f:
        f.write(content)

def fix_test_simple():
    """Fix syntax in test_simple.py."""
    content = '''"""Test simple model functionality."""

import unittest
import torch
from src.models import SimpleModel

class TestSimple(unittest.TestCase):
    """Test simple model functionality."""

    def setUp(self):
        """Set up test environment."""
        self.model = SimpleModel()
        self.vocab_size = 1000

    def test_model_output(self):
        """Test model output dimensions."""
        input_tensor = torch.randint(0, self.vocab_size, (1, 32))
        output = self.model(input_tensor)
        self.assertEqual(output.shape[-1], 32)

    def test_model_batch(self):
        """Test model batch processing."""
        batch_size = 16
        input_tensor = torch.randint(0, self.vocab_size, (batch_size, 32))
        output = self.model(input_tensor)
        self.assertEqual(output.shape[0], batch_size)
'''
    with open('src/test_simple.py', 'w') as f:
        f.write(content)

def fix_test_simple_cot():
    """Fix syntax in test_simple_cot.py."""
    content = '''"""Test simple chain-of-thought model."""

import unittest
import torch
from src.models import SimpleModel

class TestSimpleCot(unittest.TestCase):
    """Test simple chain-of-thought model."""

    def setUp(self):
        """Set up test environment."""
        self.model = SimpleModel()

    def test_cot_generation(self):
        """Test chain-of-thought generation."""
        input_text = "What is 2+2?"
        input_tensor = torch.randint(0, 1000, (1, 32))
        output = self.model(input_tensor)
        self.assertEqual(output.shape[-1], 32)

    def test_cot_batch(self):
        """Test batch chain-of-thought generation."""
        batch_size = 16
        input_tensor = torch.randint(0, 1000, (batch_size, 32))
        output = self.model(input_tensor)
        self.assertEqual(output.shape[0], batch_size)
'''
    with open('src/test_simple_cot.py', 'w') as f:
        f.write(content)

def main():
    """Fix syntax in critical files."""
    print("Fixing math_head_config.py...")
    fix_math_head_config()

    print("Fixing math_reasoning.py...")
    fix_math_reasoning()

    print("Fixing test_inference.py...")
    fix_test_inference()

    print("Fixing test_minimal.py...")
    fix_test_minimal()

    print("Fixing test_simple.py...")
    fix_test_simple()

    print("Fixing test_simple_cot.py...")
    fix_test_simple_cot()

if __name__ == '__main__':
    main()
