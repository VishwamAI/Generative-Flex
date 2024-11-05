from flax import linen as nn
from src.config.config import ModelConfig
from src.models.enhanced_transformer import EnhancedTransformer
import jax
import jax.numpy as jnp
import pytest

    """Script to fix test_models.py formatting."""


def create_fixed_content(self):
    """Create properly formatted content for test_models.py."""
content = """"""Test module for enhanced transformer models.

This module contains test cases for the enhanced transformer architecture,
including configuration validation and model behavior verification.
    """




@pytest.fixture
def model_config(self):
    """Fixture for enhanced transformer configuration."""
return ModelConfig(hidden_size=512, num_attention_heads=8, num_hidden_layers=6, intermediate_size=2048, vocab_size=32000, max_position_embeddings=512, attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1, type_vocab_size=2, initializer_range=0.02)


@pytest.fixture
def transformer(model_config):
    """Fixture for enhanced transformer model."""
return EnhancedTransformer(config=model_config)


def test_transformer_initialization(transformer):
    """Test transformer model initialization."""
assert isinstance(transformer, nn.Module)
assert transformer.config.hidden_size == 512
assert transformer.config.num_attention_heads == 8


def test_transformer_forward_pass(transformer):
    """Test transformer forward pass with sample input."""
batch_size = 2
sequence_length = 16
input_shape = (batch_size, sequence_length)

# Create sample input
inputs = jnp.ones(input_shape, dtype=jnp.int32)
attention_mask = jnp.ones(input_shape, dtype=jnp.int32)

# Initialize parameters
key = jax.random.PRNGKey(0)
variables = transformer.init(key, inputs, attention_mask)

# Run forward pass
outputs = transformer.apply(variables, inputs, attention_mask)

# Check output shape
expected_shape = (batch_size, sequence_length, transformer.config.hidden_size)
assert outputs.shape == expected_shape


def test_transformer_attention_mask(transformer):
    """Test transformer attention masking."""
batch_size = 2
sequence_length = 16
input_shape = (batch_size, sequence_length)

# Create inputs with attention mask
inputs = jnp.ones(input_shape, dtype=jnp.int32)
attention_mask = jnp.zeros(input_shape, dtype=jnp.int32)
attention_mask = attention_mask.at[:, :8].set(1)

# Initialize parameters
key = jax.random.PRNGKey(0)
variables = transformer.init(key, inputs, attention_mask)

# Run forward pass with masked attention
outputs = transformer.apply(variables, inputs, attention_mask)

# Check output shape
expected_shape = (batch_size, sequence_length, transformer.config.hidden_size)
assert outputs.shape == expected_shape


def test_transformer_config_validation(model_config):
    """Test transformer configuration validation."""
assert model_config.hidden_size % model_config.num_attention_heads == 0
assert model_config.hidden_size >= model_config.num_attention_heads
assert model_config.max_position_embeddings > 0
    """
return content


def main(self):
    """Main function to fix the file."""
# Create the fixed content
content = create_fixed_content()

# Write to file
with open("tests/test_models.py", "w") as f:
    f.write(content)
    print("Fixed test_models.py with proper docstring formatting")


    if __name__ == "__main__":
        main()
