"""Tests for the language model chatbot implementation."""

from typing import Dict, List

import jax
import jax.numpy as jnp
import pytest

from src.models.language_model import LanguageModel


def tokenize(text: str, vocab: Dict[str, int]) -> List[int]:
    """Convert text to tokens using vocabulary."""
    # Simple whitespace tokenization for demonstration
    words = text.lower().split()
    return [vocab.get(word, vocab["<unk>"]) for word in words]


@pytest.fixture
def vocab() -> Dict[str, int]:
    """Fixture providing a minimal test vocabulary."""
    return {
        "<unk>": 0,
        "<pad>": 1,
        "hello": 2,
        "hi": 3,
        "good": 4,
        "morning": 5,
        "hey": 6,
        "greetings": 7,
        "how": 8,
        "are": 9,
        "you": 10,
    }


@pytest.fixture
def model_params():
    """Fixture providing standard test parameters for the model."""
    return {
        "max_length": 32,
        "hidden_dim": 64,
        "num_heads": 4,
        "head_dim": 16,
        "mlp_dim": 256,
        "num_layers": 2,
        "dropout_rate": 0.1,
    }


@pytest.fixture
def model(vocab, model_params):
    """Fixture providing initialized LanguageModel instance."""
    return LanguageModel(
        vocab_size=len(vocab),
        hidden_dim=model_params["hidden_dim"],
        num_heads=model_params["num_heads"],
        head_dim=model_params["head_dim"],
        mlp_dim=model_params["mlp_dim"],
        num_layers=model_params["num_layers"],
        dropout_rate=model_params["dropout_rate"],
        max_seq_len=model_params["max_length"],
    )


def test_model_initialization(model):
    """Test that model initializes correctly with given parameters."""
    assert isinstance(model, LanguageModel)
    assert model.vocab_size == 11  # Length of test vocabulary
    assert model.hidden_dim == 64
    assert model.num_heads == 4
    assert model.head_dim == 16
    assert model.mlp_dim == 256
    assert model.num_layers == 2
    assert model.dropout_rate == 0.1
    assert model.max_seq_len == 32


def test_tokenization(vocab):
    """Test that tokenization works correctly."""
    test_text = "hello how are you"
    tokens = tokenize(test_text, vocab)
    assert len(tokens) == 4
    assert tokens == [2, 8, 9, 10]  # Using indices from test vocabulary

    # Test unknown token handling
    test_text_with_unknown = "hello unknown word"
    tokens = tokenize(test_text_with_unknown, vocab)
    assert len(tokens) == 3
    assert tokens[0] == 2  # 'hello'
    assert tokens[1] == 0  # '<unk>'
    assert tokens[2] == 0  # '<unk>'


@pytest.mark.parametrize(
    "input_text,expected_tokens",
    [
        ("hello", [2]),
        ("hi", [3]),
        ("good morning", [4, 5]),
        ("hey", [6]),
        ("greetings", [7]),
    ],
)
def test_model_response(model, vocab, input_text, expected_tokens):
    """Test model responses for various input phrases."""
    # Initialize random parameters for testing
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, 1), dtype=jnp.int32))

    # Tokenize input
    tokens = tokenize(input_text, vocab)
    input_array = jnp.array([tokens])

    # Generate response
    output = model.apply(params, input_array, training=False)

    # Verify output shape and type
    assert output.shape[0] == 1  # Batch size
    assert output.shape[1] == len(tokens)  # Sequence length
    assert output.shape[2] == len(vocab)  # Vocabulary size
    assert output.dtype == jnp.float32

    # Verify output is valid probability distribution
    probabilities = jax.nn.softmax(output[0], axis=-1)
    assert jnp.allclose(jnp.sum(probabilities, axis=-1), 1.0, atol=1e-6)
