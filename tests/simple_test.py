"""Tests for the simple language model implementation using Flax."""

import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn


class SimpleModel(nn.Module):
    vocab_size: int
    hidden_size: int = 64

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.hidden_size)
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense2 = nn.Dense(self.hidden_size)
        self.output = nn.Dense(self.vocab_size)

    def __call__(self, x):
        x = self.embedding(x)
        x = nn.relu(self.dense1(x))
        x = nn.relu(self.dense2(x))
        x = self.output(x)
        return x


def init_model_state(model, rng, vocab_size):
    """Initialize model state with dummy input."""
    dummy_input = jnp.ones((1,), dtype=jnp.int32)
    params = model.init(rng, dummy_input)
    return params


def load_params(file_path):
    """Load and process saved parameters."""
    try:
        with open(file_path, "r") as f:
            saved_params = json.load(f)
    except FileNotFoundError:
        pytest.fail(f"Parameter file not found: {file_path}")
    except json.JSONDecodeError:
        pytest.fail(f"Invalid JSON in parameter file: {file_path}")

    # Convert lists to numpy arrays recursively
    def process_value(x):
        if isinstance(x, list):
            return np.array(x)
        elif isinstance(x, dict):
            return {k: process_value(v) for k, v in x.items()}
        return x

    return process_value(saved_params)


@pytest.fixture
def vocab_list():
    """Fixture providing test vocabulary."""
    return [
        "<unk>",
        "<pad>",
        "hi",
        "hello",
        "how",
        "are",
        "you",
        "good",
        "morning",
        "thanks",
        "bye",
    ]


@pytest.fixture
def word_mappings(vocab_list):
    """Fixture providing word-to-id and id-to-word mappings."""
    word_to_id = {word: idx for idx, word in enumerate(vocab_list)}
    id_to_word = {idx: word for idx, word in enumerate(vocab_list)}
    return word_to_id, id_to_word


@pytest.fixture
def model_params(tmp_path, vocab_list):
    """Fixture providing test model parameters."""
    params_dict = {
        "params": {
            "embedding": {"embedding": [[0.1] * 64] * len(vocab_list)},
            "dense1": {"kernel": [[0.1] * 64] * 64, "bias": [0.1] * 64},
            "dense2": {"kernel": [[0.1] * 64] * 64, "bias": [0.1] * 64},
            "output": {
                "kernel": [[0.1] * len(vocab_list)] * 64,
                "bias": [0.1] * len(vocab_list),
            },
        }
    }
    params_path = tmp_path / "model_params_minimal.json"
    with open(params_path, "w") as f:
        json.dump(params_dict, f)
    return load_params(params_path)


@pytest.fixture
def simple_model(vocab_list):
    """Fixture providing initialized SimpleModel."""
    return SimpleModel(vocab_size=len(vocab_list))


def test_model_initialization(simple_model, vocab_list):
    """Test that model initializes with correct parameters."""
    assert isinstance(simple_model, SimpleModel)
    assert simple_model.vocab_size == len(vocab_list)
    assert simple_model.hidden_size == 64


def test_init_model_state(simple_model, vocab_list):
    """Test model state initialization."""
    rng = jax.random.PRNGKey(0)
    params = init_model_state(simple_model, rng, len(vocab_list))
    assert "params" in params
    assert all(
        layer in params["params"]
        for layer in ["embedding", "dense1", "dense2", "output"]
    )


def test_model_forward_pass(simple_model, model_params, word_mappings):
    """Test model forward pass with test input."""
    word_to_id, _ = word_mappings
    test_input = "hi"
    input_token = jnp.array([word_to_id.get(test_input.lower(), word_to_id["<unk>"])])

    # Get model output
    logits = simple_model.apply(model_params, input_token)

    # Verify output shape and type
    assert logits.shape == (1, len(word_to_id))
    assert isinstance(logits, jnp.ndarray)
    assert not jnp.any(jnp.isnan(logits))


def test_end_to_end_inference(simple_model, model_params, word_mappings):
    """Test end-to-end inference pipeline."""
    word_to_id, id_to_word = word_mappings
    test_input = "hi"
    input_token = jnp.array([word_to_id.get(test_input.lower(), word_to_id["<unk>"])])

    # Get model output
    logits = simple_model.apply(model_params, input_token)
    predicted_token = jnp.argmax(logits, axis=-1)

    # Convert prediction to words
    response = " ".join([id_to_word[int(idx)] for idx in predicted_token])

    # Verify response
    assert isinstance(response, str)
    assert response.split()[0] in word_to_id
