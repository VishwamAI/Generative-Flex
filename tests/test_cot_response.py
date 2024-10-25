"""Test module for chain-of-thought response generation."""

import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn


class SimpleChatModel(nn.Module):
    vocab_size: int
    hidden_size: int = 64

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size, features=self.hidden_size
        )
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense2 = nn.Dense(self.hidden_size)
        self.output = nn.Dense(self.vocab_size)

    def __call__(self, x):
        x = self.embedding(x)
        x = jnp.mean(x, axis=0)  # Average over sequence length
        x = nn.relu(self.dense1(x))
        x = nn.relu(self.dense2(x))
        x = self.output(x)
        return x


@pytest.fixture
def vocab():
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
def word_mappings(vocab):
    """Fixture providing word-to-id and id-to-word mappings."""
    word_to_id = {word: i for i, word in enumerate(vocab)}
    id_to_word = {i: word for i, word in enumerate(vocab)}
    return word_to_id, id_to_word


@pytest.fixture
def model_params(vocab, chat_model):
    """Fixture providing test model parameters."""
    # Initialize parameters using Flax's init method
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1,), dtype=jnp.int32)
    variables = chat_model.init(key, dummy_input)
    return variables["params"]


@pytest.fixture
def chat_model(vocab):
    """Fixture providing initialized SimpleChatModel."""
    return SimpleChatModel(vocab_size=len(vocab))


def test_model_initialization(chat_model, vocab):
    """Test that model initializes with correct parameters."""
    assert isinstance(chat_model, SimpleChatModel)
    assert chat_model.vocab_size == len(vocab)
    assert chat_model.hidden_size == 64


def test_model_forward_pass(chat_model, model_params, word_mappings):
    """Test model forward pass with test input."""
    word_to_id, _ = word_mappings

    # Test input
    test_input = "hi"
    input_tokens = jnp.array(
        [word_to_id.get(w, word_to_id["<unk>"]) for w in test_input.split()]
    )

    # Generate response
    logits = chat_model.apply({"params": model_params}, input_tokens)

    # Verify output shape and type
    assert logits.shape == (len(word_to_id),)
    assert isinstance(logits, jnp.ndarray)
    assert not jnp.any(jnp.isnan(logits))


def test_response_generation(chat_model, model_params, word_mappings):
    """Test end-to-end response generation."""
    word_to_id, id_to_word = word_mappings

    # Test input
    test_input = "hi"
    input_tokens = jnp.array(
        [word_to_id.get(w, word_to_id["<unk>"]) for w in test_input.split()]
    )

    # Generate response
    logits = chat_model.apply({"params": model_params}, input_tokens)
    predicted_tokens = jnp.argsort(logits)[-10:][::-1]

    # Convert tokens back to words
    response_words = [id_to_word[int(token)] for token in predicted_tokens]
    response = " ".join(response_words)

    # Verify response
    assert isinstance(response, str)
    assert len(response_words) == 10
    assert all(word in word_to_id for word in response_words)


def test_unknown_token_handling(chat_model, model_params, word_mappings):
    """Test model handling of unknown tokens."""
    word_to_id, _ = word_mappings

    # Test input with unknown word
    test_input = "unknown_word"
    input_tokens = jnp.array(
        [word_to_id.get(w, word_to_id["<unk>"]) for w in test_input.split()]
    )

    # Verify unknown token is handled
    assert input_tokens[0] == word_to_id["<unk>"]

    # Generate response
    logits = chat_model.apply({"params": model_params}, input_tokens)
    assert not jnp.any(jnp.isnan(logits))
