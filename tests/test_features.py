"""Comprehensive tests for all model features."""

import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn

from src.models.enhanced_transformer import EnhancedConfig, EnhancedTransformer
from src.models.knowledge_retrieval import (
    KnowledgeConfig,
    KnowledgeAugmentedTransformer,
)
from src.models.text_to_anything import GenerationConfig, TextToAnything
from src.models.apple_optimizations import AppleOptimizedTransformer


@pytest.fixture
def config():
    return EnhancedConfig(
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=4,
        use_flash_attention=True,
        use_constitutional_ai=True,
        num_experts=8,
        safety_threshold=0.8,
        use_int4_quantization=True,
        use_neural_engine=True,
    )


@pytest.fixture
def knowledge_config():
    return KnowledgeConfig(
        embedding_size=512, num_retrievers=2, max_chunks=10, update_frequency=100
    )


@pytest.fixture
def text_to_anything_config():
    return GenerationConfig(
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=4,
        supported_modalities=["text", "image", "audio", "video"],
        use_constitutional_ai=True,
        safety_threshold=0.8,
        use_int4_quantization=True,
        use_neural_engine=True,
    )


def test_openai_features(config):
    """Test OpenAI-style features (GPT-4o, o1)."""
    model = EnhancedTransformer(config)

    # Test input
    batch_size = 2
    seq_length = 16
    inputs = {
        "input_ids": jax.random.randint(
            jax.random.PRNGKey(0), (batch_size, seq_length), 0, config.vocab_size
        ),
        "position_ids": jnp.arange(seq_length)[None, :].repeat(batch_size, axis=0),
        "token_type_ids": jnp.zeros((batch_size, seq_length), dtype=jnp.int32),
    }

    # Initialize parameters
    params = model.init(jax.random.PRNGKey(0), inputs)

    # Test forward pass with hidden state output
    output = model.apply(params, inputs, return_hidden=True)
    assert output.shape == (batch_size, seq_length, config.hidden_size)

    # Test generation
    generated = model.apply(params, inputs, method=model.generate, max_new_tokens=5)
    assert generated.shape[1] > seq_length


def test_anthropic_features(config):
    """Test Anthropic-style features (Constitutional AI)."""
    model = EnhancedTransformer(config)

    # Test input with potentially harmful content
    batch_size = 2
    seq_length = 16
    inputs = {
        "input_ids": jax.random.randint(
            jax.random.PRNGKey(0), (batch_size, seq_length), 0, config.vocab_size
        ),
        "position_ids": jnp.arange(seq_length)[None, :].repeat(batch_size, axis=0),
        "token_type_ids": jnp.zeros((batch_size, seq_length), dtype=jnp.int32),
    }

    # Initialize parameters
    params = model.init(jax.random.PRNGKey(0), inputs)

    # Test forward pass with constitutional layer
    output = model.apply(params, inputs, return_hidden=True)
    assert output.shape == (batch_size, seq_length, config.hidden_size)


def test_meta_features(config):
    """Test Meta-style features (Flash Attention)."""
    model = EnhancedTransformer(config)

    # Test input
    batch_size = 2
    seq_length = 16
    inputs = {
        "input_ids": jax.random.randint(
            jax.random.PRNGKey(0), (batch_size, seq_length), 0, config.vocab_size
        ),
        "position_ids": jnp.arange(seq_length)[None, :].repeat(batch_size, axis=0),
        "token_type_ids": jnp.zeros((batch_size, seq_length), dtype=jnp.int32),
    }

    # Initialize parameters
    params = model.init(jax.random.PRNGKey(0), inputs)

    # Test forward pass with flash attention
    output = model.apply(params, inputs, return_hidden=True)
    assert output.shape == (batch_size, seq_length, config.hidden_size)


def test_grok_features(knowledge_config):
    """Test Grok-style features (Real-time updates)."""
    model = KnowledgeAugmentedTransformer(knowledge_config)

    # Test input
    batch_size = 2
    seq_length = 16
    inputs = {
        "text": jax.random.normal(
            jax.random.PRNGKey(0),
            (batch_size, seq_length, knowledge_config.embedding_size),
        )
    }

    # Initialize parameters
    params = model.init(jax.random.PRNGKey(0), inputs)

    # Test forward pass with knowledge integration
    output = model.apply(params, inputs)
    assert output.shape == (batch_size, seq_length, knowledge_config.embedding_size)

    # Test real-time updates
    new_knowledge = {
        "text": jax.random.normal(
            jax.random.PRNGKey(1), (1, knowledge_config.embedding_size)
        )
    }
    model.apply(params, inputs, context=new_knowledge)


def test_gemini_features(text_to_anything_config):
    """Test Gemini-style features (Multi-modal)."""
    model = TextToAnything(text_to_anything_config)

    # Test multi-modal input
    batch_size = 2
    seq_length = 16
    hidden_size = text_to_anything_config.hidden_size
    inputs = {
        "text": jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, seq_length, hidden_size)
        ),
        "position_ids": jnp.arange(seq_length)[None, :].repeat(batch_size, axis=0),
        "token_type_ids": jnp.zeros((batch_size, seq_length), dtype=jnp.int32),
        "image": jax.random.normal(
            jax.random.PRNGKey(1),
            (batch_size, seq_length, hidden_size),  # Match text dimensions
        ),
    }

    # Initialize parameters
    params = model.init(jax.random.PRNGKey(0), inputs, target_modality="text")

    # Test forward pass with multi-modal input
    output = model.apply(params, inputs, target_modality="text")
    assert output.shape == (batch_size, seq_length, text_to_anything_config.hidden_size)


def test_apple_optimizations():
    """Test Apple-style optimizations."""
    config = EnhancedConfig(
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=4,
        use_int4_quantization=True,
        use_neural_engine=True,
        block_size=32,
        cache_dtype="float16",
        max_sequence_length=2048,
        head_dim=64,
        dropout_rate=0.1,
        use_privacy_preserving=True,
        vocab_size=50257,
        embedding_size=512,
    )

    model = AppleOptimizedTransformer(config)

    # Test input
    batch_size = 2
    seq_length = 16
    hidden_size = config.hidden_size
    inputs = {
        "input_ids": jax.random.randint(
            jax.random.PRNGKey(0), (batch_size, seq_length), 0, config.vocab_size
        ),
        "position_ids": jnp.arange(seq_length)[None, :].repeat(batch_size, axis=0),
        "token_type_ids": jnp.zeros((batch_size, seq_length), dtype=jnp.int32),
        "hidden_states": jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, seq_length, hidden_size)
        ),
    }

    # Initialize parameters
    params = model.init(jax.random.PRNGKey(0), inputs)

    # Test forward pass with quantization
    output = model.apply(params, inputs, return_hidden=True)
    assert output.shape == (batch_size, seq_length, config.hidden_size)


def test_text_to_anything_generation(text_to_anything_config):
    """Test text-to-anything generation capabilities."""
    model = TextToAnything(text_to_anything_config)

    # Initialize tokenizer and model parameters
    batch_size = 1
    seq_length = 16
    hidden_size = text_to_anything_config.hidden_size
    inputs = {
        "text": {
            "input_ids": jax.random.randint(
                jax.random.PRNGKey(0),
                (batch_size, seq_length),
                0,
                text_to_anything_config.vocab_size,
            ),
            "position_ids": jnp.arange(seq_length)[None, :],
            "token_type_ids": jnp.zeros((batch_size, seq_length), dtype=jnp.int32),
            "attention_mask": jnp.ones((batch_size, seq_length), dtype=jnp.int32),
        }
    }
    params = model.init(jax.random.PRNGKey(0), inputs, target_modality="text")

    # Test generation for different modalities
    for modality in text_to_anything_config.supported_modalities:
        # Add modality-specific input features
        if modality == "image":
            inputs["image"] = jax.random.normal(
                jax.random.PRNGKey(1),
                (batch_size, 32, 32, 3),  # Smaller image for testing
            )
        elif modality == "audio":
            inputs["audio"] = jax.random.normal(
                jax.random.PRNGKey(2), (batch_size, seq_length, hidden_size)
            )
        elif modality == "video":
            inputs["video"] = jax.random.normal(
                jax.random.PRNGKey(3),
                (batch_size, 8, 32, 32, 3),  # (batch, frames, height, width, channels)
            )

        output, metadata = model.apply(
            params,
            inputs,
            method=model.generate,
            target_modality=modality,
            max_length=seq_length,
        )

        # Verify output shapes based on modality
        if modality == "image":
            assert output.shape == (batch_size, 32, 32, 3)
        elif modality == "audio":
            assert output.shape == (batch_size, seq_length, hidden_size)
        elif modality == "video":
            assert output.shape == (batch_size, 8, 32, 32, 3)
        elif modality == "text":
            assert output.shape == (batch_size, seq_length, hidden_size)

        # Verify metadata
        assert "constitutional_compliant" in metadata
        assert "principles_applied" in metadata
        assert "generation_params" in metadata
