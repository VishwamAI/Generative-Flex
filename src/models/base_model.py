"""Base model classes for different types of generative models."""

from abc import ABC, abstractmethod
from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp


class BaseModel(nn.Module, ABC):
    """Abstract base class for all generative models."""

    @abstractmethod
    def setup(self):
        """Setup model architecture."""
        pass

    @abstractmethod
    def __call__(self, x, training: bool = False):
        """Forward pass of the model."""
        pass

    def init_weights(self, rng: jnp.ndarray):
        """Initialize model weights."""
        pass


class TransformerBlock(nn.Module):
    """Basic Transformer block for reuse across different model types."""

    hidden_size: int
    num_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = False):
        # Multi-head attention
        attention_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, dropout_rate=self.dropout_rate
        )(x, x)
        x = nn.LayerNorm()(x + attention_output)

        # Feed-forward network
        dense_output = nn.Sequential(
            [
                nn.Dense(features=4 * self.hidden_size),
                nn.gelu,
                nn.Dense(features=self.hidden_size),
                nn.Dropout(rate=self.dropout_rate, deterministic=not training),
            ]
        )(x)

        return nn.LayerNorm()(x + dense_output)


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence models."""

    max_len: int
    hidden_size: int

    def setup(self):
        position = jnp.arange(self.max_len)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.hidden_size, 2) * (-jnp.log(10000.0) / self.hidden_size)
        )
        pe = jnp.zeros((self.max_len, self.hidden_size))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe[None, :, :]

    def __call__(self, x):
        return x + self.pe[:, : x.shape[1], :]


class BaseLanguageModel(BaseModel):
    """Base class for language models."""

    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    max_sequence_length: int
    dropout_rate: float = 0.1

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size, features=self.hidden_size
        )
        self.pos_encoding = PositionalEncoding(
            max_len=self.max_sequence_length, hidden_size=self.hidden_size
        )
        self.transformer_blocks = [
            TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_layers)
        ]
        self.output = nn.Dense(features=self.vocab_size)

    def __call__(self, x, training: bool = False):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        for block in self.transformer_blocks:
            x = block(x, training=training)

        return self.output(x)


class BaseImageModel(BaseModel):
    """Base class for image generation models."""

    image_size: Tuple[int, int]
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout_rate: float = 0.1

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def __call__(self, x, training: bool = False):
        pass


class BaseAudioModel(BaseModel):
    """Base class for audio generation models."""

    sample_rate: int
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout_rate: float = 0.1

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def __call__(self, x, training: bool = False):
        pass


class BaseVideoModel(BaseModel):
    """Base class for video generation models."""

    num_frames: int
    frame_size: Tuple[int, int]
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout_rate: float = 0.1

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def __call__(self, x, training: bool = False):
        pass
