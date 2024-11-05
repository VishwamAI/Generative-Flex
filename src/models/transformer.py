"""Core transformer architecture implementation using JAX and Flax."""

from typing import Any
import jax
import jax.numpy as jnp
import flax.linen as nn


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    num_heads: int
    head_dim: int
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs_q, inputs_kv, mask=None, deterministic=True):
        """Applies multi-head attention on the input data."""
        qkv_features = self.num_heads * self.head_dim

        # Linear projections
        query = nn.Dense(qkv_features, dtype=self.dtype, name="query")(inputs_q)
        key = nn.Dense(qkv_features, dtype=self.dtype, name="key")(inputs_kv)
        value = nn.Dense(qkv_features, dtype=self.dtype, name="value")(inputs_kv)

        # Reshape for multi-head attention
        query = query.reshape(query.shape[:-1] + (self.num_heads, self.head_dim))
        key = key.reshape(key.shape[:-1] + (self.num_heads, self.head_dim))
        value = value.reshape(value.shape[:-1] + (self.num_heads, self.head_dim))

        # Scaled dot-product attention
        depth = query.shape[-1]
        query = query / jnp.sqrt(depth).astype(self.dtype)
        attention = jnp.einsum("...qhd,...khd->...hqk", query, key)

        if mask is not None:
            # Add broadcasting dimensions to mask for heads
            while mask.ndim < attention.ndim:
                mask = mask[..., None, :, :]
            # Broadcast mask to attention shape
            mask = jnp.broadcast_to(mask, attention.shape)
            attention = jnp.where(mask, attention, -1e30)

        attention = jax.nn.softmax(attention)
        attention = nn.Dropout(rate=self.dropout_rate)(
            attention, deterministic=deterministic
        )

        # Combine heads
        output = jnp.einsum("...hqk,...khd->...qhd", attention, value)
        output = output.reshape(output.shape[:-2] + (-1,))
        return nn.Dense(inputs_q.shape[-1], dtype=self.dtype, name="output")(output)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward layers."""

    num_heads: int
    head_dim: int
    mlp_dim: int
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, mask=None, deterministic=True):
        """Applies Transformer block to the input data."""
        # Self-attention
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
        )(x, x, mask, deterministic)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        # Feed-forward network
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = nn.Dense(self.mlp_dim, dtype=self.dtype)(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        y = nn.Dense(inputs.shape[-1], dtype=self.dtype)(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)

        return x + y
