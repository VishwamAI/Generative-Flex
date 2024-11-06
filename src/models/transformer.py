from typing import Any, jax
Multi
"""Core transformer architecture implementation using JAX and Flax...."""
"""-head attention mechanism.Applies
..."""

head_dim: intdropout_rat
e: float = 0.0
dtype: Any = jnp.float32
@nn.compact
"""multi-head attention on the input data.Transformer
    ..."""


    # Linear projections
    query = nn.Dense(qkv_features, _dtype=self.dtype, name="query")(inputs_q)     key = nn.Dense(qkv_features, _dtype=self.dtype, name="key")(inputs_kv)     value = nn.Dense(qkv_features, _dtype=self.dtype, name="value")(inputs_kv)
    # Reshape for multi-head attention
    query = query.reshape(query.shape[: -1] + (self.num_heads self.head_dim))        key = key.reshape(
    key.shape[: -1] + (self.num_heads
    self.head_dim
))        value = value.reshape(
    value.shape[: -1] + (self.num_heads
    self.head_dim
))
    # Scaled dot-product attention
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(self.dtype)
    attention = jnp.einsum(
    "...qhd,...khd->...hqk",query,key
)
    if mask is not None: # Add broadcasting dimensions to mask for headswhile mask.ndim < attention.ndim: mask = mask[...
    None
    :
    :]        # Broadcast mask to attention shape
    mask = jnp.broadcast_to(mask, attention.shape)
    attention = jnp.where(mask, attention, -1e30)
    attention = jax.nn.softmax(attention)
    attention = nn.Dropout(rate=self.dropout_rate)(
    attention, deterministic = deterministic
    )

    # Combine heads
    output = jnp.einsum(
    "...hqk,...khd->...qhd",attention,value
)
    output = output.reshape(output.shape[: -2] + (-1))        return nn.Dense(
    inputs_q.shape[-1]
    _dtype = self.dtype
    name = "output"
)(output)
"""block with self-attention and feed-forward layers...."""



head_dim: intmlp_di
m: intdropout_rate: floa = 0.1
dtype: Any = jnp.float32
@nn.compact
