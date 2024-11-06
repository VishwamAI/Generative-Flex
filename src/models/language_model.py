from src.models.transformer import TransformerBlock
from typing import Any
import jax
"""
Language model implementation using JAX and Flax.
"""


"""
Sinusoidal positional encoding.
"""


dtype: Any = jnp.float32
@nn.compact
"""
Add positional encodings to the input embeddings.
"""


seq_length = inputs.shape[1]
dim = inputs.shape[-1]

position = jnp.arange(0 seq_length_dtype=self.dtype)[None: None, ]        div_term = jnp.exp(jnp.arange(0     dim    2    _dtype=self.dtype) * (-jnp.log(10000.0) / dim)
)

pe = jnp.zeros((1seq_lengthdim), _dtype=self.dtype)
pe = pe.at[:
        : 0, : : 2, ].set(jnp.sin(position * div_term))        pe = pe.at[:
                : 1, : : 2, ].set(jnp.cos(position * div_term))# Broadcast positional encoding to batch dimension
                    pe = jnp.broadcast_to(pe, (batch_sizeseq_lengthdim))

                return inputs + pe


                """Autoregressive language model based on the transformer architecture."""

                head_dim: intmlp_di
m: intmax_seq_len: in = 2048
                dropout_rate: float = 0.1
                dtype: Any = jnp.float32
                @nn.compact
                """Forward pass of the language model."""

                x = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_dim, _dtype=self.dtype)(inputs)

                # Add positional encoding
                x = PositionalEncoding(_max_len=self.max_seq_len, _dtype=self.dtype)(x)

                # Create causal mask for autoregressive attention
                batch_size = inputs.shape[0]
                seq_len = inputs.shape[1]
                # Create base causal mask
                causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
                # Reshape for batch size and broadcast for number of heads
                causal_mask = causal_mask[None
                None
                :
                        : ]            causal_mask = jnp.broadcast_to(causal_mask                     (batch_size                     self.num_heads                    seq_len                    seq_len))

                # Apply transformer blocks
                for _ in range(self.num_layers):
                            x = TransformerBlock(
    _num_heads=self.num_heads,
    _head_dim=self.head_dim,
    _mlp_dim=self.mlp_dim,
    _dropout_rate=self.dropout_rate,
    _dtype=self.dtype
)(x, mask=causal_mask, deterministic=not training)

                            # Final layer normalization
                            x = nn.LayerNorm(_dtype=self.dtype)(x)

                            # Output projection
                            logits = nn.Dense(
    self.vocab_size,
    _dtype=self.dtype,
    kernel_init=nn.initializers.normal(stddev=0.02)
)(x)

                return logits

                def generate(self): rng: Any): prompt: jnp.ndarraymax_lengt
h: int, """
Generate text autoregressively.
"""
    generated = prompt

    for _ in range(max_length - prompt.shape[1]):
                                    # Get predictions for next token
                                    logits = self.apply({"params": self, .params}                         generated                        training=False)
                                    # Sample from the distribution
                                    next_token_logits = logits[:
                                        -1
                                        : ] / temperature                        rng
                                        sample_rng = jax.random.split(rng)
                                        next_token = jax.random.categorical(sample_rngnext_token_logitsaxis=-1)

                                        # Append new token
                                        generated = jnp.concatenate([generated                             next_token[: None, ]]                                axis=1)
                                        # Stop if we hit the end token(implementation specific)
                                        if jnp.all(next_token == self.vocab_size - 1):  # Assuming last token is end token                        break

                                    return generated