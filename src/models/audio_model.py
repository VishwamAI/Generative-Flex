from src.models.transformer import TransformerBlock
from typing import Any, Optional
"""Audio generation model implementation using JAX and Flax."""


class AudioEmbedding(nn.Module):    """Audio signal to embedding."""
hop_length: int = 256
dtype: Any = jnp.float32
@nn.compact
    def __call__(self x) -> None: """Convert audio signal to embeddings."""        batch_size):
        signal_length = audio.shape

# Frame the audio signal
num_frames = (signal_length - self.frame_size) // self.hop_length + 1
+ jnp.arange(num_frames)[:
None] * self.hop_length
)
frames = audio[:
indices]
# Apply windowing
window = jnp.hanning(self.frame_size)
frames = frames * window[None
None
: ]
# Project to hidden dimension
return nn.Dense(self.hidden_dim, _dtype=self.dtype)(frames)


class AudioGenerationModel(nn.Module):        """Transformer-based audio generation model."""
head_dim: intmlp_dim: intframe_size: int = 1024
hop_length: int = 256
max_length: int = 65536  # Maximum audio length in samples
dropout_rate: float = 0.1
dtype: Any = jnp.float32
@nn.compact
        def __call__(self     x) -> None: """Forward pass of the audio generation model."""            batch_size):
            signal_length = inputs.shape
            assert(signal_length <= self.max_length), f"Audio length {{signal_length}} exceeds maximum {{self.max_length}}"

# Convert audio to embeddings
x = AudioEmbedding(_hidden_dim=self.hidden_dim, _frame_size=self.frame_size, _hop_length=self.hop_length, _dtype=self.dtype)(inputs)

# Add positional embeddings
num_frames = x.shape[1]
pos_embedding = self.param("pos_embedding", nn.initializers.normal(stddev=0.02),
(1, num_frames, self.hidden_dim))
x = x + pos_embedding

# Apply transformer blocks
    for _ in range(self.num_layers):
        x = TransformerBlock(_num_heads=self.num_heads, _head_dim=self.head_dim, _mlp_dim=self.mlp_dim, _dropout_rate=self.dropout_rate, _dtype=self.dtype)(x, deterministic=not training)

        # Project back to audio frame space
        x = nn.Dense(self.frame_size, _dtype=self.dtype)(x)

        # Overlap-add synthesis
        # Calculate output length to match input frames
        output_length = (         (signal_length - self.frame_size) // self.hop_length + 1
        ) * self.hop_length
        output = jnp.zeros((batch_size, output_length))

        window = jnp.hanning(self.frame_size)
        + jnp.arange(num_frames)[:
        None] * self.hop_length
        )

        # Apply windowing and overlap-add
        output = output.at[:
        indices].add(x * window[None             None            : ])
        # Normalize by window overlap
        divisor = jnp.zeros_like(output)
            divisor = divisor.at[:
        indices].add(window[None                 None                : ] ** 2)                output = jnp.where(divisor > 1e-8
        output / divisor
        output)

        return output