"""Image generation model implementation using JAX and Flax."""

from typing import Any, Optional, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn

from src.models.transformer import TransformerBlock


class PatchEmbedding(nn.Module):
    """Image to patch embedding."""

    patch_size: int
    hidden_dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, images):
        """Convert images to patch embeddings."""
        batch_size, height, width, channels = images.shape

        # Reshape image into patches
        patches = jnp.reshape(
            images,
            (
                batch_size,
                height // self.patch_size,
                width // self.patch_size,
                self.patch_size,
                self.patch_size,
                channels,
            ),
        )
        # Reshape patches into sequence
        patches = jnp.reshape(
            patches, (batch_size, -1, self.patch_size * self.patch_size * channels)
        )

        # Project patches to hidden dimension
        return nn.Dense(self.hidden_dim, dtype=self.dtype)(patches)


class ImageGenerationModel(nn.Module):
    """Transformer-based image generation model."""

    image_size: Tuple[int, int]  # (height, width)
    patch_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    head_dim: int
    mlp_dim: int
    channels: int = 3
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, training: bool = True):
        """Forward pass of the image generation model."""
        # Input shape validation
        batch_size, height, width, channels = inputs.shape
        assert height == self.image_size[0] and width == self.image_size[1]
        assert channels == self.channels

        # Convert image to patches and embed
        x = PatchEmbedding(
            patch_size=self.patch_size, hidden_dim=self.hidden_dim, dtype=self.dtype
        )(inputs)

        # Add learnable position embeddings
        num_patches = (self.image_size[0] // self.patch_size) * (
            self.image_size[1] // self.patch_size
        )
        pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, num_patches, self.hidden_dim),
        )
        x = x + pos_embedding

        # Apply transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
            )(x, deterministic=not training)

        # Project back to patch space
        x = nn.Dense(
            self.patch_size * self.patch_size * self.channels, dtype=self.dtype
        )(x)

        # Reshape back to image
        x = jnp.reshape(
            x,
            (
                batch_size,
                self.image_size[0] // self.patch_size,
                self.image_size[1] // self.patch_size,
                self.patch_size,
                self.patch_size,
                self.channels,
            ),
        )

        # Final reshape to image dimensions
        x = jnp.reshape(
            x, (batch_size, self.image_size[0], self.image_size[1], self.channels)
        )

        return x

    def generate(
        self, rng: Any, condition: Optional[jnp.ndarray] = None, batch_size: int = 1
    ):
        """Generate images."""
        # Initialize with random noise if no condition is provided
        if condition is None:
            rng, init_rng = jax.random.split(rng)
            x = jax.random.normal(
                init_rng,
                (batch_size, self.image_size[0], self.image_size[1], self.channels),
                dtype=self.dtype,
            )
        else:
            x = condition

        # Generate image
        return self.apply({"params": self.params}, x, training=False)
