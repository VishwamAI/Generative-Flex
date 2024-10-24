"""Video generation model implementation using JAX and Flax."""

from typing import Any, Optional, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn

from src.models.transformer import TransformerBlock


class VideoEmbedding(nn.Module):
    """Video to embedding conversion."""

    hidden_dim: int
    patch_size: Tuple[int, int, int]  # (time, height, width)
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, video):
        b, t, h, w, c = video.shape
        patches = jnp.reshape(
            video,
            (
                b,
                t // self.patch_size[0],
                h // self.patch_size[1],
                w // self.patch_size[2],
                *self.patch_size,
                c,
            ),
        )
        patches = jnp.reshape(
            patches,
            (b, -1, self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * c),
        )
        return nn.Dense(self.hidden_dim, dtype=self.dtype)(patches)


class VideoGenerationModel(nn.Module):
    """Transformer-based video generation model."""

    video_size: Tuple[int, int, int]  # (frames, height, width)
    patch_size: Tuple[int, int, int]  # (time, height, width)
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
        b, t, h, w, c = inputs.shape
        assert (
            t == self.video_size[0]
            and h == self.video_size[1]
            and w == self.video_size[2]
            and c == self.channels
        )

        x = VideoEmbedding(
            hidden_dim=self.hidden_dim, patch_size=self.patch_size, dtype=self.dtype
        )(inputs)

        num_patches = (
            (self.video_size[0] // self.patch_size[0])
            * (self.video_size[1] // self.patch_size[1])
            * (self.video_size[2] // self.patch_size[2])
        )

        pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(0.02),
            (1, num_patches, self.hidden_dim),
        )
        x = x + pos_embedding

        for _ in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
            )(x, deterministic=not training)

        x = nn.Dense(
            self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * self.channels
        )(x)

        # Reshape back to video dimensions
        x = jnp.reshape(x, (b, t, h, w, c))
        return x

    def generate(
        self, rng: Any, prompt: Optional[jnp.ndarray] = None, num_frames: int = 16
    ):
        """Generate video frames."""
        if prompt is None:
            rng, init_rng = jax.random.split(rng)
            prompt = jax.random.normal(
                init_rng, (1, 1, self.video_size[1], self.video_size[2], self.channels)
            )

        generated = prompt
        while generated.shape[1] < num_frames:
            next_frame = self.apply({"params": self.params}, generated, training=False)
            generated = jnp.concatenate([generated, next_frame[:, -1:]], axis=1)

        return generated[:, :num_frames]
