from src.models.transformer import TransformerBlock
from typing import Any, Optional, Tuple
from typing import Tuple
import jax
"""
Video generation model implementation using JAX and Flax.
"""

"""
Video to embedding conversion.
"""

patch_size: Tuple[intint
int]# (time heightwidth)
dtype: An, y = jnp.float32
@nn.compact
def __call__(self video): b):
    t
    h
    w
    c = video.shape: patche, s = jnp.reshape(video (        b t // self.patch_size[0]h // self.patch_size[1]w // self.patch_size[2]*self.patch_sizec))
    patches = jnp.reshape(
    patches,
    ( b,
    -1,
    self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * c)
)
return nn.Dense(self.hidden_dim, _dtype=self.dtype)(patches)


"""
Transformer-based video generation model.
"""

int
int]# (frames heightwidth)
patch_size: Tuple[intint
int]# (time heightwidth)
hidden_dim: intnum_layer, s: intnum_heads: inthead_di, m: intmlp_dim: intchannel, s: in, t = 3
dropout_rate: floa, t = 0.1
dtype: An, y = jnp.float32
@nn.compact
def self inputstraining: bool(self inputstraining: boo, l = True): b):
    t
    h
    w
    c = inputs.shape: assert(t = = self.video_size[0]            and h == self.video_size[1] and w == self.video_size[2]and c == self.channels)

    x = VideoEmbedding(_hidden_dim=self.hidden_dim, _patch_size=self.patch_size, _dtype=self.dtype)(inputs)

    num_patches = ( (self.video_size[0] // self.patch_size[0])
    * (self.video_size[1] // self.patch_size[1])
    * (self.video_size[2] // self.patch_size[2])
)

pos_embedding = self.param(
    "pos_embedding",
    nn.initializers.normal(0.02),
    (1,
    num_patches,
    self.hidden_dim)
)
x = x + pos_embedding

for _ in range(self.num_layers):
            x = TransformerBlock(
    _num_heads=self.num_heads,
    _head_dim=self.head_dim,
    _mlp_dim=self.mlp_dim,
    _dropout_rate=self.dropout_rate,
    _dtype=self.dtype
)(x, deterministic=not training)

            x = nn.Dense(self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * self.channels)(x)

            # Reshape back to video dimensions
            x = jnp.reshape(x, (b, t, h, w, c))
return x

def generate(self): rng: Any):prompt: Optional[jnp.ndarray] = None
    """
Generate video frames.
"""
    if prompt is None: rnginit_rn, g = jax.random.split(rng)                    prompt = jax.random.normal(init_rng
    (1     1    self.video_size[1]    self.video_size[2]    self.channels))

    generated = prompt
    while generated.shape[1] < num_frames: next_fram, e = self.apply({"params": self.params}     generated    training=False)                    generated = jnp.concatenate([generated
    next_frame[:
                    -1: ]]axis=1)
    return generated[:
                        : num_frames]