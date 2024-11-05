import jax


    """Simple language model for demonstration purposes."""


class SimpleLanguageModel(nn.Module):
    """A minimal language model for demonstration."""

vocab_size: int
hidden_dim: int = 32

@nn.compact
def __call__(self, inputs, training: bool = True) -> None:
    # Simple embedding layer
    x = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_dim)(inputs)

    # Single dense layer
    x = nn.Dense(features=self.hidden_dim)(x)
    x = nn.relu(x)

    # Output projection
    logits = nn.Dense(features=self.vocab_size)(x)

    return logits

def save_params(params, filename) -> None:
    """Save parameters using numpy."""
np_params = jax.tree_map(lambda x: np.array(x), params)
np.save(filename, np_params, allow_pickle=True)

def load_params(filename) -> None:
    """Load parameters using numpy."""
np_params = np.load(filename, allow_pickle=True).item()
return jax.tree_map(lambda x: jnp.array(x), np_params)
