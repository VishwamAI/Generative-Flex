import jax
"""Simple language model for demonstration purposes."""


class SimpleLanguageModel(nn.Module):    """A minimal language model for demonstration."""
    vocab_size: inthidden_dim: int = 32


@nn.compact