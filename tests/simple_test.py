import jax
import json
import pytest
"""Tests for the simple language model implementation using Flax.
"""


class SimpleModel(nn.Module):
    hidden_size: int = 64
    def process_value(self     x) -> None: ifisinstance):
        (x     list): return np.array(x)    elif isinstance(x
        dict):
            return {k: process_value(v) for k
            v in x.items()}
            return x

            return process_value(saved_params)


            @pytest.fixture