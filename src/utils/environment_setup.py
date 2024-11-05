"""Environment setup and verification script."""

import os
import sys
from typing import Dict, Any

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import __version__ as flax_version
import optax
from tensorboardX import SummaryWriter

from src.utils.device_config import setup_device_config

# Set up device configuration
device_config = setup_device_config()


def test_jax_installation() -> Dict[str, Any]:
    """Test JAX installation and device configuration."""
    print("\n=== Testing JAX Installation ===")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")

    # Test basic operations
    x = jnp.ones((1000, 1000))
    y = jnp.ones((1000, 1000))

    # Time matrix multiplication
    import time

    start_time = time.time()
    jnp.dot(x, y)  # Perform matrix multiplication without storing result
    end_time = time.time()

    return {
        "jax_version": jax.__version__,
        "devices": str(jax.devices()),
        "backend": jax.default_backend(),
        "matrix_mult_time": end_time - start_time,
    }


def test_flax_installation() -> Dict[str, Any]:
    """Test Flax installation with a simple model."""
    print("\n=== Testing Flax Installation ===")

    # Create a small test model
    class SimpleModel(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=32)(x)
            x = nn.relu(x)
            x = nn.Dense(features=1)(x)
            return x

    # Initialize model
    model = SimpleModel()
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 16))
    variables = model.init(rng, dummy_input)

    return {
        "flax_version": flax_version,
        "model_params": sum(x.size for x in jax.tree_util.tree_leaves(variables)),
    }


def test_optax_installation() -> Dict[str, Any]:
    """Test Optax installation with optimizer creation."""
    print("\n=== Testing Optax Installation ===")

    # Create optimizer
    learning_rate = 1e-4

    # Test scheduler
    schedule_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0.0, transition_steps=1000
    )

    return {
        "optax_version": optax.__version__,
        "scheduler_type": str(type(schedule_fn)),
    }


def test_tensorboard_logging():
    """Test TensorBoard logging setup."""
    print("\n=== Testing TensorBoard Logging ===")

    log_dir = "logs/test_run"
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)
    writer.add_scalar("test/metric", 0.5, 0)
    writer.close()

    return os.path.exists(log_dir)


def main():
    """Run all environment tests."""
    try:
        # Test JAX
        jax_results = test_jax_installation()
        print("JAX test completed successfully")

        # Test Flax
        flax_results = test_flax_installation()
        print("Flax test completed successfully")

        # Test Optax
        optax_results = test_optax_installation()
        print("Optax test completed successfully")

        # Test TensorBoard
        tensorboard_success = test_tensorboard_logging()
        print("TensorBoard test completed successfully")

        print("\n=== Environment Test Results ===")
        print("JAX Configuration:")
        for k, v in jax_results.items():
            print(f"  {k}: {v}")

        print("\nFlax Configuration:")
        for k, v in flax_results.items():
            print(f"  {k}: {v}")

        print("\nOptax Configuration:")
        for k, v in optax_results.items():
            print(f"  {k}: {v}")

        print(f"\nTensorBoard Logging: {'✓' if tensorboard_success else '✗'}")

        print("\nAll environment tests completed successfully!")
        return True

    except Exception as e:
        print(f"Environment setup failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
