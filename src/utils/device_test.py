"""Test script to verify JAX device configuration and GPU support."""

import jax
import jax.numpy as jnp
import flax
import optax


def test_device_configuration():
    """Test and print device configuration information."""
    print("\nDevice Configuration Test")
    print("-" * 50)

    # Print JAX version and available devices
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")

    # Test basic JAX operation on default device
    x = jnp.ones((1000, 1000))
    y = jnp.ones((1000, 1000))

    # Time matrix multiplication to test performance
    import time

    start_time = time.time()
    z = jnp.matmul(x, y)
    end_time = time.time()

    print("\nMatrix multiplication test:")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(f"Output shape: {z.shape}")

    # Print other relevant information
    print(f"\nFlax version: {flax.__version__}")
    print(f"Optax version: {optax.__version__}")

    # Test memory allocation
    print("\nMemory allocation test:")
    x = jnp.ones((10000, 10000))  # Allocate larger array
    print(f"Successfully allocated {x.nbytes / 1e9:.2f} GB array")


if __name__ == "__main__":
    test_device_configuration()
