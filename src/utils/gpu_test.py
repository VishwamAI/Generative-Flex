"""Test script to verify GPU configuration and CUDA support in JAX."""

import jax
import jax.numpy as jnp
import time


def test_gpu_configuration():
    """Test GPU configuration and perform basic operations."""
    print("\nGPU Configuration Test")
    print("-" * 50)

    # Check available devices
    print("Available devices:")
    print(f"All devices: {jax.devices()}")
    print(f"GPU devices: {jax.devices('gpu')}")
    print(f"Default backend: {jax.default_backend()}")

    # Perform computation test
    print("\nComputation Test:")

    # Create large matrices for testing
    n = 5000
    x = jnp.ones((n, n))
    y = jnp.ones((n, n))

    # Time the computation
    start_time = time.time()
    result = jnp.dot(x, y)
    end_time = time.time()

    print(f"Matrix multiplication ({n}x{n}):")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(f"Result shape: {result.shape}")

    # Memory test
    print("\nMemory Test:")
    try:
        large_array = jnp.ones((20000, 20000))
        print(f"Successfully allocated {large_array.nbytes / 1e9:.2f} GB array")
    except Exception as e:
        print(f"Memory allocation failed: {str(e)}")


if __name__ == "__main__":
    test_gpu_configuration()
