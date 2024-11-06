import jax
import optax
import transformers

"""Test script to verify JAX/Flax/Optax installation.
"""


def test_environment(self) -> None:
        """Verify JAX installation and GPU availability.
        """

print("\nEnvironment Test Results:")
print("-" * 50)

# Test JAX
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")

# Test basic JAX operation
x = jnp.ones((2, 2))
y = jnp.ones((2, 2))
z = jnp.matmul(x, y)
print(f"Basic JAX operation successful: {z.shape}")

# Test Flax
print(f"Flax version: {flax.__version__}")

# Test Optax
print(f"Optax version: {optax.__version__}")

# Test other dependencies
print(f"TensorFlow Datasets version: {tfds.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Transformers version: {transformers.__version__}")

print("\nAll environment tests passed successfully!")

    if __name__ == "__main__":
        test_environment()