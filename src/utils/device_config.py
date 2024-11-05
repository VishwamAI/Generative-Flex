"""Device configuration utility for handling both CPU and GPU environments."""

import jax
import jax.numpy as jnp
from typing import Dict, Any
import os


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices and their capabilities."""
    return {
        "devices": jax.devices(),
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "process_index": jax.process_index(),
        "backend": jax.default_backend(),
        "has_gpu": any(d.platform == "gpu" for d in jax.devices()),
    }


def setup_device_config(
    memory_fraction: float = 0.8, gpu_allow_growth: bool = True
) -> Dict[str, Any]:
    """Configure device settings for optimal performance."""
    config = get_device_info()

    if config["has_gpu"]:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
            "false" if gpu_allow_growth else "true"
        )
        if not gpu_allow_growth:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)

    if config["device_count"] > 1:
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
            config["device_count"]
        )

    return config


def get_compute_dtype():
    """Get optimal compute dtype based on available hardware."""
    config = get_device_info()
    return jnp.bfloat16 if config["has_gpu"] else jnp.float32


if __name__ == "__main__":
    config = setup_device_config()
    print("\n=== Device Configuration ===")
    print(f"Device Info: {config}")
    print(f"Compute dtype: {get_compute_dtype()}")
