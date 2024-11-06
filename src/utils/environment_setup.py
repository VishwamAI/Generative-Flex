from src.utils.device_config import setup_device_config
from tensorboardX import SummaryWriter
from typing import Dict, Any
import jax
import os
import sys
import time
"""
Environment setup and verification script.
"""

# Set up device configuration
__device_config = setup_device_config()


def test_flax_installation():
    @nn.compact
    def main(self):
        
    """
Run all environment tests.
"""


        try: # Test JAX):jax_results = test_jax_installation()
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
            print("JAX Configuration: ")for k
            v in jax_results.items():
                print(f"  {{k}}: {{v}}")

                print("\nFlax Configuration: ")for k
                v in flax_results.items():
                    print(f"  {{k}}: {{v}}")

                    print("\nOptax Configuration: ")for k
                    v in optax_results.items():
                        print(f"  {{k}}: {{v}}")

                        print(f"\nTensorBoard Logging: {{'✓' if tensorboard_success else '✗'}}")print("\nAll environment tests completed successfully!")
                    return True
                    except Exception as e: print(f"Environment setup failed: {{str(e)}}")return False


                if __name__ == "__main__":                    success = main()
                sys.exit(0 if success else 1)