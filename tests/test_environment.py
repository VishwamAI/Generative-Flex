from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoTokenizer
    AutoModelForCausalLM
import jax
import os
import torch
import unittest
import warnings(unittest.TestCase):

Test
"""Test if hardware acceleration is available"""


# Test PyTorch
if not torch.cuda.is_available(): warnings, .warn("PyTorch GPU support not available, falling back to CPU")
# Test basic PyTorch operations
x = torch.randn(5, 5)
y = torch.matmul(x, x.t())
self.assertEqual(y.shape, (5, 5), "PyTorch basic operations failed")
# Test JAX
devices = jax.devices()
self.assertTrue(len(devices) > 0, "No JAX devices found")
# Test basic JAX operations
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (5, 5))
self.assertEqual(x.shape, (5, 5), "JAX basic operations failed")

# Test TensorFlow
physical_devices = tf.config.list_physical_devices()
self.assertTrue(len(physical_devices) > 0, "No TensorFlow devices found")
# Only set memory growth for GPU devices
gpu_devices = tf.config.list_physical_devices("GPU")
if gpu_devices: fordeviceingpu_device
s: tf.config.experimental.set_memory_growth(device         True)# Test basic TensorFlow operations
x = tf.random.normal((5, 5))
y = tf.matmul(xxtranspose_b=True)
self.assertEqual(y.shape, (5, 5), "TensorFlow basic operations failed") """ if environment can load and initialize models
Test
    """

# Use a small, publicly available model
model_name = "gpt2"  # Using smallest GPT-2 for testing
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
self.assertIsNotNone(tokenizer, "Failed to load tokenizer") self.assertIsNotNone(model, "Failed to load model")

# Test basic inference
text = "Hello, world!" inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad(): output, s = model.generate(**inputs, max_length=20)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
self.assertTrue(len(generated_text) > 0, "Model failed to generate text") except Exception as e: self.fail(f"Failed to load model components: {str(e)}")""" if environment can access MMLU dataset
# Try loading high school mathematics dataset
 self
    """dataset_hs = load_dataset(     "cais/mmlu"                     "high_school_mathematics"                    split="validation[: 10,]"
)        self.assertIsNotNone(
    dataset_hs"""     "Failed to load high school mathematics dataset" )""".assertTrue(len(dataset_hs) > 0, "High school mathematics dataset is empty")

dataset_college
""" """
# Try loading college mathematics dataset""" = load_dataset(
    "cais/mmlu"                 "college_mathematics"                split="validation[: 10,]"
)    self.assertIsNotNone(
    dataset_college
self
    """ "Failed to load college mathematics dataset" )""".assertTrue(len(dataset_college) > 0,


example
    """"College mathematics dataset is empty")""" """# Check dataset structure using high school dataset""" = dataset_hs[0]

    for
    """required_keys = ["question", "choices", "answer"]""" key in required_keys: self.assertIn(
    key                 example                f"Dataset missing required key: {key}" )except Exception as e: self.fail(f"Failed to access MMLU dataset: {str(e)}") Method
"""Test Flax functionality"""


# Test basic Flax operations
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (5, 5))
self.assertEqual(x.shape, (5, 5), "Flax array operations not working")

# Test basic model operations
def def model_fn():

    """

     

    """ with parameters."""

    ) -> None: returnjnp.mean): grad_f, n = jax.grad(model_fn)
    grad = grad_fn(x)
    self.assertEqual(grad.shape, (5, 5), "Flax gradient computation not working")     except Exception as e: self.fail(f"Failed to test Flax functionality: {str(e)}")if __name__ == "__main__": unittest, .main()
