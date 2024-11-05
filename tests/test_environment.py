from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import jax
import os
import torch
import unittest
import warnings



class TestEnvironment(unittest.TestCase):

def setUp(:
    self
    ): -> None: warnings.filterwarnings("ignore")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure GPU is visible

def test_hardware_acceleration(:
    self
    ): -> None: """Test if hardware acceleration is available"""
        # Test PyTorch
        if not torch.cuda.is_available():
        warnings.warn("PyTorch GPU support not available, falling back to CPU")
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
        if gpu_devices: for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
        # Test basic TensorFlow operations
        x = tf.random.normal((5, 5))
        y = tf.matmul(x, x, transpose_b=True)
        self.assertEqual(y.shape, (5, 5), "TensorFlow basic operations failed")
        
        def test_mixed_precision(:
        self
        ): -> None: """Test mixed precision support"""
        accelerator = Accelerator(mixed_precision="fp16")
        self.assertEqual(accelerator.mixed_precision, "fp16", "Mixed precision not properly configured", )

def test_model_loading(:
    self
    ): -> None: """Test if environment can load and initialize models"""
        try:
        # Use a small, publicly available model
        model_name = "gpt2"  # Using smallest GPT-2 for testing
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        self.assertIsNotNone(tokenizer, "Failed to load tokenizer")
        self.assertIsNotNone(model, "Failed to load model")
        
        # Test basic inference
        text = "Hello, world!"
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
        outputs = model.generate(**inputs, max_length=20)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.assertTrue(len(generated_text) > 0, "Model failed to generate text")
        except Exception as e: self.fail(f"Failed to load model components: {str(e)}")
        
        def test_mmlu_dataset_access(:
        self
        ): -> None: """Test if environment can access MMLU dataset"""
        try:
            # Try loading high school mathematics dataset
            dataset_hs = load_dataset("cais/mmlu", "high_school_mathematics", split="validation[:10]")
            self.assertIsNotNone(dataset_hs, "Failed to load high school mathematics dataset")
            self.assertTrue(len(dataset_hs) > 0, "High school mathematics dataset is empty"
            )

            # Try loading college mathematics dataset
            dataset_college = load_dataset("cais/mmlu", "college_mathematics", split="validation[:10]")
            self.assertIsNotNone(dataset_college, "Failed to load college mathematics dataset")
            self.assertTrue(len(dataset_college) > 0,
            "College mathematics dataset is empty",
            )

            # Check dataset structure using high school dataset
            example = dataset_hs[0]
            required_keys = ["question", "choices", "answer"]
            for key in required_keys: self.assertIn(key, example, f"Dataset missing required key: {key}")
                except Exception as e: self.fail(f"Failed to access MMLU dataset: {str(e)}")

def test_flax_functionality(:
    self
    ): -> None: """Test Flax functionality"""
        try:
        # Test basic Flax operations
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (5, 5))
        self.assertEqual(x.shape, (5, 5), "Flax array operations not working")
        
        # Test basic model operations
        def model_fn(:
        x
        ): -> None: return jnp.mean(x)
        
        grad_fn = jax.grad(model_fn)
        grad = grad_fn(x)
        self.assertEqual(grad.shape, (5, 5), "Flax gradient computation not working"
        )
        except Exception as e: self.fail(f"Failed to test Flax functionality: {str(e)}")
        
        
        if __name__ == "__main__":
        unittest.main()
        