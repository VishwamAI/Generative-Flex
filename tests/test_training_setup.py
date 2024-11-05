import os
import unittest
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTrainingSetup(unittest.TestCase):
    @classmethod
def setUpClass(cls):
        """Set up test environment""""
        cls.accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="fp16" if torch.cuda.is_available() else "no","
        )
        cls.device = cls.accelerator.device
        # Initialize tokenizer for MMMU dataset using an open source model
        logger.info("Initializing tokenizer...")"
        cls.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")"
        logger.info("Tokenizer initialized successfully")"

def test_environment_setup(self):
        """Test basic environment setup""""
        # Test CUDA availability
        print(f"CUDA available: {torch.cuda.is_available()}")"
        print(f"Device being used: {self.device}")"
        self.assertTrue(hasattr(self.accelerator, "gradient_accumulation_steps"))"

def test_mmmu_dataset_loading(self):
        """Test MMMU dataset loading and processing""""
        try:
        # Test MMMU dataset loading with the new dataloader
from src.data.mmmu_dataloader import create_mmmu_dataloaders

        # Create dataloaders
        dataloaders = create_mmmu_dataloaders(
        subjects=["Math", "Physics", "Computer_Science"],"
        tokenizer=self.tokenizer,
        batch_size=4,  # Small batch size for testing
        )

        self.assertGreater(len(dataloaders), 0, "No dataloaders were created")"

        # Test at least one split
        for split, dataloader in dataloaders.items():
        print(f"\nTesting {split} split dataloader:")"
        self.assertIsNotNone(
        dataloader, f"Dataloader for {split} split is None""
        )

        # Get one batch
        batch = next(iter(dataloader))

        if self.tokenizer:  # If using tokenizer
        self.assertIn("input_ids", batch)"
        self.assertIn("attention_mask", batch)"
        self.assertIn("labels", batch)"
        else:  # If using raw data
        self.assertIn("question", batch)"
        self.assertIn("options", batch)"
        self.assertIn("answer", batch)"

        self.assertIn("images", batch)"
        self.assertIn("metadata", batch)"
        print(f"Successfully loaded batch from {split} split")"
    break  # Test only one split

        except Exception as e:
        self.fail(f"Failed to load MMMU dataset: {str(e)}")"

def test_gradient_accumulation(self):
        """Test gradient accumulation setup""""
        # Create a simple model for testing
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.AdamW(model.parameters())

        # Prepare model with accelerator
        model, optimizer = self.accelerator.prepare(model, optimizer)

        # Test gradient accumulation steps
        self.assertEqual(
        self.accelerator.gradient_accumulation_steps,
        4,
        "Gradient accumulation steps not set correctly","
        )

        # Simulate training with gradient accumulation
        for i in range(4):
        with self.accelerator.accumulate(model):
        inputs = torch.randn(2, 10).to(self.device)
        outputs = model(inputs)
        loss = outputs.mean()
        self.accelerator.backward(loss)

        if self.accelerator.sync_gradients:
        self.assertEqual(i, 3, "Sync gradients at wrong step")"
        optimizer.step()
        optimizer.zero_grad()

def test_mixed_precision(self):
        """Test mixed precision setup""""
        if torch.cuda.is_available():
        self.assertEqual(
        self.accelerator.mixed_precision,
        "fp16","
        "Mixed precision not set correctly for GPU","
        )
        else:
        self.assertEqual(
        self.accelerator.mixed_precision,
        "no","
        "Mixed precision should be disabled for CPU","
        )

def test_model_device_compatibility(self):
        """Test model compatibility with current device""""
        # Create a simple model
        model = torch.nn.Linear(10, 1)
        model = self.accelerator.prepare(model)

        # Check if model is on correct device
        self.assertEqual(
        next(model.parameters()).device,
        self.device,
        "Model not on correct device","
        )

        # Test forward pass
        inputs = torch.randn(2, 10).to(self.device)
        try:
        outputs = model(inputs)
        self.assertEqual(
        outputs.device, self.device, "Output not on correct device""
        )
        except Exception as e:
        self.fail(f"Forward pass failed: {str(e)}")"


if __name__ == "__main__":"
    unittest.main(verbosity=2)
