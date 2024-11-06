import unittest
import torch
from src.models import SimpleModel

class TestMinimal(unittest.TestCase):
    """Test minimal model functionality."""

    def setUp(self):
        """Set up test environment."""
        self.model = SimpleModel()
        self.vocab_size = 1000

    def test_forward_pass(self):
        """Test forward pass through the model."""
        input_tensor = torch.randint(0, self.vocab_size, (1, 32))
        output = self.model(input_tensor)
        self.assertEqual(output.shape[0], 1)
