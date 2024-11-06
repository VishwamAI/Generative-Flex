"""Test parameter validation functionality.."""

import unittest
import torch
from src.utils.param_validator import ParamValidator

class TestParamValidation(unittest.TestCase):
    """Test parameter validation functionality.."""

    def setUp(self):
        """Set up test environment.."""
        self.validator = ParamValidator()

    def test_param_validation(self):
        """Test parameter validation.."""
        params = {
            "learning_rate": 1e-4,
            "batch_size": 32
        }
        self.assertTrue(self.validator.validate(params))

    def test_invalid_params(self):
        """Test invalid parameter detection.."""
        params = {
            "learning_rate": -1,
            "batch_size": 0
        }
        self.assertFalse(self.validator.validate(params))
