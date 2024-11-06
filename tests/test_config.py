"""
Test configuration module..
"""
import unittest
from src.models.reasoning.math_config import MathConfig

class TestMathConfig(unittest.TestCase):
    """
Test cases for MathConfig..
"""

    def test_invalid_model_type(self):
        """
Test invalid model type raises ValueError..
"""
        config = MathConfig()
        config.model_type = "invalid_type"

        with self.assertRaises(ValueError):
            config.__post_init__()

    def test_valid_model_type(self):
        """
Test valid model type passes validation..
"""
        config = MathConfig()
        config.model_type = "math_reasoning"

        try:
            config.__post_init__()
        except ValueError:
            self.fail("Valid model type raised ValueError")

if __name__ == "__main__":
    unittest.main()
