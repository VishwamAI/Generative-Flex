from src.config.config import Config, ModelConfig, TrainingConfig, get_config
import pytest

"""Tests for configuration management."""


def test_invalid_model_type(self):
    """Test handling of invalid model type."""


with pytest.raises(ValueError):
    get_config("invalid_type")
