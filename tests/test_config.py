import pytest
from src.config.config import Config, ModelConfig, TrainingConfig, get_config

"""Tests for configuration management."""


def test_model_config():
    """Test model configuration."""
    config = ModelConfig(model_type="language")
    assert config.vocab_size == 50257
    assert config.hidden_dim == 2048
    assert config.num_heads == 32


def test_training_config():
    """Test training configuration."""
    config = TrainingConfig()
    assert config.batch_size == 32
    assert config.learning_rate == 1e-4
    assert config.num_epochs == 100


def test_get_config():
    """Test getting configurations for different model types."""
    # Language model
    lang_config = get_config("language")
    assert lang_config.model.model_type == "language"
    assert lang_config.model.vocab_size == 50257

    # Image model
    img_config = get_config("image")
    assert img_config.model.model_type == "image"
    assert img_config.model.image_size == (256, 256)
    assert img_config.model.patch_size == (16, 16)

    # Audio model
    audio_config = get_config("audio")
    assert audio_config.model.model_type == "audio"
    assert audio_config.model.audio_sample_rate == 16000
    assert audio_config.model.frame_size == 1024

    # Video model
    video_config = get_config("video")
    assert video_config.model.model_type == "video"
    assert video_config.model.video_size == (16, 256, 256)
    assert video_config.model.video_patch_size == (2, 16, 16)


def test_config_serialization(tmp_path):
    """Test configuration serialization."""
    config = get_config("language")
    config_path = tmp_path / "config.json"

    # Save config
    config.to_json(str(config_path))

    # Load config
    loaded_config = Config.from_json(str(config_path))

    # Verify loaded config matches original
    assert loaded_config.model.model_type == config.model.model_type
    assert loaded_config.model.vocab_size == config.model.vocab_size
    assert loaded_config.training.batch_size == config.training.batch_size


def test_invalid_model_type():
    """Test handling of invalid model type."""
    with pytest.raises(ValueError):
        get_config("invalid_type")
