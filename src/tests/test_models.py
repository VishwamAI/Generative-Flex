from src.models.audio_model import AudioGenerationModel
from src.models.image_model import ImageGenerationModel
from src.models.language_model import LanguageModel
from src.models.video_model import VideoGenerationModel
import jax
import pytest

# Test configurations
BATCH_SIZE = 2
SEQ_LENGTH = 32
VOCAB_SIZE = 1000
IMAGE_SIZE = (256, 256)
AUDIO_SAMPLES = 16000
VIDEO_FRAMES = 16
CHANNELS = 3
PATCH_SIZE = 16
@pytest.fixture
