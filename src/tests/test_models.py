import pytest
import jax
import jax.numpy as jnp
from src.models.language_model import LanguageModel
from src.models.image_model import ImageGenerationModel
from src.models.audio_model import AudioGenerationModel
from src.models.video_model import VideoGenerationModel

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
def language_model():
    return LanguageModel(
        vocab_size=VOCAB_SIZE,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        head_dim=32,
        mlp_dim=512,
        max_seq_len=SEQ_LENGTH,
    )


@pytest.fixture
def image_model():
    return ImageGenerationModel(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        head_dim=32,
        mlp_dim=512,
    )


@pytest.fixture
def audio_model():
    return AudioGenerationModel(
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        head_dim=32,
        mlp_dim=512,
        frame_size=1024,
        hop_length=256,
    )


@pytest.fixture
def video_model():
    return VideoGenerationModel(
        video_size=(VIDEO_FRAMES, *IMAGE_SIZE),
        patch_size=(2, PATCH_SIZE, PATCH_SIZE),
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        head_dim=32,
        mlp_dim=512,
    )


def test_language_model_init(language_model):
    rng = jax.random.PRNGKey(0)
    input_ids = jnp.ones((BATCH_SIZE, SEQ_LENGTH), dtype=jnp.int32)

    variables = language_model.init(rng, input_ids, training=False)
    assert variables is not None


def test_language_model_forward(language_model):
    rng = jax.random.PRNGKey(0)
    input_ids = jnp.ones((BATCH_SIZE, SEQ_LENGTH), dtype=jnp.int32)

    variables = language_model.init(rng, input_ids, training=False)
    output = language_model.apply(
        variables, input_ids, training=False, rngs={"dropout": rng}
    )

    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE)


def test_language_model_training(language_model):
    rng = jax.random.PRNGKey(0)
    input_ids = jnp.ones((BATCH_SIZE, SEQ_LENGTH), dtype=jnp.int32)

    init_rng, dropout_rng = jax.random.split(rng)
    variables = language_model.init(init_rng, input_ids, training=True)
    output = language_model.apply(
        variables, input_ids, training=True, rngs={"dropout": dropout_rng}
    )

    # Check training mode output
    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE)
    # Ensure gradients can flow (no NaNs)
    assert not jnp.any(jnp.isnan(output))


def test_image_model_init(image_model):
    rng = jax.random.PRNGKey(0)
    images = jnp.ones((BATCH_SIZE, *IMAGE_SIZE, CHANNELS), dtype=jnp.float32)

    variables = image_model.init(rng, images, training=False)
    assert variables is not None


def test_image_model_forward(image_model):
    rng = jax.random.PRNGKey(0)
    images = jnp.ones((BATCH_SIZE, *IMAGE_SIZE, CHANNELS), dtype=jnp.float32)

    variables = image_model.init(rng, images, training=False)
    output = image_model.apply(variables, images, training=False, rngs={"dropout": rng})

    assert output.shape == (BATCH_SIZE, *IMAGE_SIZE, CHANNELS)


def test_image_model_training(image_model):
    rng = jax.random.PRNGKey(0)
    images = jnp.ones((BATCH_SIZE, *IMAGE_SIZE, CHANNELS), dtype=jnp.float32)

    init_rng, dropout_rng = jax.random.split(rng)
    variables = image_model.init(init_rng, images, training=True)
    output = image_model.apply(
        variables, images, training=True, rngs={"dropout": dropout_rng}
    )

    assert output.shape == (BATCH_SIZE, *IMAGE_SIZE, CHANNELS)
    assert not jnp.any(jnp.isnan(output))


def test_audio_model_init(audio_model):
    rng = jax.random.PRNGKey(0)
    audio = jnp.ones((BATCH_SIZE, AUDIO_SAMPLES), dtype=jnp.float32)

    variables = audio_model.init(rng, audio, training=False)
    assert variables is not None


def test_audio_model_forward(audio_model):
    rng = jax.random.PRNGKey(0)
    audio = jnp.ones((BATCH_SIZE, AUDIO_SAMPLES), dtype=jnp.float32)

    variables = audio_model.init(rng, audio, training=False)
    output = audio_model.apply(variables, audio, training=False, rngs={"dropout": rng})

    # Account for frame size and hop length in output shape
    expected_samples = (
        (AUDIO_SAMPLES - audio_model.frame_size) // audio_model.hop_length + 1
    ) * audio_model.hop_length
    assert output.shape == (BATCH_SIZE, expected_samples)


def test_audio_model_training(audio_model):
    rng = jax.random.PRNGKey(0)
    audio = jnp.ones((BATCH_SIZE, AUDIO_SAMPLES), dtype=jnp.float32)

    init_rng, dropout_rng = jax.random.split(rng)
    variables = audio_model.init(init_rng, audio, training=True)
    output = audio_model.apply(
        variables, audio, training=True, rngs={"dropout": dropout_rng}
    )

    expected_samples = (
        (AUDIO_SAMPLES - audio_model.frame_size) // audio_model.hop_length + 1
    ) * audio_model.hop_length
    assert output.shape == (BATCH_SIZE, expected_samples)
    assert not jnp.any(jnp.isnan(output))


def test_video_model_init(video_model):
    rng = jax.random.PRNGKey(0)
    video = jnp.ones(
        (BATCH_SIZE, VIDEO_FRAMES, *IMAGE_SIZE, CHANNELS), dtype=jnp.float32
    )

    variables = video_model.init(rng, video, training=False)
    assert variables is not None


def test_video_model_forward(video_model):
    rng = jax.random.PRNGKey(0)
    video = jnp.ones(
        (BATCH_SIZE, VIDEO_FRAMES, *IMAGE_SIZE, CHANNELS), dtype=jnp.float32
    )

    variables = video_model.init(rng, video, training=False)
    output = video_model.apply(variables, video, training=False, rngs={"dropout": rng})

    assert output.shape == (BATCH_SIZE, VIDEO_FRAMES, *IMAGE_SIZE, CHANNELS)


def test_video_model_training(video_model):
    rng = jax.random.PRNGKey(0)
    video = jnp.ones(
        (BATCH_SIZE, VIDEO_FRAMES, *IMAGE_SIZE, CHANNELS), dtype=jnp.float32
    )

    init_rng, dropout_rng = jax.random.split(rng)
    variables = video_model.init(init_rng, video, training=True)
    output = video_model.apply(
        variables, video, training=True, rngs={"dropout": dropout_rng}
    )

    assert output.shape == (BATCH_SIZE, VIDEO_FRAMES, *IMAGE_SIZE, CHANNELS)
    assert not jnp.any(jnp.isnan(output))
