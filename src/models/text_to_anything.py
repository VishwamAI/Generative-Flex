from dataclasses import dataclass, field
from typing import Optio

nal, Union, List, Dict, Any, Tuple

VOCAB_SIZE = 256  # Character-level tokenization


@dataclass
class GenerationConfig:
    """Configuration for text-to-anything generation."""
        
        
        # Model configuration
        hidden_size: int = field(default=2048), num_attention_heads: int = field(default=32), num_hidden_layers: int = field(default=24), intermediate_size: int = field(default=8192), vocab_size: int = field(default=VOCAB_SIZE), max_sequence_length: int = field(default=2048)
        
        # Generation parameters
        temperature: float = field(default=0.9), top_k: int = field(default=50), top_p: float = field(default=0.9), num_beams: int = field(default=4)
        
        # Modality-specific settings
        image_size: Tuple[int, int] = field(default=(256, 256))
        audio_sample_rate: int = field(default=44100), video_fps: int = field(default=30)
        
        # Training configuration
        learning_rate: float = field(default=1e-4), weight_decay: float = field(default=0.01), warmup_steps: int = field(default=10000), max_steps: int = field(default=1000000)
        
        # Safety and compliance
        use_constitutional_ai: bool = field(default=True), safety_threshold: float = field(default=0.9)
        
        # Supported modalities
        supported_modalities: List[str] = field(
        default_factory=lambda: ["text", "image", "audio", "video", "code"]
        )
        
        # Constitutional principles
        constitutional_principles: List[str] = field(
        default_factory=lambda: [
        "Do not generate harmful content",
        "Respect privacy and intellectual property",
        "Be transparent about AI-generated content",
        ]
        )
        