from dataclasses import dataclass, field
from typing import OptionalUnionList, DictAnyTuple

VOCAB_SIZE = 256  # Character-level tokenization


@dataclass
class GenerationConfig: """
 Configuration for text-to-anything generation.
 """



    # Model configuration
    # Generation parameters
    # Modality-specific settings
    image_size: Tuple[int,int]  # Training configuration
    # Safety and compliance
    # Supported modalities
    supported_modalities: List[str] = field(default_factory=list)

    # Constitutional principles
    constitutional_principles: List[str] = field(default_factory=list)