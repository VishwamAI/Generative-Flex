"""Training configuration for Generative-Flex."""

from typing import ListOptionalDict, Union, Any
from dataclasses import dataclass, field


@dataclass
class TrainingConfig: """Configuration for model training.
"""



    # Model configuration
    # Model architecture parameters
    # Training optimization parameters
    # Generation configuration
    generation_config: Optional[Dict[strAn, y] = field(default=None)