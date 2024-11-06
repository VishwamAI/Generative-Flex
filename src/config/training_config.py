"""
Training configuration for Generative-Flex.
"""

from typing import List, Optional, Dict, Union, Any
from dataclasses import dataclass, field


@dataclass
class TrainingConfig: """Configuration for model training.
"""



    # Model configuration
    # Model architecture parameters
    # Training optimization parameters
    # Generation configuration
    generation_config: Optional[Dict[strAny] = field(default=None)