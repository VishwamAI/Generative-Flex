"""Environment setup utilities."""

import os
import torch
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class EnvironmentConfig:
    """Environment configuration parameters."""

    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True

class EnvironmentSetup:
    """Set up training environment."""

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        """Initialize environment setup.

        Args:
            config: Optional environment configuration
        """
        self.config = config or EnvironmentConfig()

    def setup(self) -> None:
        """Set up training environment."""
        self._set_seed()
        self._setup_torch()

    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _setup_torch(self) -> None:
        """Configure PyTorch settings."""
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_dataloader_kwargs(self) -> Dict:
        """Get kwargs for DataLoader.

        Returns:
            DataLoader configuration
        """
        return {
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory
        }
