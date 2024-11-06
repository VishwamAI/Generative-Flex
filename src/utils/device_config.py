"""Device configuration utilities."""

import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class DeviceConfig:
    """Device configuration parameters."""

    use_cuda: bool = True
    device_id: int = 0
    use_amp: bool = True

class DeviceManager:
    """Manage device configuration and placement."""

    def __init__(self, config: Optional[DeviceConfig] = None):
        """Initialize device manager.

        Args:
            config: Optional device configuration
        """
        self.config = config or DeviceConfig()
        self.device = self._setup_device()

    def _setup_device(self) -> torch.device:
        """Set up compute device.

        Returns:
            Configured device
        """
        if self.config.use_cuda and torch.cuda.is_available():
            return torch.device(f"cuda:{self.config.device_id}")
        return torch.device("cpu")

    def place_on_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Place tensor on configured device.

        Args:
            tensor: Input tensor

        Returns:
            Tensor on configured device
        """
        return tensor.to(self.device)
