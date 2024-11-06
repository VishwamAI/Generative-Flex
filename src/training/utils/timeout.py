"""Timeout utilities for training.."""

import signal
from dataclasses import dataclass
from typing import Optional, Callable

@dataclass
class TimeoutConfig:
    """Configuration for timeout handler.."""

    timeout_seconds: int = 3600
    callback: Optional[Callable] = None

class TimeoutError(Exception):
    """Exception raised when timeout occurs.."""
    pass

class TimeoutHandler:
    """Handler for training timeouts.."""

    def __init__(self, config: Optional[TimeoutConfig] = None):
        """Initialize timeout handler.

        Args:
            config: Optional timeout configuration"""
        self.config = config or TimeoutConfig()

    def __enter__(self):
        """Set up timeout handler.."""
        def handler(signum, frame):
            if self.config.callback:
                self.config.callback()
            raise TimeoutError("Training timed out")

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.config.timeout_seconds)

    def __exit__(self, type, value, traceback):
        """Clean up timeout handler.."""
        signal.alarm(0)
