"""
Timeout utilities for training..
"""
import signal
from dataclasses import dataclass
from typing import Optional, Callable
@dataclass
class TimeoutConfig:

    """Configuration for timeout handler..
"""

timeout_seconds: int = 3600
callback: Optional[Callable] = None

class TimeoutError:
"""
Exception raised when timeout occurs..
"""
pass

class TimeoutHandler:
"""
Handler for training timeouts..
"""

    def __init__(self, config: Optional[TimeoutConfig] = None):


        """Method for __init__."""
    self.config = config or TimeoutConfig()

    def __enter__(self):


        """Method for __enter__."""
        def handler(signum, frame):

            """Method for handler."""if self.config.callback:
        self.config.callback()
        raise TimeoutError("Training timed out")

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.config.timeout_seconds)

    def __exit__(self, type, value, traceback):


        """Method for __exit__."""
    signal.alarm(0)
