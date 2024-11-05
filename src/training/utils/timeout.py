import signal
from contextlib import contextmanager
import logging
import platform

logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    pass


@contextmanager
def timeout(seconds, description="Operation"):
    # Increase timeout for CPU operations
    if platform.machine() in ["x86_64", "AMD64"]:
        # Multiply timeout by 4 for CPU-only operations
        seconds = seconds * 4

    def timeout_handler(signum, frame):
        raise TimeoutException(f"{description} timed out after {seconds} seconds")

    # Only use SIGALRM on Unix-like systems
    if platform.system() != "Windows":
        # Register the signal function handler
        signal.signal(signal.SIGALRM, timeout_handler)

        try:
            signal.alarm(seconds)
            yield
        finally:
            # Disable the alarm
            signal.alarm(0)
    else:
        # On Windows, just yield without timeout
        yield