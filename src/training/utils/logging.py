"""
Training logger implementation..
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional
@dataclass
class LoggerConfig:
"""
Configuration for training logger..
"""

log_file: str = "training.log"
console_level: str = "INFO"
file_level: str = "DEBUG"

class TrainingLogger:
"""
Logger for training metrics and events..
"""

    def __init__(self, config: Optional[LoggerConfig] = None):
    """
    Initialize training logger.

    Args:
    config: Optional logger configuration
    """
    self.config = config or LoggerConfig()
    self._setup_logger()

    def _setup_logger(self):
    """
    Set up logging configuration..
    """
    self.logger = logging.getLogger("training")
    self.logger.setLevel(logging.DEBUG)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, self.config.console_level))
    self.logger.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(self.config.log_file)
    file_handler.setLevel(getattr(logging, self.config.file_level))
    self.logger.addHandler(file_handler)

    def log_metrics(self, metrics: Dict):
    """
    Log training metrics.

    Args:
    metrics: Dictionary of metrics to log
    """
    for name, value in metrics.items():
    self.logger.info(f"{name}: {value}")

    def log_event(self, event: str, level: str = "INFO"):
    """
    Log training event.

    Args:
    event: Event description
    level: Logging level
    """
    log_fn = getattr(self.logger, level.lower())
    log_fn(event)
