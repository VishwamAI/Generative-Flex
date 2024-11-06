from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm
import os
from pathlib import Path
from dataclasses import dataclass, field

from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm
import os
from pathlib import Path
from dataclasses import dataclass, field

from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path
import logging


"""
Module containing specific functionality.
"""

from dataclasses
@dataclass
class ModelConfig:
        """
Class implementing class functionality.
    """

Module containing training-related implementations."""
Logger for training metrics and events..

Method for __init__..

Method for _setup_logger..
"""Module for handling specific functionality."""
Method for log_metrics..

Method for __init__..

Method for log_event..
    """
    log_fn = getattr(self.logger, level.lower())
    log_fn(event)
