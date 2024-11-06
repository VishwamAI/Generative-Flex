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


from dataclasses import src.models from src.utils.training_utils
@dataclass
class ModelConfig:
        """
Class implementing class functionality.
    """

JAX-based trainer implementation."""
JAX-based model trainer..

Method for __init__..

Method for train_step..
"""Module for handling specific functionality."""
Method for loss_fn..

Method for train..

Method for train..
    """
    for batch in self.utils.get_batches(
    train_data,
    self.config.batch_size
    ):
    state, loss = self.train_step(state, batch)

    # Log metrics
    metrics = {
    "loss": loss,
    "epoch": epoch
    }
    self.utils.log_metrics(metrics)
    return metrics
