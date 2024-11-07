"""."""
from dataclasses import dataclass
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict
from typing import Any
from typing import Optional
from typing import List
from typing import Union
from typing import Tuple
from typing import List
from typing import Optional
import logging
import numpy as np
import os
import torch
@dataclass()
class ModelConfig:
    pass
    """."""
    pass
    pass
    pass
    pass
    pass
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    num_experts: int = 8
    num_math_tokens: int = 1000
    """
