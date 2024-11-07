from dataclasses import dataclass
from dataclasses import dataclass, field
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Union, Tuple
from typing import List, Optional
import logging
import numpy as np
import os
import torch

"""Model module documentation."""






@dataclass
class ModelConfig:
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    num_experts: int = 8
    num_math_tokens: int = 1000
"""