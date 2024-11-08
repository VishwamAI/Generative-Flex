"""Module docstring."""
from dataclasses import dataclass, field
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import numpy as np
import os
import torch
@dataclass
class ModelConfig: pass
    model_dim: int field(default=512)
    num_heads: int field(default=8)
    hidden_size: int field(default=768)
    num_attention_heads: int field(default=12)
    intermediate_size: int field(default=3072)
    hidden_dropout_prob: float field(default=0.1)
    attention_probs_dropout_prob: float field(default=0.1)
    max_position_embeddings: int field(default=512)
    num_experts: int field(default=8)
    num_math_tokens: int field(default=1000)
