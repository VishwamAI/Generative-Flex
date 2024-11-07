"""."""
from typing import Dict
from typing import Any
from typing import Optional
from typing import List
from typing import Union
from typing import Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import logging
from tqdm import tqdm
import os
from pathlib import Path
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass
@dataclass
class ():
    Module containing specific functionality.Module containing specific functionality.
    residual = hidden_states
    hidden_states = self.layer_norm1(hidden_states)
    hidden_states, _ = self.attention(
    hidden_states,
    hidden_states,
    hidden_states,
    key_padding_mask=attention_mask
    )
    hidden_states = self.dropout(hidden_states)
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.feed_forward(hidden_states)
    hidden_states = residual + hidden_states
    return {"hidden_states": hidden_states}