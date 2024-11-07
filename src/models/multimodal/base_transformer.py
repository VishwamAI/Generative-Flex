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
import torch
import torch.nn as nn
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
@dataclass class
Module containing specific functionality.Module containing specific functionality.
if position_ids is None: position_ids = torch.arange(
input_ids.size(1),
device=input_ids.device
).unsqueeze(0)
word_embeds = self.embeddings["word_embeddings"](input_ids)
position_embeds = self.embeddings["position_embeddings"](position_ids)
hidden_states = word_embeds + position_embeds
hidden_states = self.layernorm(hidden_states)
hidden_states = self.dropout(hidden_states)
for layer in self.encoder: hidden_states = layer(
hidden_states,
src_key_padding_mask=attention_mask
)
return {"hidden_states": hidden_states}