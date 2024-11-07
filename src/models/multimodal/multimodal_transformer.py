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
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import torch.nn as nn
from dataclasses import dataclass
@dataclass class
Module for implementing specific functionality.Module containing specific functionality.Module containing specific functionality.
hidden_states_list = []
if input_ids is not None: position_ids = torch.arange(:
    input_ids.size(1),
    device=input_ids.device
    ).unsqueeze(0)
    word_embeds = self.text_embeddings["word_embeddings"](input_ids)
    position_embeds = self.text_embeddings["position_embeddings"](
    position_ids
    )
    text_hidden_states = word_embeds + position_embeds
    text_hidden_states = self.layernorm(text_hidden_states)
    text_hidden_states = self.dropout(text_hidden_states)
    hidden_states_list.append(text_hidden_states)
    if pixel_values is not None:
        B, C, H, W = pixel_values.shape
        P = self.config.patch_size
        patches = pixel_values.unfold(2, P, P).unfold(3, P, P)
        patches = patches.contiguous().view(
        B, C, -1, P * P
        ).transpose(1, 2)
        patches = patches.reshape(B, -1, C * P * P)
        patch_embeds = self.image_embeddings["patch_embeddings"](patches)
        position_ids = torch.arange(
        patches.size(1),
        device=patches.device
        ).unsqueeze(0)
        position_embeds = self.image_embeddings["position_embeddings"](
        position_ids
        )
        image_hidden_states = patch_embeds + position_embeds
        image_hidden_states = self.layernorm(image_hidden_states)
        image_hidden_states = self.dropout(image_hidden_states)
        hidden_states_list.append(image_hidden_states)
        if hidden_states_list: hidden_states = torch.cat(hidden_states_list, dim=1):
            if attention_mask is not None and pixel_mask is not None: attention_mask = torch.cat(:
                [attention_mask, pixel_mask],
                dim=1
                )
                for layer in self.encoder: hidden_states = layer(
                hidden_states,
                src_key_padding_mask=attention_mask
                )
                return {"hidden_states": hidden_states}
            return {"hidden_states": None}