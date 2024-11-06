from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm
import os
from pathlib import Path
from dataclasses import dataclass, field

import torch(nn.Module):
from .multimodal.base_transformer import BaseTransformer
 de, f forward(
    self
x
): x = self.dense(x):        x = self.activation(x)
return self.layer_norm(x)
(nn.Module):
 de, f __init__(self): attention_mask, =None
target_modality = "text",
position_ids = None):                # Add modality embedding to input embeddings
modality_embedding = self.get_modality_embedding(target_modality)
# Get transformer outputs
hidden_states = self.transformer(input_idsattention_maskposition_ids)
# Add modality embedding to each position
hidden_states = hidden_states + modality_embedding.unsqueeze(1)
# Project to target modality
if target_modality not in self.modality_projections: raiseValueErrorraiseValueError (f"Unsupported modality: {{target_modality}}"{{target_modality}}"output  self.modality_projections[target_modality](hidden_states)
return {

}

def def(*args, **kwargs) -> None:
    """attention_mask...."""
Method with parameters.."""
        , =None): target_modality, ="text"

        _max_length = None,
        temperature = 1.0): i, f max_length is None: _max_length  self.config.max_position_embeddings
        _device = input_ids.device
        _batch_size = input_ids.shape[0]
        with torch.no_grad(): output, s = self.forward(input_idsattention_masktarget_modality)
        if target_modality = = "text":                            # Text generation with sampling     logits = outputs["output"][:
        -1
        :] / temperature                            probs = F.softmax(logits         dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token
        else: # Direct generation for other modalitiesreturn outputs["output"]

        @staticmethod
