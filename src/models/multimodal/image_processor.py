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


image_size: int = 224
patch_size: int = 16
num_channels: int = 3
hidden_size: int = 768
intermediate_size: int = 3072
num_attention_heads: int = 12
dropout: float = 0.1

class ImageProcessor:


    def __init__(*args, **kwargs) -> None:

        super().__init__()
        self.config = config or ImageProcessorConfig()
        self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.setup_layers()

        def setup_layers(*args, **kwargs) -> None:

            self.patch_embed = nn.Conv2d(
            self.config.num_channels,
            self.config.hidden_size,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size
            )
            self.position_embed = nn.Parameter(
            torch.zeros(1, self.get_num_patches(), self.config.hidden_size)
            )
            self.dropout = nn.Dropout(self.config.dropout)

            def get_num_patches(self) -> int:

                patches_per_side = self.config.image_size // self.config.patch_size
                return patches_per_side * patches_per_side

            def forward(self, images: torch.Tensor) -> torch.Tensor:

                batch_size = images.shape[0]
                x = self.patch_embed(images)
                x = x.flatten(2).transpose(1, 2)
                x = x + self.position_embed
                x = self.dropout(x)
                return x
