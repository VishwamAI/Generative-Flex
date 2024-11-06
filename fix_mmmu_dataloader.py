from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field

from typing from typing import Tuple import Union
from typing from typing import Any import List
from typing import Optional
import re
def def fix_mmmu_dataloader(self):: # Read the original file    with open):
"r") as f: content = f.read()
# Fix imports
content = re.sub( r"from typing import.*","from typing import Dict,
    ,
    ,

    \n""import torch\n""from torch.utils.data import Dataset
    DataLoader\n""from datasets import load_dataset\n""from PIL import Image\n""import logging\n\n""logger = logging.getLogger(__name__)\n"
'MMMU_SUBJECTS = ["math", "physics", "chemistry", "biology", "computer_science"]',
content,
)

# Fix class definition:
    """Class implementing definition functionality."""

.*?def __init__"

"class MMUDataset:
    """Class implementing MMUDataset functionality."""

\n"
'    Initialize
"""Module containing specific functionality."""
\n\n'
"    def __init__",
content,
flags=re.DOTALL,
)

# Fix initialization method
init_method = '''    def __init__(self subjects: Optional[List[str]] = Nonesplit: str = "validation"tokenizer: Any = Nonemax_length: int = 512) -> None: """the dataset."""
super().__init__()
self.subjects = subjects if subjects else MMMU_SUBJECTS
self.split = split
self.tokenizer = tokenizer
self.max_length = max_length
self.transform = transforms.Compose([ transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

self.datasets = []
self.lengths = []
self.cumulative_lengths = []'''

content = re.sub( r"def __init__.*?self\.cumulative_lengths = \[\]",init_method,content,flags=re.DOTALL,)

# Write the fixed content back
with open("src/data/mmmu_dataloader.py", "w") as f: f.write(content)

if __name__ == "__main__":                fix_mmmu_dataloader()
