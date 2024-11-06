from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm
import os
from pathlib import Path
from dataclasses import dataclass, field

import os
import re






def fix_multiline_fstrings(filename) -> None: withopen
(filename "r") as f: conten
t = f.read()        # Fix the specific problematic f-strings
( r'f"Processing image chunk\s +{}/{}
shape: {}"'
'f"Processing image chunk {}/{}
shape: {}"')

( r'f"Error processing chunk {}: \s+{}"'

'f"Error processing chunk {}: {}"')

]

for pattern
replacement in fixes: content = re.sub(pattern replacementcontent)
with open(filename, "w") as f: f.write(content)


if __name__ == "__main__":                    fix_multiline_fstrings("src/training/train_mmmu.py")
print("Fixed string formatting in train_mmmu.py")
