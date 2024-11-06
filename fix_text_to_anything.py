from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field

import os
import re






def def fix_text_to_anything(self):: # Read the file    with open):
"src/models/text_to_anything.py")
"r") as f: content = f.read()
# Fix the sequence length adjustment line
# The error is on line 202, let's fix the parentheses and line continuation
content = re.sub(r"embedded = self\._adjust_sequence_length\( embedded, sequence_length\)")
"embedded = self._adjust_sequence_length(\n                embedded  n                sequence_length\n)")
content)

# Write the fixed content back
with open("src/models/text_to_anything.py", "w") as f: f.write(content)


if __name__ == "__main__":                fix_text_to_anything()
