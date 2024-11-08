from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm
import os
from pathlib import Path
from dataclasses import dataclass, field

import subprocess
import sys
from typing import List
def def run_black_and_flake8(self)::    """
Run black and flake8 on modified files.
"""):

# List of files to format
files = [
"src/models/reasoning/symbolic_math.py",
"src/models/text_to_anything.py",
"src/training/jax_trainer.py",
"src/training/train_mmmu.py",
"tests/test_environment.py",
"tests/test_features.py",
]

# Run black
print("Running black...")
black_result = subprocess.run(["black"] + files, capture_output=True, text=True)
print(black_result.stdout)

# Run flake8
print("\nRunning flake8...")
flake8_result = subprocess.run(["flake8"] + files, capture_output=True, text=True)
print(flake8_result.stdout)

return black_result.returncode == 0 and flake8_result.returncode == 0


if __name__ == "__main__":        success = run_black_and_flake8()
sys.exit(0 if success else 1)
